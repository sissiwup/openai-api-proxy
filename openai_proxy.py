import requests
import argparse
import json
import time
import threading
import os
from flask import Flask, request, jsonify, Response
from waitress import serve

# 1. Initialisieren der Flask-Anwendung
app = Flask(__name__)

# 2. Default Ziel-URL (wird verwendet, wenn kein Port-Mapping passt)
DEFAULT_TARGET_URL = "https://api.openai.com"

# Mapping: eingehender Port -> Ziel-URL
# Beispiel: wenn der Request über Host:PORT kommt, wird das entsprechende Ziel verwendet.
# Sie können hier beliebig weitere Ports und Ziel-URLs hinzufügen.
PORT_TARGET_MAP = {
    # Host-Port : Ziel-URL
    "5101": "https://api.openai.com",
    "5102": "https://api.openai.com",  # Umsetzung V1 von chat/completions auf responses
}

LOG_FILE_PATH = None
LOG_LOCK = threading.Lock()
MAX_LOG_CHARS = 20000


def configure_logging(path):
    global LOG_FILE_PATH
    LOG_FILE_PATH = path
    if LOG_FILE_PATH:
        try:
            if os.path.exists(LOG_FILE_PATH):
                os.remove(LOG_FILE_PATH)
        except Exception as exc:
            print(f"WARNUNG: Log-Datei {LOG_FILE_PATH} konnte nicht gelöscht werden: {exc}")
        write_log_entry("system", "info", "-", f"Logging aktiviert: {LOG_FILE_PATH}")


def write_log_entry(port, direction, path, text):
    if not LOG_FILE_PATH:
        return
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    entry = f"[{timestamp}] port={port} direction={direction} path=/{path}\n{text}\n---\n"
    try:
        with LOG_LOCK:
            with open(LOG_FILE_PATH, 'a', encoding='utf-8') as log_file:
                log_file.write(entry)
    except Exception as exc:
        print(f"WARNUNG: Log-Datei {LOG_FILE_PATH} konnte nicht beschrieben werden: {exc}")


def log_payload(port, direction, path, payload):
    if not LOG_FILE_PATH:
        return
    if payload is None:
        text = "<no body>"
    elif isinstance(payload, bytes):
        text = payload.decode('utf-8', errors='replace')
    else:
        text = str(payload)
    if len(text) > MAX_LOG_CHARS:
        text = f"{text[:MAX_LOG_CHARS]}... [truncated {len(text) - MAX_LOG_CHARS} chars]"
    write_log_entry(port, direction, path or '-', text)


def convert_messages_to_responses_input(messages):
    """Wandelt OpenAI Chat-Completions Nachrichten in Responses-Eingaben um."""
    response_input = []
    for message in messages or []:
        if isinstance(message, dict):
            role = message.get('role', 'user')
            content = message.get('content', '')
        else:
            role = 'user'
            content = message
        normalized_content = []
        text_type = 'output_text' if role == 'assistant' else 'input_text'

        if isinstance(content, str):
            normalized_content.append({"type": text_type, "text": content})
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict):
                    block = dict(item)
                    block_type = block.get('type')
                    if block_type == 'text':
                        block['type'] = text_type
                    elif block_type in ('input_text', 'output_text'):
                        block['type'] = text_type
                    elif block_type == 'image_url':
                        new_block = {"type": "input_image"}
                        image_data = block.get('image_url')
                        detail = None
                        if isinstance(image_data, dict):
                            url_value = image_data.get('url')
                            if url_value:
                                new_block['image_url'] = url_value
                            detail = image_data.get('detail')
                        else:
                            if image_data:
                                new_block['image_url'] = image_data
                        if 'image_base64' in block:
                            new_block['image_base64'] = block['image_base64']
                        if 'file_id' in block:
                            new_block['file_id'] = block['file_id']
                        detail = detail or block.get('detail')
                        if detail:
                            new_block['detail'] = detail
                        block = new_block
                    elif block_type in ('file', 'document', 'input_file'):
                        new_block = {"type": "input_file"}
                        if 'file_id' in block:
                            new_block['file_id'] = block['file_id']
                        if 'filename' in block:
                            new_block['filename'] = block['filename']
                        if 'file_data' in block:
                            new_block['file_data'] = block['file_data']
                        block = new_block
                    normalized_content.append(block)
                elif isinstance(item, str):
                    normalized_content.append({"type": text_type, "text": item})
                else:
                    normalized_content.append({"type": text_type, "text": str(item)})
        elif content is None:
            pass
        else:
            normalized_content.append({"type": text_type, "text": str(content)})

        if not normalized_content:
            normalized_content.append({"type": text_type, "text": ""})

        response_input.append({
            "role": role,
            "content": normalized_content
        })

    return response_input


def convert_chat_completions_to_responses_payload(chat_payload):
    """Konvertiert einen Chat-Completions Payload in das Responses-Format."""
    payload = dict(chat_payload or {})
    messages = payload.pop('messages', [])
    payload['input'] = convert_messages_to_responses_input(messages)

    if 'max_tokens' in payload and 'max_output_tokens' not in payload:
        payload['max_output_tokens'] = payload.pop('max_tokens')

    return payload


def extract_output_texts(response_json):
    texts = []
    if isinstance(response_json.get('output_text'), list):
        texts.extend([str(t) for t in response_json.get('output_text', []) if t is not None])
    output_items = response_json.get('output') or []
    for item in output_items:
        contents = item.get('content') if isinstance(item, dict) else None
        if not contents:
            continue
        for block in contents:
            text = block.get('text') if isinstance(block, dict) else None
            if text:
                texts.append(str(text))
    if not texts and isinstance(response_json.get('content'), list):
        for block in response_json['content']:
            if isinstance(block, dict) and block.get('type') == 'output_text':
                text = block.get('text')
                if text:
                    texts.append(str(text))
    return texts


def normalize_usage(usage):
    if not isinstance(usage, dict):
        return usage
    prompt_tokens = usage.get('input_tokens', usage.get('prompt_tokens'))
    completion_tokens = usage.get('output_tokens', usage.get('completion_tokens'))
    total_tokens = usage.get('total_tokens')
    result = {}
    if prompt_tokens is not None:
        result['prompt_tokens'] = prompt_tokens
    if completion_tokens is not None:
        result['completion_tokens'] = completion_tokens
    if total_tokens is not None:
        result['total_tokens'] = total_tokens
    return result or usage


def responses_json_to_chat_completion(response_json, model_name):
    texts = extract_output_texts(response_json)
    message_content = "\n".join(texts).strip()
    finish_reason = response_json.get('stop_reason') or response_json.get('finish_reason') or 'stop'
    usage = normalize_usage(response_json.get('usage'))
    return {
        "id": response_json.get('id', ''),
        "object": "chat.completion",
        "created": response_json.get('created', int(time.time())),
        "model": model_name,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": message_content
            },
            "finish_reason": finish_reason
        }],
        "usage": usage
    }


def extract_reasoning_summary(response_json):
    summaries = []
    for output in response_json.get('output', []):
        if output.get('type') == 'reasoning':
            for entry in output.get('summary') or []:
                if entry.get('type') == 'summary_text' and entry.get('text'):
                    summaries.append(entry['text'])
    return summaries


def inject_reasoning_summary(response_json, chat_json):
    summaries = extract_reasoning_summary(response_json)
    if not summaries:
        return chat_json
    summary_text = "\n\nReasoning Summary:\n" + "\n".join(summaries)
    choices = chat_json.get('choices') or []
    if not choices:
        return chat_json
    message = choices[0].get('message') or {}
    original_content = message.get('content', '')
    if original_content:
        message['content'] = f"{original_content}{summary_text}"
    else:
        message['content'] = summary_text
    return chat_json


def build_chat_chunk(meta, model_name, content=None, finish_reason=None, role=None, usage=None):
    chunk = {
        "id": meta.get('id', ''),
        "object": "chat.completion.chunk",
        "created": meta.get('created', int(time.time())),
        "model": model_name,
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": finish_reason
        }]
    }
    delta = {}
    if role is not None:
        delta["role"] = role
    if content is not None:
        delta["content"] = content
    if delta:
        chunk["choices"][0]["delta"] = delta
    if usage:
        chunk["usage"] = usage
    return chunk


def format_sse_chunk(payload):
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode('utf-8')


def stream_responses_as_chat(resp, model_name, start_time, incoming_port, original_path, upstream_path):
    meta = {"id": "", "created": int(time.time())}
    usage = None

    def generator():
        nonlocal usage
        total_bytes = 0
        sent_role_chunk = False
        reasoning_header_sent = False

        def emit_downstream(payload_bytes):
            nonlocal total_bytes
            total_bytes += len(payload_bytes)
            log_payload(incoming_port or 'unknown', 'downstream-response-chunk', original_path, payload_bytes)
            return payload_bytes

        def emit_role_once():
            nonlocal sent_role_chunk
            if sent_role_chunk:
                return []
            role_chunk = build_chat_chunk(meta, model_name, role="assistant")
            chunk_bytes = format_sse_chunk(role_chunk)
            sent_role_chunk = True
            return [emit_downstream(chunk_bytes)]

        def emit_done():
            done_payload = emit_downstream(b"data: [DONE]\n\n")
            return done_payload

        def emit_reasoning_chunk(text):
            nonlocal reasoning_header_sent
            emitted = []
            for role_chunk_bytes in emit_role_once():
                emitted.append(role_chunk_bytes)
            prefix = "\n\nReasoning Summary:\n" if not reasoning_header_sent else "\n"
            reasoning_header_sent = True
            chunk = build_chat_chunk(meta, model_name, content=f"{prefix}{text}")
            chunk_bytes = format_sse_chunk(chunk)
            emitted.append(emit_downstream(chunk_bytes))
            return emitted

        try:
            for raw_line in resp.iter_lines(decode_unicode=True):
                if raw_line is not None:
                    log_payload(incoming_port or 'unknown', 'upstream-response-chunk', upstream_path, f"{raw_line}\n".encode('utf-8'))
                if raw_line is None:
                    continue
                line = raw_line.strip()
                if not line:
                    continue
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if not data:
                    continue
                if data == "[DONE]":
                    done_chunk = build_chat_chunk(meta, model_name, finish_reason="stop", usage=usage)
                    chunk_bytes = format_sse_chunk(done_chunk)
                    yield emit_downstream(chunk_bytes)
                    yield emit_done()
                    break
                try:
                    event = json.loads(data)
                except json.JSONDecodeError:
                    continue

                response_obj = event.get('response') or {}
                if response_obj.get('id'):
                    meta['id'] = response_obj['id']
                if response_obj.get('created'):
                    meta['created'] = response_obj['created']
                if response_obj.get('usage'):
                    usage = normalize_usage(response_obj.get('usage'))

                event_type = event.get('type')
                if event_type == 'response.output_text.delta':
                    delta_text = event.get('delta') or ''
                    if delta_text:
                        for role_chunk_bytes in emit_role_once():
                            yield role_chunk_bytes
                        chunk = build_chat_chunk(meta, model_name, content=delta_text)
                        chunk_bytes = format_sse_chunk(chunk)
                        yield emit_downstream(chunk_bytes)
                elif event_type == 'response.completed':
                    finish_reason = response_obj.get('stop_reason') or 'stop'
                    for role_chunk_bytes in emit_role_once():
                        yield role_chunk_bytes
                    chunk = build_chat_chunk(meta, model_name, finish_reason=finish_reason, usage=usage)
                    chunk_bytes = format_sse_chunk(chunk)
                    yield emit_downstream(chunk_bytes)
                    yield emit_done()
                    break
                elif event_type in ('response.output_item.added', 'response.output_item.done'):
                    item = event.get('item') or {}
                    if item.get('type') == 'reasoning':
                        for entry in item.get('summary') or []:
                            text = entry.get('text')
                            if text:
                                for chunk_bytes in emit_reasoning_chunk(text):
                                    yield chunk_bytes
        finally:
            resp.close()
            duration = time.time() - start_time if start_time else 0
            tokens_info = ""
            if usage:
                tokens_info = f"| Tokens: {usage.get('total_tokens', '?'):>5}"
            print(f"END:   {model_name:<20} | Output: {total_bytes:>8} bytes | {duration:.2f}s {tokens_info}")

    return generator()

@app.route('/v1/models', methods=['GET'])
def get_models():
    host_header = request.headers.get('Host', '')
    incoming_port = None
    if ':' in host_header:
        try:
            incoming_port = host_header.split(':')[-1]
        except Exception:
            incoming_port = None
    if not incoming_port:
        incoming_port = str(request.environ.get('SERVER_PORT', ''))

    if incoming_port in ("5101", "5102"):
        target_base = PORT_TARGET_MAP.get(incoming_port, DEFAULT_TARGET_URL)
        url = f"{target_base}/v1/models"
        headers = {}
        for key, value in request.headers:
            if key.lower() in ('host', 'accept-encoding'):
                continue
            headers[key] = value
        headers['Accept-Encoding'] = 'identity'
        log_payload(incoming_port, 'upstream-request', 'v1/models', '<no body>')

        try:
            resp = requests.get(url, headers=headers, timeout=(10, 60))
            filtered_body = resp.content
            content_type = resp.headers.get('Content-Type', '')
            log_payload(incoming_port, 'upstream-response', 'v1/models', filtered_body)

            try:
                payload = resp.json()
                if isinstance(payload, dict) and isinstance(payload.get('data'), list):
                    filtered_models = []
                    for model in payload['data']:
                        model_id = model.get('id') if isinstance(model, dict) else None
                        if not isinstance(model_id, str):
                            continue
                        has_codex = 'codex' in model_id.lower()
                        if incoming_port == "5102" and has_codex:
                            filtered_models.append(model)
                        elif incoming_port == "5101" and not has_codex:
                            filtered_models.append(model)
                    payload['data'] = filtered_models
                filtered_body = json.dumps(payload).encode('utf-8')
                content_type = 'application/json'
            except ValueError:
                pass

            log_payload(incoming_port, 'downstream-response', 'v1/models', filtered_body)
            response = Response(filtered_body, status=resp.status_code)
            excluded_headers = ['content-encoding', 'content-length', 'transfer-encoding', 'connection']
            for name, value in resp.headers.items():
                if name.lower() not in excluded_headers:
                    response.headers[name] = value
            response.headers['Content-Encoding'] = 'identity'
            if content_type:
                response.headers['Content-Type'] = content_type
            return response
        except requests.exceptions.RequestException as e:
            print(f"FEHLER: Modelle konnten nicht geladen werden: {e}")
            return jsonify({"error": f"Could not load models: {e}"}), 502

    return proxy('v1/models')

@app.route('/<path:path>', methods=['GET', 'POST', 'PUT', 'DELETE', 'PATCH'])
def proxy(path):
    try:
        
        # Bestimme eingehenden Port (Host-Header hat normalerweise host:port)
        host_header = request.headers.get('Host', '')
        incoming_port = None
        if ':' in host_header:
            try:
                incoming_port = host_header.split(':')[-1]
            except Exception:
                incoming_port = None
        # Fallback: SERVER_PORT (z.B. innerhalb des Containers)
        if not incoming_port:
            incoming_port = str(request.environ.get('SERVER_PORT', ''))

        target_path = path
        transform_to_responses = incoming_port == "5102" and path == 'v1/chat/completions'
        if transform_to_responses:
            target_path = 'v1/responses'

        # Wähle die Ziel-URL basierend auf dem Port (oder nutze Default)
        target_base = PORT_TARGET_MAP.get(incoming_port, DEFAULT_TARGET_URL)
        url = f"{target_base}/{target_path}"
        # Header vorbereiten und Content-Encoding kontrollieren
        headers = {}
        for key, value in request.headers:
            # Host-Header überspringen
            if key.lower() == 'host':
                continue
            # Accept-Encoding anpassen um Komprimierung zu verhindern
            if key.lower() == 'accept-encoding':
                continue
            # Alle anderen Header übernehmen, insbesondere Authorization
            headers[key] = value
            
        # Explizit keine Komprimierung anfordern
        headers['Accept-Encoding'] = 'identity'
        data = request.get_data()
        
        # Modellname und Parameter aus dem Request extrahieren falls es ein Chat/Completion Request ist
        model_name = None
        start_time = None
        request_payload = None
        is_streaming_request = False
        if path in ['v1/chat/completions', 'v1/completions'] and request.is_json:
            try:
                request_payload = request.get_json()
                model_name = request_payload.get('model', 'unbekannt')
                start_time = time.time()
                is_streaming_request = bool(request_payload.get('stream'))
                
                # Alle Parameter aus dem Request extrahieren
                params = []
                for key, value in request_payload.items():
                    # Überspringe die Nachrichtenliste und das Modell, da diese oft zu lang sind
                    if key not in ['messages', 'model']:
                        params.append(f"{key}={value}")
                
                # Start-Logzeile ausgeben
                print(f"START: {model_name:<20} | Input: {len(data):>8} bytes | {', '.join(sorted(params))}")
                
            except Exception as e:
                print(f"WARNUNG: Fehler beim Verarbeiten der Anfrage: {e}")
                model_name = 'unbekannt'
                request_payload = None

        if transform_to_responses and request.is_json:
            try:
                payload_for_conversion = request_payload
                if payload_for_conversion is None and data:
                    payload_for_conversion = json.loads(data.decode('utf-8'))
                responses_payload = convert_chat_completions_to_responses_payload(payload_for_conversion or {})
                # Xcode erwartet Websuche; wenn keine Tools angegeben sind, fügen wir das Default-Tool hinzu.
                if not responses_payload.get('tools'):
                    responses_payload['tools'] = [{"type": "web_search"}]
                reasoning_cfg = responses_payload.get('reasoning')
                if not reasoning_cfg:
                    responses_payload['reasoning'] = {"effort": "medium", "summary": "auto"}
                else:
                    if not reasoning_cfg.get('effort'):
                        reasoning_cfg['effort'] = "medium"
                    if not reasoning_cfg.get('summary'):
                        reasoning_cfg['summary'] = "auto"
                data = json.dumps(responses_payload).encode('utf-8')
                headers['Content-Type'] = 'application/json'
            except Exception as e:
                print(f"WARNUNG: Konnte Payload nicht in Responses-Format umwandeln: {e}")

        log_payload(incoming_port or 'unknown', 'upstream-request', target_path, data if data else '<no body>')

        try:
            # Separate Timeouts für Verbindungsaufbau und Lesen
            resp = requests.request(
                method=request.method,
                url=url,
                headers=headers,
                data=data,
                stream=True,
                timeout=(10, 600)  # (connect timeout, read timeout) in Sekunden - auf 600 Sekunden erhöht
            )

            # Behandle Timeout (504) und andere Fehler
            if resp.status_code != 200:
                error_msg = f"WARNUNG: Antwort von {request.method} {url} erhielt Status-Code {resp.status_code}"
                if resp.status_code == 504:
                    error_msg += " (Gateway Timeout)"
                print(error_msg)
                
                # Versuche die Fehlermeldung zu lesen
                try:
                    error_content = resp.json()
                    # Bei OpenAI API-Schlüsselproblemen detailliertere Ausgabe
                    if incoming_port in ("5101", "5102") and resp.status_code == 401:
                        print(f"FEHLER: OpenAI API-Schlüsselproblem: {error_content}")
                except:
                    pass

            if transform_to_responses and is_streaming_request and resp.status_code == 200:
                response = Response(
                    stream_responses_as_chat(
                        resp,
                        model_name or 'unbekannt',
                        start_time,
                        incoming_port,
                        path,
                        target_path
                    ),
                    status=resp.status_code
                )
                excluded_headers = ['content-encoding', 'content-length', 'transfer-encoding', 'connection']
                for name, value in resp.headers.items():
                    if name.lower() not in excluded_headers:
                        response.headers[name] = value
                response.headers['Content-Type'] = 'text/event-stream'
                response.headers['Content-Encoding'] = 'identity'
                return response

            if transform_to_responses and not is_streaming_request and resp.status_code == 200:
                try:
                    original_json = resp.json()
                    log_payload(incoming_port or 'unknown', 'upstream-response', target_path, json.dumps(original_json, ensure_ascii=False))
                    chat_json = responses_json_to_chat_completion(original_json, model_name or 'unbekannt')
                    chat_json = inject_reasoning_summary(original_json, chat_json)
                    response_body = json.dumps(chat_json).encode('utf-8')
                    log_payload(incoming_port or 'unknown', 'downstream-response', path, response_body)
                    total_bytes = len(response_body)
                    duration = time.time() - start_time if start_time else 0
                    tokens_info = ""
                    usage = chat_json.get('usage')
                    if isinstance(usage, dict):
                        tokens_info = f"| Tokens: {usage.get('total_tokens', '?'):>5}"
                    print(f"END:   {model_name:<20} | Output: {total_bytes:>8} bytes | {duration:.2f}s {tokens_info}")

                    response = Response(response_body, status=resp.status_code)
                    excluded_headers = ['content-encoding', 'content-length', 'transfer-encoding', 'connection']
                    for name, value in resp.headers.items():
                        if name.lower() not in excluded_headers:
                            response.headers[name] = value
                    response.headers['Content-Type'] = 'application/json'
                    response.headers['Content-Encoding'] = 'identity'
                    return response
                except ValueError:
                    pass

            def stream_with_logging(log_port, upstream_path):
                total_bytes = 0
                response_json = None
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        total_bytes += len(chunk)
                        log_payload(log_port or 'unknown', 'upstream-response-chunk', upstream_path, chunk)
                        # Versuche JSON-Antwort zu parsen für Token-Info
                        if not response_json and chunk.startswith(b'{'):
                            try:
                                response_json = json.loads(chunk)
                            except:
                                pass
                    yield chunk
                
                if model_name and path in ['v1/chat/completions', 'v1/completions']:
                    duration = time.time() - start_time if start_time else 0
                    tokens_info = ""
                    status_info = ""
                    
                    # Token-Informationen
                    if response_json and 'usage' in response_json:
                        usage = response_json['usage']
                        tokens_info = f"| Tokens: {usage.get('total_tokens', '?'):>5}"
                    
                    # Status-Information bei Fehlern
                    if resp.status_code != 200:
                        status_info = f"| Status: {resp.status_code}"
                        if resp.status_code == 504:
                            status_info += " (Timeout)"
                    
                    print(f"END:   {model_name:<20} | Output: {total_bytes:>8} bytes | {duration:.2f}s {tokens_info} {status_info}")
                resp.close()
            
            response = Response(stream_with_logging(incoming_port, target_path), status=resp.status_code)
            
            # Header für die Antwort vorbereiten
            excluded_headers = ['content-encoding', 'content-length', 'transfer-encoding', 'connection']
            for name, value in resp.headers.items():
                if name.lower() not in excluded_headers:
                    response.headers[name] = value
            
            # Stellen Sie sicher, dass keine Komprimierung verwendet wird
            response.headers['Content-Encoding'] = 'identity'
            
            return response

        except requests.exceptions.Timeout as e:
            error_msg = f"FEHLER: Zeitüberschreitung nach 600 Sekunden bei Anfrage an {url}"
            print(error_msg)
            return jsonify({
                "error": "Timeout Error",
                "message": "Die Anfrage wurde nach 600 Sekunden abgebrochen",
                "details": str(e)
            }), 504  # Gateway Timeout
        except requests.exceptions.RequestException as e:
            print(f"FEHLER: Verbindung zum Zielserver {url} fehlgeschlagen: {e}")
            return jsonify({"error": f"Proxy request failed: {e}"}), 502

    except Exception as e:
        print(f"Ein unerwarteter interner Fehler ist aufgetreten: {e}")
        return jsonify({"error": "An internal server error occurred."}), 500

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Ein robuster Proxy für OpenAI (Port 5101).")
    parser.add_argument('--port', type=int, default=5101,
                        help='Der Port, auf dem der Proxy-Server laufen soll (Standard: 5101).')
    parser.add_argument('--log', type=str,
                        help='Pfad zu einer Log-Datei, in der Anfragen/Antworten mitgeschnitten werden.')
    
    args = parser.parse_args()

    if args.log:
        configure_logging(args.log)

    port = args.port
    
    print(f"OpenAI Proxy wird auf http://0.0.0.0:{port} gestartet...")
    
    serve(app, host='0.0.0.0', port=port,
          url_scheme='http', 
          channel_timeout=600,  # 10 Minuten Timeout für die Verbindung - auf 600 Sekunden erhöht
          cleanup_interval=600,  # Cleanup alle 10 Minuten - auf 600 Sekunden erhöht
          threads=8)
