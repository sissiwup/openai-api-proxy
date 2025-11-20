# Repository overview

## Python Proxy für die OpenAI API (Compat-Ports 5101/5102)
Ein leichter, in Python (Flask + Waitress) umgesetzter Proxy, der zwischen OpenAI-kompatiblen Clients und der offiziellen OpenAI REST API vermittelt. Er stellt fehlende Komfortfunktionen wie `/v1/models`, Responses-Bridging, Tool-/Reasoning-Defaults sowie Logging bereit und bleibt dabei zu 100 % API-kompatibel.

### Zweck
Viele Clients (z. B. Xcode, LobeChat, n8n oder eigene Skripte) erwarten historische OpenAI-Endpunkte wie `/v1/chat/completions`, obwohl OpenAI neue Features ausschließlich über `/v1/responses` ausliefert. Dieser Proxy löst das Delta:

- Stellt `/v1/models` mit portabhängiger Filterlogik bereit (Codex-Only vs. generische Modelle).
- Port `5101`: transparentes Passthrough für klassische `/v1/chat/completions`.
- Port `5102`: konvertiert `/v1/chat/completions` → `/v1/responses`, inkl. Streaming, Reasoning-Summary und automatischem `web_search`-Tool.
- Vision-/File-Payloads (z. B. `image_url`, `file_id`) werden in das Responses-konforme Format (`input_image`, `input_file`) überführt.
- Automatische Function-Mapping-Schicht: Ein `functions`/`function_call`-Feld im Chat-Completions-Request wird nahtlos in Responses-`tools` + `tool_choice` übertragen (inkl. erzwungenen Function-Calls und `parallel_tool_calls`).
- Reasoning-aware Modell-Aliasse: Auf Port `5102` tauchen Modell-IDs wie `gpt-5.1-codex:res-high:sum-auto` auf; der Proxy sorgt dafür, dass OpenAI nur den Basisnamen sieht, die passenden Reasoning-Defaults aber automatisch gesetzt werden.
- Optionales strukturiertes Logging (Requests, Responses, Streaming-Chunks).

Der Proxy passt keine Authorization-Header an und speichert keine Schlüssel: Alles, was der Client sendet, wird direkt an `https://api.openai.com` durchgereicht.

### Quellcode
Der komplette Code steht im zugehörigen GitHub-Repository:

[sissiwup/openai-proxy](https://github.com/sissiwup/openai-proxy)

---

## Schnellstart

### Image pullen
```bash
docker pull sissiwup/openai-proxy:latest
```

### Container starten (Standardbetrieb)
Startet beide Ports im Container (intern 5101) und veröffentlicht sie auf dem Host.
```bash
docker run -d \
  -p 5101:5101 \
  -p 5102:5101 \
  sissiwup/openai-proxy:latest
```
> Port 5102 wird optional über ein zweites Host-Port-Mapping veröffentlicht (z. B. `-p 5102:5102` in Kombination mit `--port 5102`). Passen Sie das Mapping an Ihre Umgebung an.

### Container mit eigener Modell-Liste
Mounten Sie eine `models.json`, um den `/v1/models`-Output zu kontrollieren (falls nicht vorhanden, greift die interne Default-Liste).
```bash
docker run -d \
  -p 5101:5101 \
  -p 5102:5101 \
  sissiwup/openai-proxy:latest
```

### Logging aktivieren
```bash
docker run -d \
  -p 5101:5101 \
  -v $PWD/logs:/logs \
  --name openai-proxy \
  sissiwup/openai-proxy:latest \
  python openai_proxy.py --port 5101 --log /logs/proxy_5101.log
```
Die Log-Datei wird bei jedem Start neu erzeugt.

> Automatische Websuche deaktivieren:
> `python openai_proxy.py --port 5102 --disable-auto-web-search`
> (dann injiziert Port 5102 kein `web_search`-Tool mehr, solange Clients keines anfordern.)

---

## Ports & Routing
| Host-Port | Verhalten | Besonderheiten |
|-----------|-----------|----------------|
| 5101      | Direkter `/v1/chat/completions`-Proxy | `/v1/models` listet alle Nicht-Codex-Modelle. Keine Tool-/Reasoning-Injektion. |
| 5102      | `/v1/chat/completions` → `/v1/responses` Bridge | Fügt bei Bedarf `web_search`-Tool + Reasoning (effort `medium`, summary `auto`) hinzu, streamt Reasoning-Summaries als Chat-Chunks und konvertiert Vision/File-Blöcke automatisch. |

### Reasoning-Profile per Modellalias

- Standardmäßig listet `/v1/models` auf Port `5102` den Basisnamen (z. B. `gpt-5.1-codex`) plus zusätzliche Aliasse `:res-{low|medium|high}:sum-{auto|none}`.
- Clients wählen einfach den gewünschten Alias; der Proxy zerlegt ihn, setzt `reasoning.effort`/`reasoning.summary` entsprechend (falls der Request keinen eigenen `reasoning`-Block enthält) und sendet nur den ursprünglichen Modellnamen an OpenAI.
- Der Basisname ohne Suffix steht weiterhin für `res-medium:sum-auto`.

### Function-Calling Kompatibilität

Clients, die noch das klassische `functions`-Feld nutzen, müssen ihre Payload nicht mehr anpassen:

- Alle `functions` werden beim Bridging zu Responses-`tools` umgewandelt.
- `function_call`-Vorgaben landen automatisch im Responses-`tool_choice` (inkl. `"auto"`, `"none"` oder explizitem Funktionsnamen).
- `parallel_tool_calls` wird unverändert übernommen, sodass parallele Funktionsaufrufe weiterhin funktionieren.

Beispiel:

```json
{
  "model": "gpt-5.1-codex",
  "messages": [{"role": "user", "content": "List all open pull requests"}],
  "functions": [{
    "name": "list_pull_requests",
    "description": "Returns open PRs",
    "parameters": {"type": "object", "properties": {}}
  }],
  "function_call": "auto",
  "parallel_tool_calls": false
}
```

Der Proxy injiziert automatisch `tools=[{"type":"function",...}]` und setzt `tool_choice="auto"`, bevor die Anfrage an `/v1/responses` gesendet wird.

---

## Anwendungsbeispiele

### Modellliste abrufen (Port 5101)
```bash
curl http://localhost:5101/v1/models
```

### Reasoning-Upgrade nutzen (Port 5102)
```bash
curl -X POST "http://localhost:5102/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d '{
        "model": "gpt-5.1-codex",
        "stream": true,
        "messages": [
          {"role": "system", "content": "You are a helpful assistant."},
          {"role": "user", "content": "Beschreibe das Bild."}
        ],
        "messages": [
          {
            "role": "user",
            "content": [
              {"type": "input_text", "text": "Was ist auf dem Bild?"},
              {"type": "input_image", "image_url": "https://example.com/cat.jpg"}
            ]
          }
        ]
     }'
```
Der Proxy wandelt den Call in eine Responses-Anfrage um, übernimmt Streaming-Antworten und re-formatiert sie wieder zu Chat-Completions.

### Transparente Weiterleitung (Port 5101)
```bash
curl -X POST "http://localhost:5101/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d '{
        "model": "gpt-4.1",
        "messages": [{"role": "user", "content": "Hallo Welt!"}]
     }'
```

---

## Rechtliches & Verantwortung

- **API Terms:** Die Nutzung unterliegt weiterhin den [OpenAI Terms of Use](https://openai.com/policies/terms-of-use).
- **Keine Schlüssel-Speicherung:** Der Proxy verwaltet keine API-Keys; übermittelte Secrets werden nur weitergeleitet.
- **Logs:** Enthalten vollständige Payloads. Aktivieren Sie Logging nur, wenn Sie die Daten vertraulich behandeln können.
- **Lizenz:** Dieses Projekt steht unter der MIT-Lizenz. Bereitstellung ohne Gewähr.

---

Viel Erfolg beim Einsatz! Feedback, Issues oder PRs gerne im GitHub-Repository.
