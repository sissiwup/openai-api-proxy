# 1. Basis-Image wählen
FROM python:3.11-slim

# NEU: Setzt die Umgebungsvariable für alle Container, die von diesem Image gebaut werden
ENV PYTHONUNBUFFERED=1

# 2. Arbeitsverzeichnis im Container festlegen
WORKDIR /app

# 3. Abhängigkeiten installieren
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Den Quellcode der Anwendung in den Container kopieren
COPY openai_proxy.py .

# 5. Den Port freigeben, auf dem die Anwendung lauscht
EXPOSE 5101

# 6. Der Befehl, der beim Starten des Containers ausgeführt wird
CMD ["python3", "openai_proxy.py", "--port", "5101"]
