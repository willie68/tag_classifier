# Tag Classifier

Klassifiziert Bildertags in Kategorien mittels Zero-Shot Classification.

## Beschreibung

Dieses Tool verwendet das Zero-Shot Classification Modell `MoritzLaurer/deberta-v3-base-zeroshot-v2.0`, um eine Liste von Bildertags automatisch der wahrscheinlichsten Kategorie zuzuordnen.

**Hauptmerkmale:**
- Zero-Shot Learning: Keine Trainings-Daten erforderlich
- Flexible Kategorien: Beliebige Kategorien können verwendet werden
- Mehrsprachig: Funktioniert mit Tags in verschiedenen Sprachen
- Confidence Scores: Gibt Wahrscheinlichkeiten für alle Kategorien zurück

## Installation

### Manuelle Installation

**Schritt 1: Virtuelles Environment erstellen**

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Windows (CMD):**
```cmd
python -m venv venv
venv\Scripts\activate.bat
```

**Linux/macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Schritt 2: Abhängigkeiten installieren**

```bash
pip install -r requirements.txt
```

Beim ersten Start wird das Modell (~1.5 GB) automatisch heruntergeladen.

## Verwendung

### REST API Server

Der einfachste Weg ist die Nutzung des REST API Servers:

```bash
# Server mit Standardport 8766 starten
python server.py

# Server mit eigenem Port starten
python server.py --port 9000

# Mit Auto-Reload für Entwicklung
python server.py --reload
```

**Oder unter Windows:**
```cmd
start_server.bat
```

**API-Endpunkte:**

- `GET /` - Health Check (inkl. Modellname)
- `POST /classify` - Tag-Klassifizierung
- `GET /info` - API-Informationen
- `GET /docs` - Swagger UI Dokumentation
- `GET /redoc` - ReDoc Dokumentation

**Beispiel-Request (cURL):**
```bash
curl -X POST http://localhost:8766/classify \
  -H "Content-Type: application/json" \
  -d '{
    "tags": ["dog", "park", "outdoor"],
    "categories": ["animals", "landscape", "portrait"],
    "return_scores": true
  }'
```

**Beispiel-Request (Python):**
```python
import requests

response = requests.post("http://localhost:8766/classify", json={
    "tags": ["dog", "park", "outdoor"],
    "categories": ["animals", "landscape", "portrait"],
    "return_scores": True
})

result = response.json()
print(f"Kategorie: {result['category']}")
print(f"Score: {result['score']}")
```

**API testen:**
```bash
# Alle Tests ausführen
python test_api.py

# cURL-Beispiele anzeigen
python test_api.py --curl
```

**Postman Collection:**

Die Datei `Tag_Classifier_API.postman_collection.json` kann direkt in Postman importiert werden und enthält:
- Alle API-Endpunkte mit Beispiel-Requests
- Vorgefertigte Testfälle (Animals, Landscape, Portrait, Food, etc.)
- Deutsche Tags-Beispiele
- Fehlerbehandlungs-Beispiele
- Response-Beispiele
- Environment-Variable für baseUrl

### Python Library

Alternativ kann die Klasse direkt in Python verwendet werden:

### Basis-Beispiel

```python
from classify_tags import TagClassifier

# Classifier initialisieren
classifier = TagClassifier()

# Tags klassifizieren
tags = ["dog", "park", "playing", "outdoor"]
categories = ["animals", "landscape", "portrait", "architecture"]

result = classifier.classify_tags(tags, categories)

print(f"Kategorie: {result['category']}")
print(f"Confidence: {result['score']:.2%}")
```

### Mit allen Scores

```python
result = classifier.classify_tags(tags, categories, return_scores=True)

print(f"Beste Kategorie: {result['category']}")
for cat, score in result['all_scores'].items():
    print(f"  {cat}: {score:.2%}")
```

### Einzelnen Tag klassifizieren

```python
result = classifier.classify_single_tag("sunset", categories)
print(f"Kategorie: {result['category']}")
```

### Beispiele ausführen

```bash
# Basis-Beispiele
python classify_tags.py

# Erweiterte Beispiele
python example_usage.py
```

## API Referenz

### REST API

#### `POST /classify`

Klassifiziert Bildertags in Kategorien.

**Request Body:**
```json
{
  "tags": ["dog", "park", "outdoor"],
  "categories": ["animals", "landscape", "portrait"],
  "multi_label": false,
  "return_scores": false
}
```

**Response:**
```json
{
  "category": "animals",
  "score": 0.92,
  "all_scores": {
    "animals": 0.92,
    "landscape": 0.05,
    "portrait": 0.03
  }
}
```

**Parameter:**
- `tags` (required): Liste von Bildertags
- `categories` (required): Liste möglicher Kategorien (min. 2)
- `multi_label` (optional): Boolean, default `false`
- `return_scores` (optional): Boolean, default `false`

#### `GET /`

Health Check Endpunkt.

**Response:**
```json
{
  "status": "healthy",
  "model": "MoritzLaurer/deberta-v3-base-zeroshot-v2.0",
  "timestamp": "2026-01-24T10:30:00",
  "version": "1.0.0"
}
```

#### `GET /info`

API-Informationen.

**Response:**
```json
{
  "name": "Tag Classifier API",
  "version": "1.0.0",
  "model": "MoritzLaurer/deberta-v3-base-zeroshot-v2.0",
  "endpoints": {...}
}
```

### TagClassifier

#### `__init__(model_name: str = "MoritzLaurer/deberta-v3-base-zeroshot-v2.0")`

Initialisiert den Classifier.

**Parameter:**
- `model_name`: Name des Hugging Face Modells

#### `classify_tags(tags: List[str], categories: List[str], multi_label: bool = False, return_scores: bool = False) -> Dict`

Klassifiziert eine Liste von Tags.

**Parameter:**
- `tags`: Liste von Bildertags (z.B. `["dog", "outdoor", "sunny"]`)
- `categories`: Liste möglicher Kategorien (z.B. `["animals", "landscape", "portrait"]`)
- `multi_label`: Wenn True, können mehrere Kategorien gleichzeitig zutreffen
- `return_scores`: Wenn True, werden alle Scores zurückgegeben

**Rückgabe:**
```python
{
    "category": "animals",          # Beste Kategorie
    "score": 0.92,                  # Confidence Score (0-1)
    "all_scores": {                 # Optional, wenn return_scores=True
        "animals": 0.92,
        "landscape": 0.05,
        "portrait": 0.02,
        "architecture": 0.01
    }
}
```

#### `classify_single_tag(tag: str, categories: List[str], return_scores: bool = False) -> Dict`

Klassifiziert einen einzelnen Tag.

**Parameter:**
- `tag`: Einzelner Tag (z.B. `"dog"`)
- `categories`: Liste möglicher Kategorien
- `return_scores`: Wenn True, werden alle Scores zurückgegeben

## Anwendungsfälle

### 1. Automatische Bild-Kategorisierung

```python
classifier = TagClassifier()
categories = ["animals", "nature", "people", "food", "architecture"]

images = [
    {"id": "IMG_001", "tags": ["cat", "sitting", "window"]},
    {"id": "IMG_002", "tags": ["mountains", "snow", "landscape"]},
]

for image in images:
    result = classifier.classify_tags(image["tags"], categories)
    print(f"{image['id']}: {result['category']}")
```

### 2. Schwellenwert-Filterung

```python
threshold = 0.7
result = classifier.classify_tags(tags, categories)

if result['score'] >= threshold:
    print(f"Kategorie: {result['category']}")
else:
    print("Unsichere Klassifizierung")
```

### 3. Deutsche Tags

```python
tags = ["Hund", "Park", "spielend"]
categories = ["Tiere", "Landschaft", "Portrait", "Architektur"]

result = classifier.classify_tags(tags, categories)
print(result['category'])  # → "Tiere"
```

## Technische Details

- **Modell:** MoritzLaurer/deberta-v3-base-zeroshot-v2.0
- **Typ:** Zero-Shot Text Classification
- **Framework:** Hugging Face Transformers
- **Sprachen:** Mehrsprachig (inkl. Englisch, Deutsch, Französisch, etc.)
- **Hardware:** CPU (Standard), GPU optional

## Performance

- Erste Ausführung: ~10-30 Sekunden (Modell-Download)
- Nachfolgende Ausführungen: ~1-3 Sekunden pro Klassifizierung
- GPU-Nutzung: Für GPU `device=0` in `__init__` setzen

## Lizenz

MIT
