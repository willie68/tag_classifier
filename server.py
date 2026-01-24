"""
REST API Server für Tag Classification

Stellt einen REST-Endpunkt zur Verfügung, um Bildertags in Kategorien zu klassifizieren.
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import uvicorn
import argparse
import logging
from datetime import datetime

from classify_tags import TagClassifier

# Logging konfigurieren
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Globale Variable für den Classifier
classifier = None
MODEL_NAME = "MoritzLaurer/deberta-v3-base-zeroshot-v2.0"

# FastAPI App erstellen
app = FastAPI(
    title="Tag Classifier API",
    description="Klassifiziert Bildertags in Kategorien mittels Zero-Shot Classification",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)


# Pydantic Models für Request/Response
class ClassificationRequest(BaseModel):
    """Request Model für Tag-Klassifizierung"""
    tags: List[str] = Field(
        ...,
        description="Liste von Bildertags",
        example=["dog", "park", "outdoor", "sunny"]
    )
    categories: List[str] = Field(
        ...,
        description="Liste möglicher Kategorien",
        example=["animals", "landscape", "portrait", "architecture", "food"]
    )
    multi_label: bool = Field(
        default=False,
        description="Wenn True, können mehrere Kategorien gleichzeitig zutreffen"
    )
    return_scores: bool = Field(
        default=False,
        description="Wenn True, werden alle Kategorie-Scores zurückgegeben"
    )


class ClassificationResponse(BaseModel):
    """Response Model für Tag-Klassifizierung"""
    category: str = Field(description="Die beste/wahrscheinlichste Kategorie")
    score: float = Field(description="Confidence Score (0-1)")
    all_scores: Optional[Dict[str, float]] = Field(
        default=None,
        description="Alle Kategorie-Scores (nur wenn return_scores=True)"
    )


class HealthResponse(BaseModel):
    """Response Model für Health Check"""
    status: str = Field(description="Status des Services")
    model: str = Field(description="Name des verwendeten Modells")
    timestamp: str = Field(description="Aktueller Zeitstempel")
    version: str = Field(description="API Version")


@app.on_event("startup")
async def startup_event():
    """Wird beim Start des Servers ausgeführt - lädt das Modell"""
    global classifier
    logger.info("Server wird gestartet...")
    logger.info(f"Lade Modell: {MODEL_NAME}")
    
    try:
        classifier = TagClassifier(model_name=MODEL_NAME)
        logger.info("Modell erfolgreich geladen!")
        logger.info("Server ist bereit für Anfragen")
    except Exception as e:
        logger.error(f"Fehler beim Laden des Modells: {e}")
        raise


@app.get("/", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health Check Endpunkt
    
    Gibt den Status des Services und Informationen zum geladenen Modell zurück.
    """
    return HealthResponse(
        status="healthy" if classifier is not None else "unhealthy",
        model=MODEL_NAME,
        timestamp=datetime.now().isoformat(),
        version="1.0.0"
    )


@app.post("/classify", response_model=ClassificationResponse, tags=["Classification"])
async def classify_tags(request: ClassificationRequest):
    """
    Klassifiziert Bildertags in Kategorien
    
    Verwendet Zero-Shot Classification, um eine Liste von Tags der wahrscheinlichsten
    Kategorie zuzuordnen.
    
    **Parameter:**
    - **tags**: Liste von Bildertags (z.B. ["dog", "park", "outdoor"])
    - **categories**: Liste möglicher Kategorien (z.B. ["animals", "landscape"])
    - **multi_label**: Ob mehrere Kategorien gleichzeitig zutreffen können (optional)
    - **return_scores**: Ob alle Kategorie-Scores zurückgegeben werden sollen (optional)
    
    **Rückgabe:**
    - **category**: Die beste Kategorie
    - **score**: Confidence Score (0-1)
    - **all_scores**: Alle Scores (nur wenn return_scores=True)
    """
    if classifier is None:
        logger.error("Classifier ist nicht initialisiert")
        raise HTTPException(
            status_code=503,
            detail="Service nicht verfügbar - Modell nicht geladen"
        )
    
    # Validierung
    if not request.tags:
        raise HTTPException(
            status_code=400,
            detail="Mindestens ein Tag muss angegeben werden"
        )
    
    if len(request.categories) < 2:
        raise HTTPException(
            status_code=400,
            detail="Mindestens zwei Kategorien müssen angegeben werden"
        )
    
    try:
        logger.info(f"Klassifiziere Tags: {request.tags} in Kategorien: {request.categories}")
        
        # Klassifizierung durchführen
        result = classifier.classify_tags(
            tags=request.tags,
            categories=request.categories,
            multi_label=request.multi_label,
            return_scores=request.return_scores
        )
        
        logger.info(f"Ergebnis: {result['category']} (Score: {result['score']:.2%})")
        
        return ClassificationResponse(**result)
        
    except Exception as e:
        logger.error(f"Fehler bei der Klassifizierung: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Interner Fehler bei der Klassifizierung: {str(e)}"
        )


@app.get("/info", tags=["Info"])
async def get_info():
    """
    Gibt Informationen über die API zurück
    """
    return {
        "name": "Tag Classifier API",
        "version": "1.0.0",
        "model": MODEL_NAME,
        "endpoints": {
            "/": "Health Check",
            "/classify": "Tag Classification (POST)",
            "/info": "API Information",
            "/docs": "Swagger UI Dokumentation",
            "/redoc": "ReDoc Dokumentation"
        }
    }


def parse_args():
    """Kommandozeilen-Argumente parsen"""
    parser = argparse.ArgumentParser(
        description="Tag Classifier REST API Server"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8766,
        help="Port für den Server (default: 8766)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host-Adresse (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Auto-Reload bei Code-Änderungen aktivieren"
    )
    
    return parser.parse_args()


def main():
    """Hauptfunktion zum Starten des Servers"""
    args = parse_args()
    
    logger.info("=" * 60)
    logger.info("Tag Classifier REST API Server")
    logger.info("=" * 60)
    logger.info(f"Host: {args.host}")
    logger.info(f"Port: {args.port}")
    logger.info(f"Modell: {MODEL_NAME}")
    logger.info("=" * 60)
    logger.info(f"Swagger Docs: http://{args.host if args.host != '0.0.0.0' else 'localhost'}:{args.port}/docs")
    logger.info(f"ReDoc: http://{args.host if args.host != '0.0.0.0' else 'localhost'}:{args.port}/redoc")
    logger.info("=" * 60)
    
    # Server starten
    uvicorn.run(
        "server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info"
    )


if __name__ == "__main__":
    main()
