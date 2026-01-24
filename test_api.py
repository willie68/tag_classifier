"""
Test-Script für die Tag Classifier API

Testet alle Endpunkte des REST-Servers
"""

import requests
import json
from typing import Dict, Any


class TagClassifierClient:
    """Client für die Tag Classifier API"""
    
    def __init__(self, base_url: str = "http://localhost:8766"):
        """
        Initialisiert den API Client
        
        Args:
            base_url: Basis-URL des Servers (default: http://localhost:8766)
        """
        self.base_url = base_url.rstrip('/')
    
    def health_check(self) -> Dict[str, Any]:
        """Ruft den Health Check Endpunkt auf"""
        response = requests.get(f"{self.base_url}/")
        response.raise_for_status()
        return response.json()
    
    def classify(
        self,
        tags: list,
        categories: list,
        multi_label: bool = False,
        return_scores: bool = False
    ) -> Dict[str, Any]:
        """
        Klassifiziert Tags
        
        Args:
            tags: Liste von Tags
            categories: Liste von Kategorien
            multi_label: Ob mehrere Labels möglich sind
            return_scores: Ob alle Scores zurückgegeben werden sollen
        """
        data = {
            "tags": tags,
            "categories": categories,
            "multi_label": multi_label,
            "return_scores": return_scores
        }
        
        response = requests.post(
            f"{self.base_url}/classify",
            json=data
        )
        response.raise_for_status()
        return response.json()
    
    def get_info(self) -> Dict[str, Any]:
        """Ruft API-Informationen ab"""
        response = requests.get(f"{self.base_url}/info")
        response.raise_for_status()
        return response.json()


def test_api():
    """Führt Tests gegen die API aus"""
    
    print("=" * 60)
    print("Tag Classifier API Tests")
    print("=" * 60)
    
    client = TagClassifierClient()
    
    # Test 1: Health Check
    print("\n1. Health Check Test")
    print("-" * 40)
    try:
        health = client.health_check()
        print(f"✓ Status: {health['status']}")
        print(f"  Modell: {health['model']}")
        print(f"  Version: {health['version']}")
        print(f"  Zeitstempel: {health['timestamp']}")
    except Exception as e:
        print(f"✗ Fehler: {e}")
        return
    
    # Test 2: API Info
    print("\n2. API Info Test")
    print("-" * 40)
    try:
        info = client.get_info()
        print(f"✓ Name: {info['name']}")
        print(f"  Version: {info['version']}")
        print("  Endpunkte:")
        for endpoint, desc in info['endpoints'].items():
            print(f"    {endpoint}: {desc}")
    except Exception as e:
        print(f"✗ Fehler: {e}")
    
    # Test 3: Einfache Klassifizierung
    print("\n3. Einfache Klassifizierung")
    print("-" * 40)
    tags = ["dog", "park", "outdoor"]
    categories = ["animals", "landscape", "portrait", "architecture"]
    
    try:
        result = client.classify(tags, categories)
        print(f"✓ Tags: {tags}")
        print(f"  Kategorien: {categories}")
        print(f"  → Ergebnis: {result['category']}")
        print(f"  → Score: {result['score']:.2%}")
    except Exception as e:
        print(f"✗ Fehler: {e}")
    
    # Test 4: Klassifizierung mit allen Scores
    print("\n4. Klassifizierung mit allen Scores")
    print("-" * 40)
    tags = ["sunset", "mountains", "hiking"]
    
    try:
        result = client.classify(tags, categories, return_scores=True)
        print(f"✓ Tags: {tags}")
        print(f"  → Beste Kategorie: {result['category']} ({result['score']:.2%})")
        print("  → Alle Scores:")
        for cat, score in result['all_scores'].items():
            print(f"      {cat}: {score:.2%}")
    except Exception as e:
        print(f"✗ Fehler: {e}")
    
    # Test 5: Portrait Klassifizierung
    print("\n5. Portrait Klassifizierung")
    print("-" * 40)
    tags = ["woman", "smiling", "close-up", "face"]
    
    try:
        result = client.classify(tags, categories, return_scores=True)
        print(f"✓ Tags: {tags}")
        print(f"  → Kategorie: {result['category']} ({result['score']:.2%})")
    except Exception as e:
        print(f"✗ Fehler: {e}")
    
    # Test 6: Deutsche Tags
    print("\n6. Deutsche Tags")
    print("-" * 40)
    tags = ["Hund", "Park", "spielend"]
    categories_de = ["Tiere", "Landschaft", "Portrait", "Architektur"]
    
    try:
        result = client.classify(tags, categories_de)
        print(f"✓ Tags: {tags}")
        print(f"  Kategorien: {categories_de}")
        print(f"  → Kategorie: {result['category']} ({result['score']:.2%})")
    except Exception as e:
        print(f"✗ Fehler: {e}")
    
    # Test 7: Fehlerbehandlung - Leere Tags
    print("\n7. Fehlerbehandlung - Leere Tags")
    print("-" * 40)
    try:
        result = client.classify([], categories)
        print(f"✗ Sollte einen Fehler werfen")
    except requests.exceptions.HTTPError as e:
        print(f"✓ Erwarteter Fehler: {e.response.status_code}")
        print(f"  Details: {e.response.json()['detail']}")
    
    # Test 8: Fehlerbehandlung - Zu wenig Kategorien
    print("\n8. Fehlerbehandlung - Zu wenig Kategorien")
    print("-" * 40)
    try:
        result = client.classify(tags, ["nur_eine_kategorie"])
        print(f"✗ Sollte einen Fehler werfen")
    except requests.exceptions.HTTPError as e:
        print(f"✓ Erwarteter Fehler: {e.response.status_code}")
        print(f"  Details: {e.response.json()['detail']}")
    
    print("\n" + "=" * 60)
    print("Tests abgeschlossen!")
    print("=" * 60)


def curl_examples():
    """Gibt cURL-Beispiele aus"""
    
    print("\n" + "=" * 60)
    print("cURL Beispiele")
    print("=" * 60)
    
    print("\n# Health Check")
    print('curl http://localhost:8766/')
    
    print("\n# API Info")
    print('curl http://localhost:8766/info')
    
    print("\n# Einfache Klassifizierung")
    print('''curl -X POST http://localhost:8766/classify \\
  -H "Content-Type: application/json" \\
  -d '{
    "tags": ["dog", "park", "outdoor"],
    "categories": ["animals", "landscape", "portrait", "architecture"]
  }' ''')
    
    print("\n# Mit allen Scores")
    print('''curl -X POST http://localhost:8766/classify \\
  -H "Content-Type: application/json" \\
  -d '{
    "tags": ["sunset", "mountains", "hiking"],
    "categories": ["animals", "landscape", "portrait", "architecture"],
    "return_scores": true
  }' ''')


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--curl":
        curl_examples()
    else:
        print("\nHinweis: Server muss laufen (python server.py)")
        print("Warte 2 Sekunden bevor Tests starten...\n")
        import time
        time.sleep(2)
        
        test_api()
        
        print("\n\nFür cURL-Beispiele: python test_api.py --curl")
