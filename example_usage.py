"""
Erweiterte Beispiele für die Nutzung des Tag Classifiers
"""

from classify_tags import TagClassifier


def batch_classification_example():
    """Beispiel: Batch-Klassifizierung mehrerer Bilder"""
    print("=== Batch Klassifizierung ===\n")
    
    classifier = TagClassifier()
    
    # Mehrere Bilder mit ihren Tags
    images = [
        {
            "image_id": "IMG_001",
            "tags": ["cat", "sitting", "window", "indoor"]
        },
        {
            "image_id": "IMG_002", 
            "tags": ["mountains", "snow", "landscape", "winter"]
        },
        {
            "image_id": "IMG_003",
            "tags": ["man", "woman", "wedding", "happy"]
        },
        {
            "image_id": "IMG_004",
            "tags": ["pasta", "tomatoes", "basil", "plate"]
        }
    ]
    
    categories = ["animals", "nature", "people", "food", "architecture"]
    
    # Alle Bilder klassifizieren
    for image in images:
        result = classifier.classify_tags(image["tags"], categories)
        print(f"{image['image_id']}: {', '.join(image['tags'])}")
        print(f"  → Kategorie: {result['category']} ({result['score']:.1%})\n")


def threshold_filtering_example():
    """Beispiel: Nur Ergebnisse über einem Schwellenwert akzeptieren"""
    print("\n=== Schwellenwert-Filterung ===\n")
    
    classifier = TagClassifier()
    categories = ["sports", "music", "technology", "nature"]
    
    test_cases = [
        ["football", "stadium", "players"],
        ["guitar", "concert", "stage"],
        ["laptop", "keyboard", "mouse"],
        ["trees", "forest", "hiking"],
        ["abstract", "colorful", "pattern"]  # Ambiguous case
    ]
    
    threshold = 0.5  # Mindest-Confidence
    
    for tags in test_cases:
        result = classifier.classify_tags(tags, categories)
        
        print(f"Tags: {', '.join(tags)}")
        if result['score'] >= threshold:
            print(f"  ✓ Kategorie: {result['category']} ({result['score']:.1%})")
        else:
            print(f"  ✗ Unsicher - Score zu niedrig ({result['score']:.1%})")
        print()


def custom_categories_example():
    """Beispiel: Verwendung mit eigenen Kategorien"""
    print("\n=== Eigene Kategorien ===\n")
    
    classifier = TagClassifier()
    
    # Spezifische Kategorien für einen Foto-Service
    categories = [
        "product photography",
        "real estate",
        "events and celebrations",
        "nature and wildlife",
        "automotive"
    ]
    
    test_cases = [
        ["car", "red", "sports car", "garage"],
        ["house", "garden", "exterior", "for sale"],
        ["birthday", "cake", "celebration", "children"],
        ["shoes", "white background", "product"]
    ]
    
    for tags in test_cases:
        result = classifier.classify_tags(tags, categories, return_scores=True)
        print(f"Tags: {', '.join(tags)}")
        print(f"  Kategorie: {result['category']} ({result['score']:.1%})")
        
        # Top 3 Kategorien anzeigen
        top_3 = list(result['all_scores'].items())[:3]
        print("  Top 3:")
        for cat, score in top_3:
            print(f"    {cat}: {score:.1%}")
        print()


def multi_language_example():
    """Beispiel: Mehrsprachige Tags (Deutsch)"""
    print("\n=== Mehrsprachige Tags ===\n")
    
    classifier = TagClassifier()
    
    # Deutsche Kategorien
    categories = ["Tiere", "Landschaft", "Portrait", "Essen", "Architektur"]
    
    test_cases = [
        ["Hund", "Park", "spielend"],
        ["Berge", "Sonnenuntergang", "Wandern"],
        ["Frau", "lächelnd", "Nahaufnahme"],
        ["Pizza", "Restaurant", "italienisch"]
    ]
    
    for tags in test_cases:
        result = classifier.classify_tags(tags, categories)
        print(f"Tags: {', '.join(tags)}")
        print(f"  Kategorie: {result['category']} ({result['score']:.1%})\n")


if __name__ == "__main__":
    # Alle Beispiele ausführen
    batch_classification_example()
    threshold_filtering_example()
    custom_categories_example()
    multi_language_example()
