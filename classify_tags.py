"""
Tag Classifier - Ordnet Bildertags einer Kategorie zu

Verwendet das Zero-Shot Classification Modell MoritzLaurer/deberta-v3-base-zeroshot-v2.0
um eine Liste von Bildertags der wahrscheinlichsten Kategorie zuzuordnen.
"""

from transformers import pipeline
from typing import List, Dict, Optional
import warnings
import torch

warnings.filterwarnings("ignore")


class TagClassifier:
    """Klassifiziert Bildertags in Kategorien mittels Zero-Shot Classification"""
    
    def __init__(self, model_name: str = "MoritzLaurer/deberta-v3-base-zeroshot-v2.0"):
        """
        Initialisiert den Tag Classifier
        
        Args:
            model_name: Name des Hugging Face Modells für Zero-Shot Classification
        """
        print(f"Lade Modell: {model_name}...")
        # Automatische Device-Auswahl: GPU wenn verfügbar, sonst CPU
        device = 0 if torch.cuda.is_available() else -1
        self.classifier = pipeline(
            "zero-shot-classification",
            model=model_name,
            device=device
        )
        print("Modell erfolgreich geladen!")
        
        # Ausgabe des verwendeten Devices
        device_info = self.classifier.model.device
        if device_info.type == 'cuda':
            print(f"Modell läuft auf GPU (Device: {device_info})")
        else:
            print(f"Modell läuft auf CPU (Device: {device_info})")
    
    def classify_tags(
        self,
        tags: List[str],
        categories: List[str],
        multi_label: bool = False,
        return_scores: bool = False
    ) -> Dict:
        """
        Klassifiziert eine Liste von Tags in eine Kategorie
        
        Args:
            tags: Liste von Bildertags (z.B. ["dog", "outdoor", "sunny"])
            categories: Liste möglicher Kategorien (z.B. ["animals", "landscape", "portrait"])
            multi_label: Wenn True, können mehrere Kategorien gleichzeitig zutreffen
            return_scores: Wenn True, werden alle Scores zurückgegeben
        
        Returns:
            Dictionary mit:
                - category: Die beste Kategorie
                - score: Confidence Score (0-1)
                - all_scores: Alle Scores (optional, wenn return_scores=True)
        """
        # Tags zu einem Text zusammenfügen
        text = ", ".join(tags)
        
        # Zero-Shot Classification durchführen
        result = self.classifier(
            text,
            candidate_labels=categories,
            multi_label=multi_label
        )
        
        # Ergebnis formatieren
        output = {
            "category": result["labels"][0],
            "score": result["scores"][0]
        }
        
        if return_scores:
            output["all_scores"] = {
                label: score 
                for label, score in zip(result["labels"], result["scores"])
            }
        
        return output
    
    def classify_single_tag(
        self,
        tag: str,
        categories: List[str],
        return_scores: bool = False
    ) -> Dict:
        """
        Klassifiziert einen einzelnen Tag
        
        Args:
            tag: Einzelner Tag (z.B. "dog")
            categories: Liste möglicher Kategorien
            return_scores: Wenn True, werden alle Scores zurückgegeben
        
        Returns:
            Dictionary mit Kategorie und Score
        """
        return self.classify_tags([tag], categories, return_scores=return_scores)


def main():
    """Beispiel-Nutzung des Tag Classifiers"""
    
    # Classifier initialisieren
    classifier = TagClassifier()
    
    # Beispiel 1: Klassifizierung von Bildertags
    print("\n=== Beispiel 1: Mehrere Tags ===")
    tags = ["dog", "park", "playing", "outdoor", "sunny"]
    categories = ["animals", "landscape", "portrait", "architecture", "food", "sports"]
    
    result = classifier.classify_tags(tags, categories, return_scores=True)
    
    print(f"Tags: {tags}")
    print(f"Beste Kategorie: {result['category']}")
    print(f"Confidence: {result['score']:.2%}")
    print("\nAlle Scores:")
    for cat, score in result['all_scores'].items():
        print(f"  {cat}: {score:.2%}")
    
    # Beispiel 2: Einzelner Tag
    print("\n=== Beispiel 2: Einzelner Tag ===")
    tag = "sunset"
    result = classifier.classify_single_tag(tag, categories, return_scores=True)
    
    print(f"Tag: {tag}")
    print(f"Beste Kategorie: {result['category']}")
    print(f"Confidence: {result['score']:.2%}")
    
    # Beispiel 3: Verschiedene Kategorien
    print("\n=== Beispiel 3: Portrait Tags ===")
    tags = ["woman", "smiling", "close-up", "face"]
    result = classifier.classify_tags(tags, categories, return_scores=True)
    
    print(f"Tags: {tags}")
    print(f"Beste Kategorie: {result['category']}")
    print(f"Confidence: {result['score']:.2%}")
    
    # Beispiel 4: Food Tags
    print("\n=== Beispiel 4: Food Tags ===")
    tags = ["pizza", "cheese", "restaurant", "delicious"]
    result = classifier.classify_tags(tags, categories, return_scores=True)
    
    print(f"Tags: {tags}")
    print(f"Beste Kategorie: {result['category']}")
    print(f"Confidence: {result['score']:.2%}")


if __name__ == "__main__":
    main()
