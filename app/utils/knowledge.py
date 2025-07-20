import json
from functools import lru_cache
from pathlib import Path

from ..core.config import settings

@lru_cache()
def load_knowledge_base() -> dict:
    """
    Loads the knowledge base from the JSON file.
    Uses lru_cache to ensure the file is only read from disk once.
    """
    kb_path = settings.KNOWLEDGE_BASE_PATH 
    if not kb_path.exists():
        print(f"ERROR: Knowledge base file not found at {kb_path}")
        return {}
    
    try:
        with open(kb_path, "r", encoding="utf-8") as f:
            knowledge_base = json.load(f)
        print("INFO:     Knowledge base loaded successfully.")
        return knowledge_base
    except Exception as e:
        print(f"ERROR: An unexpected error occurred while loading the knowledge base: {e}")
        return {}

def get_disease_info(disease_key: str) -> dict | None:
    """
    Retrieves information for a specific disease from the structured knowledge base.

    Args:
        disease_key (str): The key from the model's prediction 
                           (e.g., 'Tomato_leaf_blight').

    Returns:
        A dictionary with the disease information, or None if not found.
    """
    if not disease_key or "_" not in disease_key:
        return None

    kb = load_knowledge_base()
    
    crop = disease_key.split('_')[0].capitalize()
    
    if crop not in kb:
        return None
        
    for disease_entry in kb[crop]:
        if disease_entry.get("name") == disease_key:
            disease_entry['crop'] = crop 
            return disease_entry
            
    return None