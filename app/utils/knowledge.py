import json
from functools import lru_cache
from pathlib import Path

# This assumes your config file is in app/core/config.py
# If your project structure is different, you may need to adjust the import path.
from ..core.config import settings

@lru_cache()
def load_knowledge_base() -> dict:
    """
    Loads the knowledge base from the JSON file.
    Uses lru_cache to ensure the file is only read from disk once.
    """
    # This path is defined in your config.pyaz
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
    
    # 1. Determine the crop from the disease_key
    # Example: "Tomato_leaf_blight" -> "Tomato"
    crop = disease_key.split('_')[0].capitalize()
    
    # 2. Check if the crop exists in our knowledge base
    if crop not in kb:
        return None
        
    # 3. Find the specific disease entry in the list for that crop
    for disease_entry in kb[crop]:
        if disease_entry.get("name") == disease_key:
            # Add the crop to the dictionary for easier use in the template
            disease_entry['crop'] = crop 
            return disease_entry
            
    # Return None if no matching disease name was found for that crop
    return None