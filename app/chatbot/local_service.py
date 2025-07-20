# =============================================================================
# FILE: app/chatbot/local_service.py (NEW FILE - PURELY LOCAL RAG)
# =============================================================================
import faiss
import numpy as np
from functools import lru_cache
from sentence_transformers import SentenceTransformer

from ..utils.knowledge import load_knowledge_base

class LocalKnowledgeBot:
    def __init__(self):
        print("INFO: Initializing Local Knowledge Bot...")
        
        # 1. Load the sentence transformer model for creating embeddings
        # This model is downloaded and runs entirely on your machine.
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # 2. Load and process the knowledge base data
        self.raw_data, self.processed_chunks = self._load_and_process_kb()
        
        # 3. Build the vector store for efficient searching
        self.vector_store = self._build_vector_store()
        print("INFO: Local Knowledge Bot initialized successfully.")

    def _load_and_process_kb(self):
        """Loads data from JSON and prepares it for indexing."""
        kb = load_knowledge_base()
        raw_data = {}
        processed_chunks = []
        for crop, diseases in kb.items():
            for disease_info in diseases:
                # We use the 'name' as a unique key to retrieve the full structured data later
                key = disease_info["name"]
                raw_data[key] = disease_info
                
                # Create a comprehensive text chunk for embedding. The more info, the better the match.
                chunk = (
                    f"Disease: {disease_info['name'].replace('_', ' ')}. "
                    f"Crop: {crop}. "
                    f"Symptoms and effects: {disease_info['effects']}. "
                    f"Caused by: {disease_info['causes']}. "
                    f"How to diagnose: {disease_info['diagnosis']}. "
                    f"Chemical treatment: {disease_info['recommended_chemical_treatment']}."
                )
                processed_chunks.append({"key": key, "text": chunk})
        return raw_data, processed_chunks

    def _build_vector_store(self):
        """Creates embeddings and builds a FAISS index for local search."""
        texts = [chunk['text'] for chunk in self.processed_chunks]
        print(f"INFO: Generating local embeddings for {len(texts)} knowledge base entries...")
        embeddings = self.embedding_model.encode(texts, convert_to_tensor=False)
        
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(np.array(embeddings, dtype=np.float32))
        print("INFO: Local FAISS vector store built.")
        return index

    def find_best_match(self, query: str) -> dict | None:
        """
        Finds the most relevant knowledge base entry for a query and returns its structured data.
        """
        query_embedding = self.embedding_model.encode([query])
        # Search for the single best match (k=1)
        _, I = self.vector_store.search(np.array(query_embedding, dtype=np.float32), 1)
        
        if not I.size:
            return None
            
        best_match_index = I[0][0]
        # Get the unique key of the best match
        context_key = self.processed_chunks[best_match_index]['key']
        
        print(f"DEBUG: Chatbot query '{query}' matched best with KB entry: '{context_key}'")
        # Return the complete, original dictionary for that entry
        return self.raw_data.get(context_key)

    def format_answer(self, query: str, disease_info: dict) -> str:
        """
        Formats a response based on keywords in the user's query, using the provided disease info.
        This replaces the need for an external LLM.
        """
        query = query.lower()
        disease_name = disease_info['name'].replace('_', ' ')

        if "treat" in query or "cure" in query or "chemical" in query or "manage" in query:
            return f"For {disease_name}, the recommended chemical treatment is: {disease_info['recommended_chemical_treatment']}"
        
        if "symptom" in query or "effect" in query or "look like" in query or "signs" in query:
            return f"The main effects and symptoms for {disease_name} are: {disease_info['effects']}"
            
        if "cause" in query or "spread" in query or "why" in query:
            return f"{disease_name} is typically caused by: {disease_info['causes']}"
            
        if "diagnos" in query or "identif" in query: # Catches 'diagnose' and 'identify'
            return f"To diagnose {disease_name}, look for the following: {disease_info['diagnosis']}"

        # Default fallback answer
        return f"Here is some general information about {disease_name}: {disease_info['information']}"

# Use lru_cache to ensure we only create ONE instance of the chatbot service
@lru_cache()
def get_bot_service():
    return LocalKnowledgeBot()