"""
Module for rag 
"""
from .configs import Configs
from langchain.vectorstores import FAISS

configs = Configs()

class RAGPipeline:

    def __init__(self, faiss_index: FAISS, json_data: list, index_path: str):
        self.faiss_index = faiss_index
        self.embedding_model = configs.embedding_model
        self.json_data = json_data
        self.index_path = index_path

    def format_text(self):
        texts = [
            f"Crop: {entry['crop']}\nDisease: {entry['disease']}\n\nSymptoms:\n{entry['symptoms']}\n\nCause:\n{entry['cause']}\n\nManagement:\n{entry['management']}"
            for entry in self.json_data
        ]
        return texts

    def _create_faiss_index(self):
        texts = self.format_text()
        faiss_index = self.faiss_index.from_texts(texts, self.embedding_model)
        return faiss_index   

    def save_faiss_index(self):
        faiss_index = self._create_faiss_index()
        faiss_index.save_local(self.index_path)
        return f"FAISS index saved to {self.index_path}"

    def _load_faiss_index(self):
        return self.faiss_index.load_local("faiss_index", embeddings=self.embedding_model, 
        allow_dangerous_deserialization=True,
        )

    def search(self, query: str, k: int = 1):
        faiss_index = self._load_faiss_index()
        return faiss_index.similarity_search(query, k=k)

    def content_retriever(self, query: str, k: int = 1):
        faiss_index = self._load_faiss_index()
        retriever = faiss_index.as_retriever(search_type="similarity", search_kwargs={"k": k})
        rel_docs = retriever.get_relevant_documents(query)
        contents = [doc.page_content for doc in rel_docs]
        return contents
