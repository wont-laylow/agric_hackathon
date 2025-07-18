import os
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import openai
from langchain.vectorstores import FAISS


load_dotenv(override=True)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

current_dir = os.path.dirname(__file__)
faiss_index_path = os.path.join(current_dir, "..", "..", "faiss_index")
faiss_index_path = os.path.abspath(faiss_index_path)


class Configs:

    def __init__(self):
        self.faiss_index = FAISS.load_local(faiss_index_path, 
        OpenAIEmbeddings(api_key=OPENAI_API_KEY),
        allow_dangerous_deserialization=True,
        )
        self.embedding_model = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

        self.rag_model_1 = "gpt-4o-mini"

        self.rag_model_2 = "gpt-3.5-turbo"
        self.client = openai.OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
        )

        self.retriever = self. faiss_index.as_retriever(search_type="similarity", search_kwargs={"k": 1})

    

    



