from .configs import Configs
import os

current_dir = os.path.dirname(__file__)
prompt_path = os.path.join(current_dir, "..", "prompt_templates", "main_prompt_template.txt")
prompt_path = os.path.abspath(prompt_path)


configs = Configs()

def chat_Toutbot(user_query):
    
    relevant_docs = configs.retriever.get_relevant_documents(user_query)
    context = [doc.page_content for doc in relevant_docs]

    # loading the prompt
    with open(prompt_path, "r", encoding="utf-8") as f:
        tout_system_prompt = f.read().strip()
    tout_system_prompt += f"Use the following context to answer:\n{context}"

    # print(tout_system_prompt)

    msg_to_model = [{"role": "system", "content": tout_system_prompt}, {"role": "user", "content": user_query}]

    model = configs.client.chat.completions.create(
        model=configs.rag_model_1,
        messages=msg_to_model,
    )

    toutbot_response = model.choices[0].message.content
    
    return toutbot_response 

if __name__ == "__main__":
    response = chat_Toutbot("what is fall army worm?")
    print("\nToutbot Response:")
    print(response)