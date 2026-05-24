import os
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from huggingface_hub import InferenceClient
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.outputs import ChatResult, ChatGeneration
import boto3
from langchain_aws import ChatBedrock
from dotenv import load_dotenv
import ollama
load_dotenv()


HF_TOKEN = os.getenv("HF_TOKEN")
from langchain_community.chat_models import ChatOllama

# class OllamaChatModel(ChatOllama):
#     # Kita menggunakan ChatOllama bawaan langchain-community karena lebih stabil 
#     # menghadapi isu SSL HTTPS Ngrok di Windows dibanding ollama-python native.
#     def __init__(self, **kwargs):
#         # Override default parameter di sini
#         super().__init__(
#             base_url=os.getenv("NGROK_KAGGLE_OLLAMA"),
#             model="qwen3.5:9b", # Ganti jika nama modelnya berbeda di Kaggle
#             temperature=0.3,
#             **kwargs
#         )


# Ambil dari .env
# AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
# AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
# BEDROCK_ENDPOINT_URL = os.getenv("BEDROCK_ENDPOINT_URL")
# BEDROCK_REGION = os.getenv("BEDROCK_REGION", "us-east-1")
# BEDROCK_MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-3-haiku-20240307-v1:0")

# def get_llm():
#     # 1. Bikin custom boto3 client yang ngarah ke URL khusus
#     bedrock_client = boto3.client(
#         service_name="bedrock-runtime",
#         region_name=BEDROCK_REGION,
#         endpoint_url=BEDROCK_ENDPOINT_URL,
#         aws_access_key_id=AWS_ACCESS_KEY_ID,
#         aws_secret_access_key=AWS_SECRET_ACCESS_KEY
#     )

#     # 2. Masukkan client tersebut ke ChatBedrock bawaan Langchain
#     llm = ChatBedrock(
#         client=bedrock_client,
#         model_id=BEDROCK_MODEL_ID,
#         model_kwargs={
#             "temperature": 0.3,
#             "max_tokens": 4000  # Pastikan max_tokens cukup panjang
#         }
#     )
    
#     return llm


class HFChatModel(BaseChatModel):
    client: InferenceClient = None
    model_id: str = "meta-llama/Llama-3.1-8B-Instruct"
    temperature: float = 0.3
    max_tokens: int = 4000

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.client = InferenceClient(model=self.model_id, token=HF_TOKEN)

    @property
    def _llm_type(self) -> str:
        return "hf-chat"

    def _generate(self, messages, stop=None, **kwargs) -> ChatResult:
        hf_msgs = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                hf_msgs.append({"role": "system", "content": msg.content})
            elif isinstance(msg, HumanMessage):
                hf_msgs.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                hf_msgs.append({"role": "assistant", "content": msg.content})
            else:
                hf_msgs.append({"role": "user", "content": str(msg.content)})
        
        response = self.client.chat_completion(
            messages=hf_msgs,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            stop=stop
        )
        output_text = response.choices[0].message.content
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=output_text))])

def get_llm():
    # return OllamaChatModel()
    return HFChatModel()
