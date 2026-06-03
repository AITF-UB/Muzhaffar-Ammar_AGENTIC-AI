import os
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from huggingface_hub import InferenceClient
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.outputs import ChatResult, ChatGeneration
from dotenv import load_dotenv

load_dotenv()

# =====================================================================
# 1. HUGGING FACE INFERENCE (CUSTOM CLASS)
# =====================================================================
class HFChatModel(BaseChatModel):
    client: InferenceClient = None
    model_id: str = os.getenv("HF_MODEL_ID", "meta-llama/Llama-3.1-8B-Instruct")
    temperature: float = 0.3
    max_tokens: int = 4000

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        hf_token = os.getenv("HF_TOKEN")
        self.client = InferenceClient(model=self.model_id, token=hf_token)

    @property
    def _llm_type(self) -> str:
        return "hf-chat"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
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


# =====================================================================
# FACTORY PATTERN: GET LLM BASED ON .ENV
# =====================================================================
def get_llm():
    provider = os.getenv("LLM_PROVIDER", "huggingface").lower().strip()

    if provider == "bedrock":
        import boto3
        from langchain_aws import ChatBedrock
        
        bedrock_client = boto3.client(
            service_name="bedrock-runtime",
            region_name=os.getenv("BEDROCK_REGION", "us-east-1"),
            endpoint_url=os.getenv("BEDROCK_ENDPOINT_URL"),
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
        )
        return ChatBedrock(
            client=bedrock_client,
            model_id=os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-3-haiku-20240307-v1:0"),
            model_kwargs={"temperature": 0.3, "max_tokens": 4000}
        )

    elif provider == "runpod":
        from langchain_openai import ChatOpenAI
        
        return ChatOpenAI(
            openai_api_base=os.getenv("RUNPOD_ENDPOINT_URL"),
            openai_api_key=os.getenv("RUNPOD_API_KEY", "empty"),
            model_name=os.getenv("RUNPOD_MODEL_ID", "meta-llama/Meta-Llama-3-8B-Instruct"),
            temperature=0.3,
            max_tokens=4000
        )

    elif provider == "kaggle_vllm":
        from langchain_openai import ChatOpenAI
        
        # vLLM di Kaggle menyediakan endpoint OpenAI-compatible
        return ChatOpenAI(
            openai_api_base=os.getenv("NGROK_KAGGLE_VLLM"),
            openai_api_key="empty", # vLLM lokal tidak butuh api key
            model_name=os.getenv("KAGGLE_VLLM_MODEL_ID", "AITF-SR-02/ub-sr-02-qwen3.5-9b-base-5k-CPT-SFT-v2"),
            temperature=0.3,
            max_tokens=4000
        )

    elif provider == "kaggle_ollama":
        from langchain_community.chat_models import ChatOllama
        
        return ChatOllama(
            base_url=os.getenv("NGROK_KAGGLE_OLLAMA"),
            model=os.getenv("KAGGLE_OLLAMA_MODEL_ID", "qwen3.5:9b"), 
            temperature=0.3
        )

    else:
        # Default fallback ke Hugging Face Inference
        return HFChatModel()
