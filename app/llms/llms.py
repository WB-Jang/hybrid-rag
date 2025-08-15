from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_ollama import OllamaLLM
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# 기본 LLM

dna_llm = OllamaLLM(
    model="dnotitia/dna",    # 터미널에서 `ollama list` 명령으로 확인한 모델 이름
    temperature=0.0
)

llama3_70b_llm = ChatGroq(model="llama3-70b-8192", temperature=0.0, max_tokens=3000, streaming=True)
llama3_8b_llm = ChatGroq(model="llama3-8b-8192", temperature=0.0, max_tokens=2000, streaming=True)
gpt4o_mini_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, streaming=True)
gpt4o_llm = ChatOpenAI(model="gpt-4o", temperature=0, streaming=True)
gpt3_5_turbo_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, streaming=True)
gemini_15_flash_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash",temperature=0,streaming=True)






