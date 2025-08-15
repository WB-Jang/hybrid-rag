from llms.llms import *
from typing import List

from sklearn.metrics.pairwise import cosine_similarity

from langchain_chroma import Chroma
from langchain_core.tools import tool
from langchain_core.documents import Document
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.retrievers import TavilySearchAPIRetriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder


rerank_model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-v2-m3")
cross_reranker = CrossEncoderReranker(model=rerank_model, top_n=10)

embeddings_model = OllamaEmbeddings(model="bge-m3")

# 4-1) 벡터 DB(Chroma) 불러오기
fss_standards_db = Chroma(
    embedding_function=embeddings_model,
    collection_name="fss_contextual_standards",
    persist_directory="./chroma_db"
)

base_retriever = fss_standards_db.as_retriever(search_kwargs={"k": 30})
fss_standards_db_retriever = ContextualCompressionRetriever(
    base_compressor=cross_reranker,
    base_retriever=base_retriever
)

@tool
def fss_standards_search(query: str) -> List[Document]:
    """은행업 감독규정 시행세칙 별표3 관련 조항을 검색합니다.
    은행업 감독규정 시행세칙은 은행이 리스크관리를 위하여 준수해야 하는 다양한 규제들을 어떻게 실행하는지 구체적으로 설명하는 문서입니다.
    위의 컨텍스트를 참고하여 도구를 선택하세요.
    """
    docs = fss_standards_db_retriever.invoke(query)

    if len(docs) > 0:
        return docs
    
    return [Document(page_content="관련 정보를 찾을 수 없습니다.")]




# 개인정보보호호법 검색 
personal_db = Chroma(
    embedding_function=embeddings_model,   
    collection_name="personal_law",
    persist_directory="./chroma_db",
)

personal_db_retriever = ContextualCompressionRetriever(
    base_compressor=cross_reranker, 
    base_retriever=personal_db.as_retriever(search_kwargs={"k":30}),
)


@tool
def personal_law_search(query: str) -> List[Document]:
    """개인정보보호호법 법률 조항을 검색합니다. 개인정보 보호와 관련된 법률 내용이 있습니다"""
    docs = personal_db_retriever.invoke(query)

    if len(docs) > 0:
        return docs
    
    return [Document(page_content="관련 정보를 찾을 수 없습니다.")]


# 은행법 검색 
banking_db = Chroma(
    embedding_function=embeddings_model,   
    collection_name="banking_law",
    persist_directory="./chroma_db",
)

banking_db_retriever = ContextualCompressionRetriever(
    base_compressor=cross_reranker, 
    base_retriever=banking_db.as_retriever(search_kwargs={"k":30}),
)



@tool
def banking_law_search(query: str) -> List[Document]:
    """은행법 법률 조항을 검색합니다. 은행 산업 설립과 기본적인 법률 근거에 대하여 포괄적으로 담고 있는 법 조항입니다"""
    docs = banking_db_retriever.invoke(query)

    if len(docs) > 0:
        return docs
    
    return [Document(page_content="관련 정보를 찾을 수 없습니다.")]



# 웹 검색
web_retriever = ContextualCompressionRetriever(
    base_compressor=cross_reranker, 
    base_retriever=TavilySearchAPIRetriever(k=30)
)

@tool
def web_search(query: str) -> List[str]:
    """문서 데이터베이스에 없는 정보 또는 실제 산업 현장에서의 최신 정보를 웹에서 검색합니다."""

    docs = web_retriever.invoke(query)

    formatted_docs = []
    for doc in docs:
        formatted_docs.append(
            Document(
                page_content= f'<Document href="{doc.metadata["source"]}"/>\n{doc.page_content}\n</Document>',
                metadata={"source": "web search", "url": doc.metadata["source"]}
            )
        )

    if len(formatted_docs) > 0:
        return formatted_docs
    
    return [Document(page_content="관련 정보를 찾을 수 없습니다.")]

# 도구 목록을 정의 
tools = [fss_standards_search, personal_law_search, banking_law_search, web_search]

llm_with_tools = llama3_8b_llm.bind_tools(tools)