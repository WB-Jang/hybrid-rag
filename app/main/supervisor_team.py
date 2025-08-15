from llms.llms import *

from operator import add
from typing import List, TypedDict, Annotated, Optional, Tuple
from textwrap import dedent

from memory_update.routing_node import SupervisorTeamState, load_memory_node,update_memory_node, classify_question, route_datasources_tool_search, analyze_question_tool_search, classify_to_next
from agents.banking_law_agent import BankingRagState, banking_law_agent
from agents.fss_standards_agent import FssRagState, fss_standards_agent
from agents.personal_law_agent import PersonalRagState, personal_law_agent
#from agents.web_search_agent import SearchRagState, search_web_agent
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langgraph.checkpoint.base import Checkpoint

max_answer_items = 3
# 노드 정의 

import logging
from pprint import pformat

logging.basicConfig(level=logging.DEBUG)

def fss_rag_node(state: SupervisorTeamState) -> SupervisorTeamState:
    print("--- 은행업감독규정시행세칙 전문가 에이전트 시작 ---")
    logging.debug(f"▶▶ Before agent answer:\n{pformat(state)}")
    question = state["question"]
    context = state.get("context_prompt", []) 
    answer = fss_standards_agent.invoke({
        "question": question,
        "context_prompt": context
        })
    ans = answer["extracted_info"]
    current_answers = state.get("fss_answers", [])
    updated_answers = current_answers + ([ans] if ans else [])
    logging.debug(f"▶▶ After agent current answer:\n{pformat(current_answers)}")
    logging.debug(f"▶▶ After agent updated answer:\n{pformat(updated_answers)}")
 
    return {"fss_answers": updated_answers} 

def personal_rag_node(state: SupervisorTeamState) -> SupervisorTeamState:
    print("--- 개인정보보호법 전문가 에이전트 시작 ---")
    logging.debug(f"▶▶ Before agent answer:\n{pformat(state)}")
    question = state["question"]
    context = state.get("context_prompt", [])
    answer = personal_law_agent.invoke({
        "question": question,
        "context_prompt": context
        })
    ans = answer["extracted_info"]
    current_answers = state.get("personal_answers", [])
    updated_answers = current_answers + ([ans] if ans else [])
    logging.debug(f"▶▶ After agent current answer:\n{pformat(current_answers)}")
    logging.debug(f"▶▶ After agent updated answer:\n{pformat(updated_answers)}")
    return {"personal_answers": updated_answers} 
    

def banking_rag_node(state: SupervisorTeamState) -> SupervisorTeamState:
    print("--- 은행법 전문가 에이전트 시작 ---")
    logging.debug(f"▶▶ Before agent answer:\n{pformat(state)}")
    question = state["question"]
    context = state.get("context_prompt", [])
    answer =  banking_law_agent.invoke({
        "question": question,
        "context_prompt": context
        })
    ans = answer["extracted_info"]
    current_answers = state.get("banking_answers", [])
    updated_answers = current_answers + ([ans] if ans else [])
    logging.debug(f"▶▶ After agent current answer:\n{pformat(current_answers)}")
    logging.debug(f"▶▶ After agent updated answer:\n{pformat(updated_answers)}")
    return {"banking_answers": updated_answers} 

# def web_rag_node(state: SupervisorTeamState) -> SupervisorTeamState:
#     print("--- 인터넷 검색 전문가 에이전트 시작 ---")
#     logging.debug(f"▶▶ Before agent answer:\n{pformat(state)}")
#     question = state["question"]
#     context = state.get("context_prompt", [])
#     answer = search_web_agent.invoke({
#         "question": question,
#         "context_prompt": context
#         })
#     ans = answer["node_answer"]
#     current_answers = state.get("answers", [])
#     updated_answers = current_answers + ([ans] if ans else [])
#     logging.debug(f"▶▶ After agent current answer:\n{pformat(current_answers)}")
#     logging.debug(f"▶▶ After agent updated answer:\n{pformat(updated_answers)}")
#     return {"answers": updated_answers} 
   
# 최종 답변 생성 노드
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# RAG 프롬프트 정의
rag_prompt = ChatPromptTemplate.from_messages([
    ("system", """당신은 은행법 전문가입니다. 주어진 질문과 추출된 정보를 바탕으로 답변을 생성해주세요. 
         답변은 마크다운 형식으로 작성하며, 각 정보의 출처를 명확히 표시해야 합니다. 
         답변 구조:
         1. 질문에 대한 직접적인 답변
         2. 관련 법률 조항 및 해석
         3. 추가 설명 또는 예시 (필요한 경우)
         4. 결론 및 요약
         각 섹션에서 사용된 정보의 출처를 괄호 안에 명시하세요. 예: (출처: 주택임대차보호법 제15조)"""),
         ("human", "Answer the following question using these documents:\n\n[context]\n{context}\n\n[Documents]\n{documents}\n\n[Question]\n{question}")
     ])
from itertools import chain

def answer_final(state: SupervisorTeamState) -> SupervisorTeamState:
    """
    Generate answer using the retrieved_documents
    """
    print("---최종 답변---")
    question = state["question"]
    documents = [state.get("fss_answers", []),state.get("personal_answers", []),state.get("banking_answers", [])]
    all_docs = list(chain.from_iterable(documents))
    context = state.get("context_prompt")
    if not isinstance(all_docs, list):
        documents = [all_docs]

    # 문서 내용을 문자열로 결합 
    documents_text = "\n\n".join(all_docs)
    context_text = "\n\n".join(context)

    # RAG generation
    rag_chain = rag_prompt | gpt3_5_turbo_llm | StrOutputParser()
    generation = rag_chain.invoke({"context": context_text,"documents": documents_text, "question": question})
    logging.debug(f"▶▶ After final answers generated :\n{pformat(state)}")
    
    return {"final_answer": generation, "question":question, "answers": []}

def reset_answers_node(state: SupervisorTeamState, config: Checkpoint) -> SupervisorTeamState:
    mutable_state = state.copy()
    mutable_state.pop("fss_answers", None)
    mutable_state["fss_answers"]=[]
    print("fss_answers 초기화 완료:", mutable_state["fss_answers"])
    mutable_state.pop("personal_answers", None)
    mutable_state["personal_answers"]=[]
    print("personal_answers 초기화 완료:", mutable_state["personal_answers"])
    mutable_state.pop("banking_answers", None)
    mutable_state["banking_answers"]=[]
    print("banking_answers 초기화 완료:", mutable_state["banking_answers"])

    return mutable_state
# LLM Fallback 프롬프트 정의
fallback_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an AI assistant helping with various topics. Follow these guidelines:

1. Provide accurate and helpful information to the best of your ability.
2. Express uncertainty when unsure; avoid speculation.
3. Keep answers concise yet informative.
4. Respond ethically and constructively.
5. Mention reliable general sources when applicable."""),
    ("human", "question :{question}\n context: {context}"),
])

def llm_fallback(state: SupervisorTeamState) -> SupervisorTeamState:
    """
    Generate answer using the LLM without context
    """
    print("---Fallback 답변---")
    question = state["question"]
    context = state.get("context_prompt", [])

    llm_chain = fallback_prompt | gpt4o_mini_llm | StrOutputParser()

    generation = llm_chain.invoke({"question": question, "context": context})
    return {"final_answer": generation, "question":question}

# 노드 정의를 딕셔너리로 관리
nodes = {
    "load_memory": load_memory_node,
    "classify_question": classify_question,
    "analyze_question": analyze_question_tool_search,
    "search_personal": personal_rag_node,
    "search_fss": fss_rag_node,
    "search_banking": banking_rag_node,
    #"search_web": web_rag_node,
    "update_memory": update_memory_node,
    "generate_answer": answer_final,
    "llm_fallback": llm_fallback,
    "reset_answers": reset_answers_node

}

from langgraph.graph import StateGraph, START, END

# 그래프 생성을 위한 StateGraph 객체를 정의
search_builder = StateGraph(SupervisorTeamState)

# 노드 추가
for node_name, node_func in nodes.items():
    search_builder.add_node(node_name, node_func)

# 엣지 추가 (대화 히스토리 관리 및 질문 분류 추가)
search_builder.add_edge(START, "reset_answers")
search_builder.add_edge("reset_answers", "load_memory")
search_builder.add_edge("load_memory", "classify_question")
search_builder.add_conditional_edges(
    "classify_question",
    classify_to_next,
    { "llm_fallback": "llm_fallback", "analyze_question": "analyze_question" }
)

# 조건부 엣지 추가
search_builder.add_conditional_edges(
    "analyze_question",
    route_datasources_tool_search,
    ["search_personal", "search_fss", "search_banking"] #"search_web",  "llm_fallback"
)

# 검색 노드들을 generate_answer에 연결
for node in ["search_personal", "search_fss", "search_banking"]: #, "search_web"
    search_builder.add_edge(node, "generate_answer")

search_builder.add_edge("generate_answer", "update_memory")
search_builder.add_edge("llm_fallback", "update_memory") 
search_builder.add_edge("update_memory", END)


from langgraph.checkpoint.memory import MemorySaver
# 메모리 추가
memory = MemorySaver()

# 그래프 컴파일
legal_rag_agent = search_builder.compile(
    checkpointer=memory
    )
