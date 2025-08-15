from llms.llms import *
from operator import add
from typing import List, TypedDict, Annotated, Optional, Tuple
from textwrap import dedent
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableConfig

from typing import TypedDict, Annotated, List, Tuple, Optional
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationSummaryBufferMemory
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph
from langgraph.checkpoint.base import Checkpoint
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

# 메인 그래프 상태 정의
class SupervisorTeamState(TypedDict):
    question: str
    fss_answers: Optional[List[str]]
    personal_answers: Optional[List[str]]
    banking_answers: Optional[List[str]]
    final_answer: str
    datasources: List[str]
    evaluation_report: Optional[dict]
    user_decision: Optional[str]
    conversation_history: Annotated[list[BaseMessage], add_messages]
    context_prompt: List[str]

class QuestionType(BaseModel):
    """질문의 유형을 분류합니다."""
    is_casual: bool = Field(description="일상적인 대화인지 여부")
    needs_search: bool = Field(description="문서 검색이 필요한지 여부")
    reason: str = Field(description="분류 이유")

session_memory = {}
summary_memory: dict = {}
RECENT_MESSAGE_COUNT = 5

def get_summary_memory(config: Checkpoint) -> ConversationSummaryBufferMemory:
    thread_id = config["configurable"]["thread_id"]
    if thread_id not in summary_memory:
        summary_memory[thread_id] = ConversationSummaryBufferMemory(
            llm=gpt4o_mini_llm,
            max_token_limit=128, 
            memory_key="conversation_summary",
            return_messages=False # 문자열(summary)만 리턴
        )
    return summary_memory[thread_id]

import logging
from pprint import pformat

logging.basicConfig(level=logging.DEBUG)

def load_memory_node(state: SupervisorTeamState, config: Checkpoint) -> SupervisorTeamState:
    tid = config["configurable"]["thread_id"]
    logging.debug(f"[load_memory_node] thread_id = {tid}")
    logging.debug(f"▶▶ Before load_memory_node:\n{pformat(state)}")

    # mutable_state = state.copy() # 상태는 직접 수정하지 않는 것이 좋으므로 복사
    # mutable_state.pop("answers", None) # 확실하게 제거
    # mutable_state["answers"] = []       # 빈 리스트로 재할당
    # print(f"DEBUG: load_memory_node 내부 answers 초기화 완료: {mutable_state['answers']}")
    # --- 추가 및 수정된 코드 끝 ---

    summary_mem = get_summary_memory(config)
    summary_vars = summary_mem.load_memory_variables({})
    summary_text = summary_vars.get("conversation_summary", "")

    history_msgs: list = state.get("conversation_history", [])
    recent_msgs = [msg.content for msg in history_msgs[-RECENT_MESSAGE_COUNT:]]

    prompt_parts = []
    if summary_text:
        prompt_parts.append(f"[요약된 이전 대화]\n{summary_text}")
    if recent_msgs:
        prompt_parts.append("[최근 대화]\n" + "\n".join(recent_msgs)) 
        #recent_msg가 너무 크게 쌓이는 문제가 있어서, 요약된 대화들은 recent_msg에 들어오지 않으면 좋겠다
    # state["answers"] = []
    state["context_prompt"] = prompt_parts

    logging.debug(f"▶▶ After  load_memory_node:\n{pformat(state)}")
    return state

def update_memory_node(state: SupervisorTeamState, config: Checkpoint) -> SupervisorTeamState:
    """
     1) state에서 현재 질문/답변을 가져온다.
     2) state의 conversation_history 리스트에 추가한다.
     3) 요약 메모리에 save_context를 호출한다.
     """
    tid = config["configurable"]["thread_id"]
    logging.debug(f"[update_memory_node] thread_id = {tid}")
    logging.debug(f"▶▶ Before update_memory_node:\n{pformat(state)}")

    question = state.get("question")
    final_answer = state.get("final_answer")

    if question is None or final_answer is None:
        return {}    
    
    history = state.get("conversation_history", [])
    history.append(HumanMessage(content=state["question"]))
    history.append(AIMessage(content=state["final_answer"]))

    summary_mem = get_summary_memory(config)
    summary_mem.save_context(
        {"input": state["question"]},
        {"output": state["final_answer"]}
    )
    logging.debug(f"▶▶ After  update_memory_node:\n{pformat(state)}")
    return {"conversation_history": history,
            "fss_answers": [],
            "personal_answers": [],
            "banking_answers": []
            
    }

def classify_question(state: SupervisorTeamState) -> SupervisorTeamState:
    """질문의 유형을 분류하여 적절한 처리 방법을 결정합니다."""
    classify_prompt = ChatPromptTemplate.from_messages([
        ("system", """당신은 사용자의 질문을 분석하여 적절한 처리 방법을 결정하는 전문가입니다.
        다음 기준에 따라 질문을 분류하세요:
        1. 일상적인 대화나 간단한 질문인 경우 (예: "안녕하세요", "날씨가 어때요?", "감사합니다")
        2. 법률이나 규정에 대한 구체적인 질문인 경우
        3. 전문적인 정보나 최신 데이터가 필요한 경우
        
        각 질문에 대해 다음 정보를 제공하세요:
        - is_casual: 일상적인 대화인지 여부
        - needs_search: 문서 검색이 필요한지 여부
        - reason: 분류 이유"""),
        ("human", "{question}")
    ])
    
    classifier = gpt4o_mini_llm.with_structured_output(QuestionType)
    result = classifier.invoke(classify_prompt.format(question=state["question"]))
    
    state["question_type"] = result
    return state

def classify_to_next(state: SupervisorTeamState) -> str:
    """
    classify_question() 이후에 state["question_type"]가 채워졌다고 가정.
    캐주얼(is_casual=True)이면 llm_fallback으로, 
    그렇지 않으면 analyze_question으로 분기.
    """
    qt = state.get("question_type", None)
    if qt and getattr(qt, "is_casual", False):
        return "llm_fallback"
    else:
        return "analyze_question"

from typing import Literal
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

# 라우팅 결정을 위한 데이터 모델
class ToolSelector(BaseModel):
    """Routes the user question to the most appropriate tool."""
    tool: Literal["search_fss", "search_personal", "search_banking"] = Field( #"search_web", "llm_fallback"
        description="Select one of the tools, based on the user's question.",
    )

class ToolSelectors(BaseModel):
    """Select the appropriate tools that are suitable for the user question."""
    tools: List[ToolSelector] = Field(
        description="Select one or more tools, based on the user's question.",
    )

# 구조화된 출력을 위한 LLM 설정
structured_llm_tool_selector = gpt4o_mini_llm.with_structured_output(ToolSelectors)

# 라우팅을 위한 프롬프트 템플릿
system = dedent("""You are an AI assistant specializing in routing user questions to the appropriate tools.
Use the following guidelines:
- For questions specifically about legal provisions or articles of the korean finanacial supervisory service policy and standards(은행업감독규정시행세칙), use the search_fss tool.
- For questions specifically about legal provisions or articles of the personal infomation protection law (개인정보보호법), use the search_personal tool.
- For questions specifically about legal provisions or articles of the banking law (은행법), use the search_banking tool.
 
Always choose all of the appropriate tools based on the user's question. 
""")

route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)

# 질문 라우터 정의
question_tool_router = route_prompt | structured_llm_tool_selector


# 질문 라우팅 노드 
def analyze_question_tool_search(state: SupervisorTeamState):
    question = state["question"]
    result = question_tool_router.invoke({"question": question})
    datasources = [tool.tool for tool in result.tools]
    return {"datasources": datasources}


def route_datasources_tool_search(state: SupervisorTeamState) -> List[str]:
    """질문 유형에 따라 적절한 도구를 선택합니다."""
    question_type = state.get("question_type")
    
    if question_type and question_type.is_casual:
        return ["llm_fallback"]
        
    datasources = set(state['datasources'])
    valid_sources = {"search_fss", "search_personal", "search_banking"} #"search_web",,  "llm_fallback"

    if datasources.issubset(valid_sources):
        return list(datasources)

    return list(valid_sources)