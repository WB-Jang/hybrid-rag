import streamlit as st
import uuid
import time
from main.supervisor_team import legal_rag_agent # legal_rag_agent는 그대로 사용합니다.

# --- Streamlit UI 시작 ---

st.set_page_config(page_title="Risk AI Agent 챗봇", layout="centered")

st.title("⚖️ Risk AI Agent")
st.write("은행법, 은행업감독규정시행세칙, 개인정보보호법 관련 질문에 답변해 드립니다.")

# 세션 상태 초기화 (Thread ID 및 대화 기록)
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
    st.write(f"새로운 대화 세션이 시작되었습니다. Thread ID: **{st.session_state.thread_id}**")
else:
    st.write(f"현재 대화 세션. Thread ID: **{st.session_state.thread_id}**")


if "messages" not in st.session_state:
    st.session_state.messages = []

# 기존 대화 내용 표시
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 사용자 입력 처리
if prompt := st.chat_input("여기에 질문을 입력하세요."):
    # 사용자 메시지 기록 및 화면에 표시
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 챗봇 응답 생성
    with st.chat_message("assistant"):
        # response_placeholder는 스트리밍되는 내용을 실시간으로 업데이트하기 위한 공간입니다.
        response_placeholder = st.empty()
        
        # 에이전트 실행 과정을 단계별로 저장할 리스트
        agent_steps = []
        
        # --- 요청사항 1: 답변 생성 시간 측정 시작 ---
        start_time = time.time()

        config = {"configurable": {"thread_id": st.session_state.thread_id}}
        inputs = {"question": prompt}

        # --- 요청사항 2: .stream()을 사용하여 에이전트 실행 과정 스트리밍 ---
        # stream_mode='updates'는 각 단계(노드)가 실행될 때마다 변경 사항을 전달해줍니다.
        try:
            with st.spinner("에이전트가 생각 중입니다..."):
                for chunk in legal_rag_agent.stream(inputs, config=config, stream_mode="updates"):
                    # chunk는 {'node_name': node_output} 형태의 딕셔너리입니다.
                    for node_name, node_output in chunk.items():
                        # 실행된 노드 이름을 단계 리스트에 추가합니다.
                        step_message = f"✅ **{node_name}** 노드 실행 완료"
                        if step_message not in agent_steps:
                            agent_steps.append(step_message)
                        
                        # 현재까지의 진행 상황을 UI에 실시간으로 업데이트합니다.
                        response_placeholder.markdown("### 🕵️ 에이전트 실행 과정\n\n" + "\n\n".join(agent_steps))

            # 스트림이 모두 완료된 후, 최종 상태를 가져옵니다.
            current_state = legal_rag_agent.get_state(config)
            
            # --- 요청사항 1: 답변 생성 시간 측정 종료 ---
            end_time = time.time()
            duration = end_time - start_time

            # 최종 답변 및 중간 검색 결과 추출
            final_answer = current_state.values.get("final_answer", "최종 답변을 생성하지 못했습니다.")
            intermediate_answers = current_state.values.get('answers', [])
            
            answers_text = ""
            if intermediate_answers:
                for i, ans_text in enumerate(intermediate_answers):
                    answers_text += f"**📄 검색된 정보 {i+1}**\n"
                    answers_text += f"{ans_text}\n\n"
            else:
                answers_text = "검색된 관련 정보가 없습니다."

            # 최종 응답 메시지 구성
            full_response = f"""### 💡 최종 답변
{final_answer}

---

### 📚 검색 과정 요약
{answers_text}

---
*답변 생성에 걸린 시간: {duration:.2f}초*
"""
            
            # 최종 응답을 placeholder에 표시합니다.
            response_placeholder.markdown(full_response)
            
            # 챗봇 응답을 전체 대화 기록에 추가합니다.
            st.session_state.messages.append({"role": "assistant", "content": full_response})

        except Exception as e:
            st.error(f"오류가 발생했습니다: {str(e)}")
            full_response = "죄송합니다. 응답을 생성하는 동안 오류가 발생했습니다. 다시 시도해 주세요."
            response_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})


# --- Streamlit UI 끝 ---
