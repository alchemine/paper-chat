"""Streamlit application

- Reference: https://github.com/streamlit/llm-examples
"""

import re
from os import environ as env
import traceback
import streamlit as st
from streamlit import session_state as STATE

from paper_chat.agents import RetrievalAgentExecutor
from paper_chat.core.configs import CONFIGS_LLM


def initialize_session():
    STATE.messages = []
    STATE.session_id = ()
    STATE.use_summary = False
    STATE.no_summary = False


def add_message(role: str, msg: str, error: bool = False):
    STATE.messages.append({"role": role, "content": msg, "error": error})


def add_and_write_message(role: str, msg: str, error: bool = False):
    STATE.messages.append({"role": role, "content": msg, "error": error})
    st.chat_message(role).write(msg)


def write_messages():
    for msg in STATE.messages:
        st.chat_message(msg["role"]).write(msg["content"])


def add_and_write_information(arxiv_id: str):
    paper_info = STATE[arxiv_id].load_paper_info(arxiv_id)

    msg = "**논문 정보**"
    add_and_write_message("user", msg)
    add_and_write_message("assistant", paper_info["information"])


def add_and_write_summary(paper_info: str, summary_exception: None | Exception):
    if summary_exception:
        print(traceback.format_exc())
        msg = f"논문을 요약하는 도중 오류가 발생하였지만, 대화를 계속 진행할 수 있습니다. \n\n```{summary_exception}```"
        add_and_write_message("assistant", msg, error=True)
    else:
        msg = "**논문 요약**"
        add_and_write_message("user", msg)
        add_and_write_message("assistant", paper_info["summary"])


st.set_page_config(layout="wide")
st.title("💬 Paper-Chat")
st.caption("🚀 A Streamlit chatbot powered by OpenAI")


if "initialize_session" not in STATE:
    STATE.initialize_session = initialize_session()


with st.sidebar:
    openai_api_key = st.text_input(
        "**OpenAI API Key**",
        key="openai_api_key",
        placeholder="sk-proj-*****",
        value=env.get("OPENAI_API_KEY", ""),
    )
    st.write(
        f"""
- Model name
    - ChatOpenAI: `{CONFIGS_LLM.chat_openai.model}`
    - Embeddings: `{CONFIGS_LLM.openai_embeddings.model}`
"""
    )

    st.write("**문서 요약 옵션**")
    col1, col2 = st.columns(2)
    with col1:
        use_summary = st.checkbox(
            "문서 요약 O", value=STATE.use_summary, key="use_summary"
        )
    with col2:
        no_summary = st.checkbox(
            "문서 요약 X", value=STATE.no_summary, key="no_summary"
        )

    example_id = "2004.07606"
    arxiv_id_input = st.text_input(
        "**arXiv ID**", key="arxiv_id_input", placeholder=example_id
    )

    st.write(
        f"""
e.g.
- https://arxiv.org/pdf/2004.07606
- https://arxiv.org/pdf/1706.03762
- https://arxiv.org/pdf/2201.05966
- https://arxiv.org/pdf/2305.02301
"""
    )
    session_id = (openai_api_key, (use_summary, no_summary), arxiv_id_input)

    # Check session
    if openai_api_key == "":
        st.stop()

    if not use_summary and not no_summary:
        st.stop()
    elif not (use_summary ^ no_summary):
        st.info("문서 요약 옵션을 한 가지만 선택해주세요.")
        st.stop()

    if arxiv_id_input == "":
        st.stop()
    elif match := re.search(r"(\d{4}\.\d{5})", arxiv_id_input):
        # arxiv_id: https://arxiv.org/pdf/2004.07606 -> 2004.07606
        arxiv_id = match.group(1)
    else:
        st.info(
            "적절하지 않은 arXiv ID입니다. arXiv ID는 2004.07606 와 같은 형식을 따라야 합니다."
        )
        st.stop()


if session_id != STATE.session_id:
    write_messages()

    # Load session
    if arxiv_id not in STATE:
        try:
            with st.spinner("LLM을 불러오고 데이터베이스에 접속하는 중.."):
                STATE[arxiv_id] = RetrievalAgentExecutor(arxiv_id, openai_api_key)
        except Exception as e:
            print(traceback.format_exc())
            msg = f"LLM을 불러오고 데이터베이스에 접속하는 도중 오류가 발생했습니다. 프로그램이 정상적으로 동작하고 있는지 확인해주세요. \n\n```{e}```"
            st.chat_message("assistant").write(msg)
            st.stop()

    try:
        with st.spinner("논문 정보를 불러오는 중.."):
            paper_info = STATE[arxiv_id].load_paper_info(arxiv_id)
            add_and_write_information(arxiv_id)

        if use_summary:
            with st.spinner("논문을 요약하는 중.."):
                summary_exception = STATE[arxiv_id].append_summary(paper_info)
                STATE[arxiv_id].insert_document(paper_info)
                add_and_write_summary(paper_info, summary_exception)

        with st.spinner("챗봇 모델을 생성하는 중.."):
            STATE[arxiv_id].build(paper_info["information"])
            msg = "**대화를 시작할 수 있습니다! 어떤 것이 궁금하신가요?**"
            add_and_write_message("assistant", msg)

            # Update when successful
            STATE.session_id = session_id
    except Exception as e:
        # STATE.pop(arxiv_id, None)
        STATE.pop(session_id)
        print(traceback.format_exc())
        msg = f"논문의 정보를 불러오는 도중 오류가 발생했습니다. 다른 논문을 준비해주세요. \n\n```{e}```"
        st.chat_message("assistant").write(msg)
        st.stop()


if prompt := st.chat_input():
    write_messages()
    add_and_write_message("user", prompt)

    try:
        output = STATE[arxiv_id].stream(prompt)
        add_and_write_message("assistant", output["msg"])
    except Exception as e:
        print(traceback.format_exc())
        msg = "답변을 생성하는 중 오류가 발생했습니다. \n\n```{e}```"
        add_and_write_message("assistant", msg, error=True)
