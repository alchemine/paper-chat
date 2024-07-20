"""Streamlit application

- Reference: https://github.com/streamlit/llm-examples
"""

import re
import traceback
import streamlit as st
from streamlit import session_state as STATE

from paper_chat.agents import RetrievalAgentExecutor
from paper_chat.core.configs import CONFIGS_LLM


def initialize_session():
    STATE.messages = []
    STATE.arxiv_id = ""


def add_message(role: str, msg: str, error: bool = False):
    STATE.messages.append({"role": role, "content": msg, "error": error})


st.set_page_config(layout="wide")
st.title("💬 Paper-Chat")
st.caption("🚀 A Streamlit chatbot powered by OpenAI")

if "initialize_session" not in STATE:
    STATE.initialize_session = initialize_session()


with st.sidebar:
    openai_api_key = st.text_input(
        "OpenAI API Key", key="openai_api_key", placeholder="sk-proj-*****"
    )
    st.write(
        f"""
- Model name
    - ChatOpenAI: `{CONFIGS_LLM.chat_openai.model}`
    - Embeddings: `{CONFIGS_LLM.openai_embeddings.model}`
"""
    )

    example_id = "2004.07606"
    arxiv_id_input = st.text_input(
        "arXiv ID", key="arxiv_id_input", placeholder=example_id
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

    if openai_api_key == "":
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

    if arxiv_id != STATE.arxiv_id:
        if arxiv_id not in STATE:
            try:
                with st.spinner("논문 정보를 불러오고 요약하는 중.."):
                    STATE[arxiv_id] = RetrievalAgentExecutor(
                        arxiv_id, openai_api_key, reset=True
                    )
                    STATE[arxiv_id].build()

                paper_info = STATE[arxiv_id].get_paper_info()
                msg = "**논문 요약**"
                add_message("user", msg)
                add_message("assistant", paper_info["summary"])

                if e := STATE[arxiv_id].get_summary_exception():
                    print(traceback.format_exc())
                    msg = f"논문을 요약하는 도중 오류가 발생하였지만, 대화를 계속 진행할 수 있습니다. \n\n```{e}```"
                    add_message("assistant", msg, error=True)

                # Update when successful
                STATE.arxiv_id = arxiv_id
            except Exception as e:
                STATE.pop(arxiv_id, None)
                print(traceback.format_exc())
                msg = f"논문의 정보를 불러오는 도중 오류가 발생했습니다. 다른 논문을 준비해주세요. \n\n```{e}```"
                add_message("assistant", msg, error=True)


for msg in STATE.messages:
    # if msg["error"]:
    #     continue
    st.chat_message(msg["role"]).write(msg["content"])


if prompt := st.chat_input():
    if not arxiv_id:
        st.info("arXiv ID를 입력해주세요.")
        st.stop()

    add_message("user", prompt)
    st.chat_message("user").write(prompt)

    try:
        output = STATE[arxiv_id].stream(prompt)
        answer = output["answer"]

        queries = output["queries"]
        joined_queries = ", ".join(queries)

        contexts = output["contexts"]
        formatted_contexts = "\n\n".join([f"```{context}```" for context in contexts])
        msg = f"{answer}\n\n- Queries: {joined_queries} \n\n- Contexts:\n {formatted_contexts}"
        add_message("assistant", msg)
    except Exception as e:
        print(traceback.format_exc())
        msg = "답변을 생성하는 중 오류가 발생했습니다. \n\n```{e}```"
        add_message("assistant", msg, error=True)

    st.chat_message("assistant").write(msg)
