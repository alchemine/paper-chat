"""Streamlit application

- Reference: https://github.com/streamlit/llm-examples
"""

import re
import streamlit as st
from streamlit import session_state as STATE

from paper_chat.agents import RetrievalAgentExecutor
from paper_chat.core.configs import CONFIGS_LLM


def initialize_session():
    STATE.messages = []
    STATE.arxiv_id = ""


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
        try:
            if arxiv_id not in STATE:
                STATE[arxiv_id] = RetrievalAgentExecutor(arxiv_id, openai_api_key)
                STATE[arxiv_id].build()

            paper_info = STATE[arxiv_id].get_paper_info()

            msg = f"**논문 요약**\n\n- {paper_info['title']} ({paper_info['arxiv_id']})"
            STATE.messages.append({"role": "user", "content": msg})
            summary = STATE[arxiv_id].get_summary()

            # Update when successful
            STATE.arxiv_id = arxiv_id
        except Exception as e:
            summary = f"요약을 생성하는 중 오류가 발생했습니다. \n\n```{e}```"
        STATE.messages.append({"role": "assistant", "content": summary})


for msg in STATE.messages:
    st.chat_message(msg["role"]).write(msg["content"])


if prompt := st.chat_input():
    if not arxiv_id:
        st.info("arXiv ID를 입력해주세요.")
        st.stop()

    STATE.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    try:
        output = STATE[arxiv_id].stream(prompt)
        answer = output["answer"]

        queries = output["queries"]
        joined_queries = ", ".join(queries)

        contexts = output["contexts"]
        formatted_contexts = "\n\n".join([f"```{context}```" for context in contexts])
        msg = f"{answer}\n\n- Queries: {joined_queries} \n\n- Contexts:\n {formatted_contexts}"
    except Exception as e:
        msg = "답변을 생성하는 중 오류가 발생했습니다. \n\n```{e}```"

    STATE.messages.append({"role": "assistant", "content": msg})
    st.chat_message("assistant").write(msg)
