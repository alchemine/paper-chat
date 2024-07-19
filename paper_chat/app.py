"""Streamlit application

- Reference: https://github.com/streamlit/llm-examples
"""

import re
import streamlit as st
from streamlit import session_state as STATE

from paper_chat.agents import RetrievalAgentExecutor


# @st.cache_data
def get_cached_summary(arxiv_id):
    return STATE[arxiv_id].get_summary()


st.set_page_config(layout="wide")
st.title("💬 Chatbot")
st.caption("🚀 A Streamlit chatbot powered by OpenAI")

if "messages" not in STATE:
    STATE.messages = []
if "arxiv_id" not in STATE:
    STATE.arxiv_id = ""


with st.sidebar:
    # TODO: remove default value
    example_id = "2004.07606"
    arxiv_id_input = st.text_input(
        "arXiv ID", key="arxiv_id_input", placeholder=example_id
    )
    st.write(
        f"""
- https://arxiv.org/pdf/2004.07606
- https://arxiv.org/pdf/1706.03762
- https://arxiv.org/pdf/2201.05966
- https://arxiv.org/pdf/2305.02301
"""
    )

    if match := re.search(r"(\d{4}\.\d{5})", arxiv_id_input):
        arxiv_id = match.group(1)
    else:
        st.info(
            "적절하지 않은 arXiv ID입니다. arXiv ID는 2004.07606 와 같은 형식을 따라야 합니다."
        )
        st.stop()

    if arxiv_id != STATE.arxiv_id:
        STATE.arxiv_id = arxiv_id
        if arxiv_id not in STATE:
            STATE[arxiv_id] = RetrievalAgentExecutor(arxiv_id)
            STATE[arxiv_id].build()

        paper_info = STATE[arxiv_id].get_paper_info()

        msg = f"**논문 요약**\n\n- {paper_info['title']} ({paper_info['arxiv_id']})"
        STATE.messages.append({"role": "user", "content": msg})
        try:
            summary = STATE[arxiv_id].get_summary()
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

    output = STATE[arxiv_id].stream(prompt)
    answer = output["answer"]

    queries = output["queries"]
    joined_queries = ", ".join(queries)

    contexts = output["contexts"]
    formatted_contexts = "\n\n".join([f"```{context}```" for context in contexts])
    msg = f"{answer}\n\n- Queries: {joined_queries} \n\n- Contexts:\n {formatted_contexts}"

    STATE.messages.append({"role": "assistant", "content": msg})
    st.chat_message("assistant").write(msg)
