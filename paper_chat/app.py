"""Streamlit application

- Reference: https://github.com/streamlit/llm-examples
"""

import streamlit as st

from paper_chat.agents import RetrievalAgentExecutor


@st.cache_data
def get_summary(_agent_executor: RetrievalAgentExecutor):
    return _agent_executor.get_summary()


st.set_page_config(layout="wide")


with st.sidebar:
    # arxiv_id = st.text_input("arXiv ID", key="arxiv_id", value="1706.03762")
    arxiv_id = st.text_input("arXiv ID", key="arxiv_id", value="2407.09298")
    arxiv_url = f"https://arxiv.org/pdf/{arxiv_id}"
    "e.g. https://arxiv.org/pdf/1706.03762"
    if "agent_executor" not in st.session_state:
        agent_executor = RetrievalAgentExecutor()
        agent_executor.build(arxiv_url)
        st.session_state["agent_executor"] = agent_executor


st.title("💬 Chatbot")
st.caption("🚀 A Streamlit chatbot powered by OpenAI")


if "messages" not in st.session_state:
    if "agent_executor" in st.session_state:
        # 1. Show summary
        msg = "논문 요약"
        st.session_state["messages"] = [{"role": "user", "content": msg}]

        summary = get_summary(st.session_state["agent_executor"])
        st.session_state.messages.append({"role": "assistant", "content": summary})

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])


if prompt := st.chat_input():
    if not arxiv_id:
        st.info("Please input arXiv ID to continue.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    output = st.session_state["agent_executor"].stream(prompt)
    queries = output["queries"]
    answer = output["answer"]
    contexts = output["contexts"]
    msg = f"{answer}\n\nQueries: {queries}\n\nContexts: {contexts}"

    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.chat_message("assistant").write(msg)
