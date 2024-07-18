"""Chain builder class"""

from paper_chat.core.llm import CHAT_LLM, EMBEDDINGS

from langchain import hub
from langchain.chains import MapReduceDocumentsChain, ReduceDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.chains.summarize import load_summarize_chain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.tools.retriever import create_retriever_tool
from langchain_core.messages import HumanMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
)
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.sqlite import SqliteSaver

from paper_chat.core.timer import T


SYSTEM_PROMPT = """[Role]
Act as if you are an instructor who writes summaries for lecture materials to explain complex scientific papers to beginners.

[Instruction]
To understand exactly what the author is arguing, you need to understand the key points of the text.
After understand the key points, write a very detailed and informative summary.

[Constraints]
1. Use markdown formatting with itemized lists and numbered lists.
2. Emphasize what the authors most wanted to argue.
    - Do NOT summarize points with unimportant, unrelated or exceptional cases.
3. The sentences should be concrete, not abstract or vague.
4. Answer with Korean.

[Format]
1. You should follow the general structure (7 sections) of a scientific paper.
    - Information, Abstract, Introduction, Related Works, Methodology, Experiments, Conclusion.

2. Provide very detailed three-sentences summary in "Abstract" section with numbered lists.
    1. First sentence(Challenges): the challenges that the authors faced.
    2. Second sentence(Existing works and limitations): previous works to solve the challenges and their limitations.
    3. Third sentence(Proposed method and contributions): proposed method to solve the challenges and their contributions.

3. Each section have multiple **key sentences** extracted from the paragraphs in the section with numbered lists.
    - Every paragraph in the given text consists of one **key sentence** and several supporting sentences.
    - Extract each **key sentence** from the paragraphs.
    - If the supporting sentences are necessary to understand the **key sentence**, include them.

[Example]
### Information
1. Title: Neural message passing for quantum chemistry
2. Authors: Justin Gilmer, Samuel S. Schoenholz, Patrick F. Riley, Oriol Vinyals, George E. Dahl
3. arXiv: https://arxiv.org/pdf/1704.01212.pdf

### Abstract
1. Challenges: ...
2. Existing works and limitations: ...
3. Proposed method and contributions: ...

### 1. Introduction
1. <key sentence from a paragraph 1>
    - <supporting sentence 1 from a paragraph 1>
2. <key sentence from a paragraph 2>
    - <supporting sentence 1 from a paragraph 2>
    - <supporting sentence 2 from a paragraph 2>
3. <key sentence from a paragraph 3>
...

### 2. Related Works
1. <key sentence from a paragraph 1>
    - <supporting sentence 1 from a paragraph 1>
2. <key sentence from a paragraph 2>
    - <supporting sentence 1 from a paragraph 2>
    - <supporting sentence 2 from a paragraph 2>
3. <key sentence from a paragraph 3>
...

### 3. Proposed Methods
1. <key sentence from a paragraph 1>
    - <supporting sentence 1 from a paragraph 1>
2. <key sentence from a paragraph 2>
    - <supporting sentence 1 from a paragraph 2>
    - <supporting sentence 2 from a paragraph 2>
3. <key sentence from a paragraph 3>
...

### 4. Experiments
1. <key sentence from a paragraph 1>
    - <supporting sentence 1 from a paragraph 1>
2. <key sentence from a paragraph 2>
    - <supporting sentence 1 from a paragraph 2>
    - <supporting sentence 2 from a paragraph 2>
3. <key sentence from a paragraph 3>
...

### 5. Discussion
1. <key sentence from a paragraph 1>
    - <supporting sentence 1 from a paragraph 1>
2. <key sentence from a paragraph 2>
    - <supporting sentence 1 from a paragraph 2>
    - <supporting sentence 2 from a paragraph 2>
3. <key sentence from a paragraph 3>
...

### 6. Conclusion
1. <key sentence from a paragraph 1>
    - <supporting sentence 1 from a paragraph 1>
2. <key sentence from a paragraph 2>
    - <supporting sentence 1 from a paragraph 2>
    - <supporting sentence 2 from a paragraph 2>
3. <key sentence from a paragraph 3>
...
"""


class RetrievalAgentExecutor:
    def __init__(self, llm=CHAT_LLM) -> None:
        self._llm = llm
        self._text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )
        self._memory = SqliteSaver.from_conn_string(":memory:")
        self._config = {"configurable": {"thread_id": "default"}}
        self._summary = ""

    def build(self, arxiv_url: str) -> None:
        # 1. Load
        loader = PyPDFLoader(arxiv_url)
        self._docs = loader.load()
        self._splits = self._text_splitter.split_documents(self._docs)

        # 2. Create agent
        vectorstore = FAISS.from_documents(self._splits, embedding=EMBEDDINGS)
        self._retriever = vectorstore.as_retriever(kwargs={"k": 5})
        retriever_tool = create_retriever_tool(
            self._retriever,
            "scientific_paper_retriever",
            "Searches and returns information from scientific papers",
        )

        # TODO: Is it best to include summary in system message?
        added_system_message = """
[Role]
You are QA bot for scientific papers.

[Instruction]
Answer the following questions based on the summary and retrieved contexts.

[Constraints]
1. The sentences should be concrete, not abstract or vague.
2. If you don't know the answer, inform that you don't know and ask for more information.
3. Answer with Korean.

[Summary]"""
        added_system_message += "\n------\n"
        added_system_message += self._summary
        added_system_message += "\n------\n"
        self._agent_executor = create_react_agent(
            self._llm,
            tools=[retriever_tool],
            checkpointer=self._memory,
            messages_modifier=added_system_message,
        )

    def stream(self, prompt: str) -> dict:
        queries = []
        for step in self._agent_executor.stream(
            {"messages": [HumanMessage(prompt)]}, config=self._config
        ):
            if "agent" in step:
                # Print
                message = step["agent"]["messages"][-1]
                if isinstance(message, tuple):
                    print(message)
                else:
                    message.pretty_print()

                answer = step["agent"]["messages"][0].content
                if answer:
                    usage_metadata = step["agent"]["messages"][0].usage_metadata
                else:
                    tool_calls = step["agent"]["messages"][0].additional_kwargs[
                        "tool_calls"
                    ]
                    queries = [
                        eval(tool_call["function"]["arguments"])["query"]
                        for tool_call in tool_calls
                    ]
            elif "tools" in step:
                contexts = [msg.content for msg in step["tools"]["messages"]]

        if not queries:
            contexts = []
        return dict(
            queries=queries,
            contexts=contexts,
            answer=answer,
            usage_metadata=usage_metadata,
        )

    @T
    def get_summary(self, version: str = "stuff") -> str:
        if self._summary:
            return self._summary

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT),
                ("human", "{text}"),
            ]
        )

        match version:
            case "stuff":
                # 10s
                chain = load_summarize_chain(
                    self._llm, chain_type="stuff", prompt=prompt
                )
                summary = chain.run(self._docs)

            case "map_reduce":
                # 104s
                chain = load_summarize_chain(
                    self._llm, chain_type="map_reduce", combine_prompt=prompt
                )
                summary = chain.run(self._splits)

            case "refine":
                # 300s
                refine_system_prompt = """
[Role]
Act as if you are an instructor who writes summaries to explain complex scientific papers to beginners.

[Instruction]
Your job is to produce a final summary.
We have provided an existing summary up to a certain point: {existing_answer}
We have the opportunity to refine the existing summary
(only if needed) with some more context below.
------------
{text}
------------
Given the new context, refine the original summary in Italian.
If the context isn't useful, return the original summary.

[Constraints]
1. Use markdown formatting with itemized lists and numbered lists.
2. Utilize the general structure of a scientific paper.
3. Highlight what the authors most want to argue.
- Use bold text to emphasize the main points.
4. The summary should be concise and easy to understand.
5. Readability is most important.
6. Answer with Korean.
7. As the first section, provide "세 줄 요약"(three-line summary).
- Each line should be a very easy to understand sentence.
- The challenges that the authors faced and the solutions they proposed should be revealed clearly.

Take a deep breath and summarize step by step.
"""
                refine_prompt = ChatPromptTemplate.from_messages(
                    [
                        ("system", refine_system_prompt),
                        ("human", "{text}"),
                    ]
                )
                chain = load_summarize_chain(
                    llm=self._llm,
                    chain_type="refine",
                    question_prompt=prompt,
                    refine_prompt=refine_prompt,
                    return_intermediate_steps=True,
                    input_key="input_documents",
                    output_key="output_text",
                )
                summary = chain.invoke(self._splits)["output_text"]

        self._summary = summary
        return self._summary
