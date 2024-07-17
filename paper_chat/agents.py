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

[Format]
1. As the first section, provide very detailed "세 줄 요약"(three-sentences summary) with numbered lists.
    1. First sentence(Challenges): the challenges that the authors faced.
    2. Second sentence(Existing works and limitations): previous works to solve the challenges and their limitations.
    3. Third sentence(Proposed method and contributions): the solutions that the authors proposed and their contributions.

2. You should follow the general structure of a scientific paper.
    - Information, Abstract, Introduction, Related Works, Methodology, Experiments, Conclusion.

3. Each section should have multiple **key sentences** with numbered lists.
    - Each paragraph of the given text consists of one **key sentence** and several supporting sentences.
    - Extract each **key sentence** from the paragraphs and summarize entire texts utilizing them.
    - If the supporting sentences are necessary to understand the **key sentence**, include them.
        
[Constraints]
1. Use markdown formatting with itemized lists and numbered lists.
2. Emphasize what the authors most wanted to argue.
    - Do NOT summarize points with unimportant, unrelated or exceptional cases.
3. The sentences should be concrete, not abstract or vague.
4. Answer with Korean.

[Example]
### Information
1. 이름: Neural message passing for quantum chemistry
2. 저자: Justin Gilmer, Samuel S. Schoenholz, Patrick F. Riley, Oriol Vinyals, George E. Dahl
3. 링크: https://arxiv.org/pdf/1704.01212.pdf

### Abstract
1. Challenges: ...
2. Existing works and limitations: ...
3. Proposed method and contributions: ...

### 1. Introduction
1. 타 분야와 달리 machine learning을 통해 분자, 물질들의 물성을 예측하는 일은 여전히 어려움을 겪고 있음.
    - 대부분의 연구들은 feature engineering에 맴돌고 있고, NN을 사용하는 경우는 흔치 않다.
2. 적절한 inductive biases(유도 편향?)을 가진 모델(GNN)을 찾을 수 있다면 ML을 적용할 수 있을 것이다.
    - 최근 quantum chemistry calculation과 molecular dynamics simulations와 관련된 굉장히 많은 데이터가 생성있음.
    - 원자 시스템의 대칭성, 그래프의 동형성(isomorphism)에 불변한 GNN은 분자에도 잘 적용될 것이다.
3. 본 논문의 목표는, chemical prediction 문제에 사용할 수 있는 ML 모델을 설명하는 것이다.

### 2. Related Works
1. 이전 연구들은 transformer 레이어의 조합을 찾거나 매개변수 수를 줄이는 데 초점을 맞추었음.
2. Kim et al. (2024)는 전체 transformer 레이어를 제거하여 지연 시간을 줄일 수 있다는 것을 보여줌.
3. Bhojanapalli et al. (2021)은 MLP 레이어를 제거하는 것이 전체 transformer 레이어를 제거하는 것보다 성능에 덜 영향을 미침을 관찰함.

### 3. Proposed Methods
1. MPNNs의 forward pass는 2개의 phases를 가진다.
2. Family: Convolutional Networks for Learning Molecular Fingerprints
3. Family: Gated Graph Neural Networks (GG-NN)
4. Family: Interaction Networks 등에 대한 설명

### 4. Experiments
1. 중간 레이어가 공통 표현 공간을 공유하는지 확인하기 위해 transformer가 특정 레이어를 건너뛰거나 인접 레이어의 순서를 변경하는 것에 견고한지 테스트.
2. 중간 레이어의 가중치를 중앙 레이어의 가중치로 대체하여 중간 레이어를 건너뛰는 실험을 통해 중간 레이어가 서로 다른 기능을 수행하는 것을 확인.
3. 중간 레이어의 실행 순서가 얼마나 중요한지 테스트하고, 레이어를 역순으로 실행하거나 무작위 순서로 실행하여 결과를 평균화함.

### 5. Discussion
1. Transformer 레이어가 대부분의 변화에 견고한 이유에 대한 완전한 설명은 미래 연구에 남겨두었음.
2. 레이어의 존재가 실행 순서보다 중요한 경우, 간단한 방법으로 정확도를 지연 시간 획득으로 교환할 수 있음을 시사함.
3. 또한, 레이어를 실행하는 경로 메커니즘은 Switch Transformers와 유사하게 사용될 수 있음을 제안함.

### 6. Conclusion
1. 다양한 크기의 Pythia 모델에 대한 평균 유사성 행렬을 계산하여 모델 크기가 증가함에 따라 "유사한 임베딩 공유" 속성이 어떻게 변하는지 밝혔음.
2. Bhojanapalli et al. (2021)과 Kim et al. (2024)의 관찰 사이에는 세부 단위에서 차이가 있음을 발견함.
3. Denseformer Pagliardini et al. (2024)는 DWA를 적용한 후에도 모듈이 원래 transformer 모듈과 코사인 유사성을 가지고 있음을 발견함.
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

        # added_system_message = "You are QA bot for scientific papers. Answer the following questions based on the summary and retrieved contexts.\nSummary: "
        # added_system_message += self._summary
        # self._agent_executor = create_react_agent(
        #     self._llm,
        #     tools=[retriever_tool],
        #     checkpointer=self._memory,
        #     messages_modifier=added_system_message,
        # )
        self._agent_executor = create_react_agent(
            self._llm, tools=[retriever_tool], checkpointer=self._memory
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
        if not self._summary:
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
