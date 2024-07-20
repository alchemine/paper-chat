"""Chain builder class"""

from langchain.chains.summarize import load_summarize_chain
from langchain.tools.retriever import create_retriever_tool
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_elasticsearch import ElasticsearchStore
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.sqlite import SqliteSaver

from paper_chat.core.depth_logging import D
from paper_chat.utils.utils import fetch_paper_info_from_url
from paper_chat.core.configs import CONFIGS_ES
from paper_chat.core.llm import get_llm, get_embeddings
from paper_chat.utils.elasticsearch_manager import ElasticSearchManager


SYSTEM_PROMPT = """[Role]
Act as if you are an instructor who writes summaries for lecture materials to explain complex scientific papers to beginners.

[Instruction]
To understand exactly what the author is arguing, you need to understand the key points of the text.
After understanding the key points, write a very detailed and informative summary.

[Format]
1. You should follow the **nine-sections**:
    - Information, Abstract, Introduction, Related Works, Methodology, Experiments, Discussion, Conclusion and Keywords.

2. Provide very detailed six-sentences summary in "Abstract" section with numbered lists.
    1. First sentence(Challenges): the challenges that the authors faced.
    2. Second sentence(Existing works): previous works to solve the challenges.
    3. Third sentence(Limitations): limitations of the existing or previous works.
    4. Fourth sentence(Motivation): motivation of the authors to propose a new method.
    5. Fifth sentence(Proposed method): detailed explanation of the proposed method to solve the challenges by the authors.
    6. Sixth sentence(Contributions): contributions of the proposed method by the authors.

3. Each section have multiple **key sentences** extracted from the paragraphs in the section with numbered lists.
    - Every paragraph in the given text consists of one **key sentence** and several supporting sentences.
    - Extract each **key sentence** from the paragraphs.
    - If the supporting sentences are necessary to understand the **key sentence**, include them.

[Constraints]
1. Use markdown formatting with itemized lists and numbered lists.
2. Use markdown underline or colorful font on one or two sentences to emphasize the main points.
3. The sentences should be concrete, not abstract or vague.
    - Do NOT summarize points with unimportant, unrelated or exceptional cases.
4. You must answer with Korean. Because the user is not familiar with english.

[Example]
### Information
1. Title: Neural message passing for quantum chemistry (양자화학을 위한 신경 메시지 전달)
2. Authors: Justin Gilmer, Samuel S. Schoenholz, Patrick F. Riley, Oriol Vinyals, George E. Dahl
3. arXiv: https://arxiv.org/pdf/1704.01212.pdf

### Abstract
1. Challenges: ...
2. Existing works: ...
3. Limitations: ...
4. Proposed method: ...
5. Contributions: ...

### 1. Introduction
1. <문단 1의 핵심 문장>
2. <문단 2의 핵심 문장>
3. <문단 3의 핵심 문장>
...

### 2. Related Works
1. <문단 1의 핵심 문장>
2. <문단 2의 핵심 문장>
3. <문단 3의 핵심 문장>
...

### 3. Methodology
1. <문단 1의 핵심 문장>
2. <문단 2의 핵심 문장>
3. <문단 3의 핵심 문장>
...

### 4. Experiments
1. <문단 1의 핵심 문장>
2. <문단 2의 핵심 문장>
3. <문단 3의 핵심 문장>
...

### 5. Discussion
1. <문단 1의 핵심 문장>
2. <문단 2의 핵심 문장>
3. <문단 3의 핵심 문장>
...

### 6. Conclusion
1. <문단 1의 핵심 문장>
2. <문단 2의 핵심 문장>
3. <문단 3의 핵심 문장>
...

### 7. Keywords
<논문과 밀접하게 관련이 있는 여러가지 주제들>
"""


class RetrievalAgentExecutor:
    def __init__(self, arxiv_id: str, openai_api_key: str) -> None:
        self.arxiv_id = arxiv_id
        self.llm = get_llm(openai_api_key)
        self.embeddings = get_embeddings(openai_api_key)
        self.arxiv_url = f"https://arxiv.org/pdf/{arxiv_id}"
        self.docs, self.splits = self.chunk(self.arxiv_url)

        self.es_manager = ElasticSearchManager()
        self.indices = {
            "papers_contents": CONFIGS_ES.papers_contents.index,
            "papers_metadata": CONFIGS_ES.papers_metadata.index,
        }

        # TODO: refactoring
        self.memory = SqliteSaver.from_conn_string(":memory:")
        self.config = {"configurable": {"thread_id": "default"}}
        self.paper_info = {}

    @D
    def build(self) -> None:
        # 1. Load retriever
        self.paper_info = self.fetch_paper_info(self.arxiv_id)
        vectorstore = self.insert_paper(self.paper_info)
        self.retriever = vectorstore.as_retriever(kwargs={"k": 5})
        retriever_tool = create_retriever_tool(
            self.retriever,
            "scientific_paper_retriever",
            "Searches and returns information from scientific papers",
        )

        # 2. Genearate agent executor
        # TODO: Is it best to include summary in system message?
        system_message = """[Role]
You are an helpful and informative assistant for question-answering tasks. 

[Instruction]
Answer the following questions based on the summary and retrieved contexts.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

[Constraints]
1. The sentences should be concrete, not abstract or vague.
2. If you don't know the answer, inform that you don't know and ask for more information.
3. Answer with Korean.

"""
        summary = f"""[Summary]
---
{self.get_summary()}
---
"""
        self._agent_executor = create_react_agent(
            self.llm,
            tools=[retriever_tool],
            checkpointer=self.memory,
            messages_modifier=system_message + summary,
        )

    @D
    def stream(self, prompt: str) -> dict:
        queries = []
        for step in self._agent_executor.stream(
            {"messages": [HumanMessage(prompt)]}, config=self.config
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

    @D
    def chunk(self, arxiv_url: str):
        docs = PyPDFLoader(arxiv_url).load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )
        splits = text_splitter.split_documents(docs)
        return docs, splits

    @D
    def get_summary(self, version: str = "stuff") -> str:
        if doc := self.search_with_arxiv_id(self.arxiv_id):
            summary = doc["summary"]
        else:
            # TODO: logging
            print(f"Summary of {self.arxiv_id} does not exist. Generating summary..")

            # If the document does not exist, summarize it.
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", SYSTEM_PROMPT),
                    ("human", "{text}"),
                ]
            )

            match version:
                case "stuff":
                    try:
                        # 30s
                        chain = load_summarize_chain(
                            self.llm, chain_type="stuff", prompt=prompt
                        )
                        summary = chain.run(self.docs)
                    except Exception as e:
                        raise Exception(e)

                        # TODO: fallback but not too slow
                        # 300s
                        # fallback: map_reduce
                        chain = load_summarize_chain(
                            self.llm, chain_type="map_reduce", combine_prompt=prompt
                        )
                        summary = chain.run(self.splits)

                # # NOTE: too slow
                # case "map_reduce":
                #     # 104s
                #     chain = load_summarize_chain(
                #         self.llm, chain_type="map_reduce", combine_prompt=prompt
                #     )
                #     summary = chain.run(self._splits)

                #             case "refine":
                #                 # 300s
                #                 refine_system_prompt = """
                # [Role]
                # Act as if you are an instructor who writes summaries to explain complex scientific papers to beginners.

                # [Instruction]
                # Your job is to produce a final summary.
                # We have provided an existing summary up to a certain point: {existing_answer}
                # We have the opportunity to refine the existing summary
                # (only if needed) with some more context below.
                # ------------
                # {text}
                # ------------
                # Given the new context, refine the original summary in Italian.
                # If the context isn't useful, return the original summary.

                # [Constraints]
                # 1. Use markdown formatting with itemized lists and numbered lists.
                # 2. Utilize the general structure of a scientific paper.
                # 3. Highlight what the authors most want to argue.
                # - Use bold text to emphasize the main points.
                # 4. The summary should be concise and easy to understand.
                # 5. Readability is most important.
                # 6. Answer with Korean.
                # 7. As the first section, provide "세 줄 요약"(three-line summary).
                # - Each line should be a very easy to understand sentence.
                # - The challenges that the authors faced and the solutions they proposed should be revealed clearly.

                # Take a deep breath and summarize step by step.
                # """
                #                 refine_prompt = ChatPromptTemplate.from_messages(
                #                     [
                #                         ("system", refine_system_prompt),
                #                         ("human", "{text}"),
                #                     ]
                #                 )
                #                 chain = load_summarize_chain(
                #                     llm=self.llm,
                #                     chain_type="refine",
                #                     question_prompt=prompt,
                #                     refine_prompt=refine_prompt,
                #                     return_intermediate_steps=True,
                #                     input_key="input_documents",
                #                     output_key="output_text",
                #                 )
                #                 summary = chain.invoke(self._splits)["output_text"]
                case _:
                    raise ValueError(f"Invalid version: {version}")

        return summary

    @D
    def insert_paper(self, paper_info: dict):
        # TODO: split function into smaller functions

        # 1. Check if the paper is in DB
        kwargs = {
            "embedding": self.embeddings,
            "es_connection": self.es_manager.es,
            "index_name": self.indices["papers_contents"],
        }
        if self.search_with_arxiv_id(paper_info["arxiv_id"]):
            vectorstore = ElasticsearchStore(**kwargs)
            return vectorstore

        # 2. Add metadata
        self.es_manager.insert_document(
            paper_info, index=self.indices["papers_metadata"]
        )

        # 3. Add content
        vectorstore = ElasticsearchStore.from_documents(self.splits, **kwargs)
        return vectorstore

    @D
    def fetch_paper_info(self, arxiv_id: str) -> dict:
        if doc := self.search_with_arxiv_id(arxiv_id):
            return doc

        paper_info = fetch_paper_info_from_url(self.arxiv_url)
        paper_info["summary"] = self.get_summary()

        # Check null values
        for key, value in paper_info.items():
            if isinstance(value, str) and (value.lower() in CONFIGS_ES.null_values):
                paper_info[key] = None
                print(f"{key}: {value} -> None")

        return paper_info

    def get_paper_info(self) -> dict:
        return self.paper_info

    def search_with_arxiv_id(self, arxiv_id: str) -> dict:
        response = self.es_manager.search(
            index=self.indices["papers_metadata"],
            body={"query": {"term": {"arxiv_id": arxiv_id}}},
        )
        hits = response["hits"]["hits"]
        if hits:
            srcs = [hit["_source"] for hit in hits]
            assert len(srcs) == 1, srcs
            result = srcs[0]
        else:  # empty list
            result = {}

        return result
