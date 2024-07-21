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
from paper_chat.utils.utils import (
    fetch_paper_info_from_url,
    strip_list_string,
    add_newlines,
)
from paper_chat.core.configs import CONFIGS_ES, CONFIGS_AGENT
from paper_chat.core.llm import get_llm, get_embeddings
from paper_chat.utils.elasticsearch_manager import ElasticSearchManager


SYSTEM_PROMPT = """[Role]
Act as if you are an instructor who writes summaries for lecture materials to explain complex scientific papers to beginners.

[Instruction]
To understand exactly what the author is arguing, you need to understand the key points of the text.
After understanding the key points, write a very detailed and informative summary.

[Format]
1. You should follow the general **seven-sections**:
    - Abstract, Introduction, Related Works, Methodology, Experiments, Discussion, Conclusion.
2. Provide very detailed six-sentences summary in "Abstract" section with numbered lists.
    - Each sentence should start with **bold name** and colon.
3. Each section has five or more **key sentences** extracted from the paragraphs in the section with numbered lists.
    - Every paragraph in the given text consists of one **key sentence** and several supporting sentences.
    - Extract each **key sentence** from the paragraphs.
    - If the supporting sentences are necessary to understand the **key sentence**, include them.

[Constraints]
1. Use markdown formatting with itemized lists and numbered lists.
2. Use markdown underline or colorful font on one or two sentences to emphasize the main points.
3. The sentences should be concrete, not abstract or vague.
    - Do NOT summarize points with unimportant, unrelated or exceptional cases.
4. You must answer with Korean. Because the user is not familiar with english.
    - But each section title should be in English and start with ### (three hashtags).

[Example]
### Abstract
1. **Challenges**: <the challenges that the authors faced>
2. **Existing works**: <previous works to solve the challenges>
3. **Limitations**: <limitations of the existing or previous works>
4. **Motivation**: <motivation of the authors to propose a new method>
5. **Proposed method**: <detailed explanation of the proposed method to solve the challenges by the authors>
6. **Contributions**: <contributions of the proposed method by the authors>

### Introduction
1. <문단 1의 핵심 문장>
2. <문단 2의 핵심 문장>
3. <문단 3의 핵심 문장>
4. <문단 4의 핵심 문장>
5. <문단 5의 핵심 문장>

### Related Works
1. <문단 1의 핵심 문장>
2. <문단 2의 핵심 문장>
3. <문단 3의 핵심 문장>
4. <문단 4의 핵심 문장>
5. <문단 5의 핵심 문장>

### Methodology
1. <문단 1의 핵심 문장>
2. <문단 2의 핵심 문장>
3. <문단 3의 핵심 문장>
4. <문단 4의 핵심 문장>
5. <문단 5의 핵심 문장>

### Experiments
1. <문단 1의 핵심 문장>
2. <문단 2의 핵심 문장>
3. <문단 3의 핵심 문장>
4. <문단 4의 핵심 문장>
5. <문단 5의 핵심 문장>

### Discussion
1. <문단 1의 핵심 문장>
2. <문단 2의 핵심 문장>
3. <문단 3의 핵심 문장>
4. <문단 4의 핵심 문장>
5. <문단 5의 핵심 문장>

### Conclusion
1. <문단 1의 핵심 문장>
2. <문단 2의 핵심 문장>
3. <문단 3의 핵심 문장>
4. <문단 4의 핵심 문장>
5. <문단 5의 핵심 문장>
"""


class RetrievalAgentExecutor:
    def __init__(self, arxiv_id: str, openai_api_key: str, reset: bool = False) -> None:
        self.arxiv_id = arxiv_id
        self.llm = get_llm(openai_api_key)
        self.embeddings = get_embeddings(openai_api_key)
        self.arxiv_url = f"https://arxiv.org/pdf/{arxiv_id}"
        self.docs, self.splits = self.chunk(self.arxiv_url)

        self.es_manager = ElasticSearchManager(reset)
        self.indices = {
            "papers_contents": CONFIGS_ES.papers_contents.index,
            "papers_metadata": CONFIGS_ES.papers_metadata.index,
        }

        # TODO: refactoring
        self.memory = SqliteSaver.from_conn_string(":memory:")
        self.config = {"configurable": {"thread_id": "default"}}
        self.paper_info = {}

    @D
    def load_paper_info(self, arxiv_id: str) -> dict:
        # 1. Load cache if exists
        if doc := self.search_with_arxiv_id(arxiv_id):
            return doc

        # 2. Load fundamental paper info from web
        paper_info = fetch_paper_info_from_url(arxiv_id=arxiv_id)
        return paper_info

    @D
    def append_summary(self, paper_info: dict) -> Exception | None:
        if "summary" in paper_info:
            return None

        # Generate summary and insert paper_info into DB
        summary_result = self.get_summary()
        paper_info["summary"] = summary_result["summary"]
        return summary_result["exception"]

    @D
    def build(self, information: str) -> None:
        retriever = self.get_retriever(self.indices["papers_contents"])
        retriever_tool = create_retriever_tool(
            retriever,
            "scientific_paper_retriever",
            "Searches and returns information from scientific papers",
        )

        # 3. Genearate agent executor
        # TODO: Is it best to include information in system message?
        system_message = """[Role]
You are an helpful and informative assistant for question-answering tasks. 

[Instruction]
Answer the following questions based on the information and retrieved contexts.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

[Constraints]
1. The sentences should be concrete, not abstract or vague.
2. If you don't know the answer, inform that you don't know and ask for more information.
3. Answer with Korean.

"""
        information = f"""[Information]
---
{information}
---
"""
        self._agent_executor = create_react_agent(
            self.llm,
            tools=[retriever_tool],
            checkpointer=self.memory,
            messages_modifier=system_message + information,
        )

    @D
    def get_summary(self) -> dict:
        try:
            # If the document does not exist, summarize it.
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", SYSTEM_PROMPT),
                    ("human", "{text}"),
                ]
            )
            chain = load_summarize_chain(self.llm, chain_type="stuff", prompt=prompt)
            summary = chain.run(self.docs)
            return {"summary": summary, "exception": None}
        except Exception as e:
            # If the summarization fails, return an empty summary.
            return {"summary": "", "exception": e}

        # match version:
        #     case "stuff":
        #         # 30s
        #         chain = load_summarize_chain(
        #             self.llm, chain_type="stuff", prompt=prompt
        #         )
        #         summary = chain.run(self.docs)
        #
        # NOTE: too slow
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
        # case _:
        #     raise ValueError(f"Invalid version: {version}")

    @D
    def insert_documents(self, paper_info: dict):
        # 1. Check null values for DB insertion
        for key, value in paper_info.items():
            if isinstance(value, str) and (value.lower() in CONFIGS_ES.null_values):
                paper_info[key] = None

        # 2. Add metadata
        self.es_manager.insert_document(
            paper_info, index=self.indices["papers_metadata"]
        )

        # 3. Add content
        kwargs = {
            "embedding": self.embeddings,
            "es_connection": self.es_manager.es,
            "index_name": self.indices["papers_contents"],
        }
        ElasticsearchStore.from_documents(self.splits, **kwargs)

    def get_retriever(self, index: str):
        kwargs = {
            "embedding": self.embeddings,
            "es_connection": self.es_manager.es,
            "index_name": index,
        }
        vectorstore = ElasticsearchStore(**kwargs)
        retriever = vectorstore.as_retriever(kwargs=CONFIGS_AGENT.retriever)
        return retriever

    def search_with_arxiv_id(self, arxiv_id: str) -> dict:
        response = self.es_manager.search(
            index=self.indices["papers_metadata"],
            body={"query": {"term": {"arxiv_id": arxiv_id}}},
        )
        hits = response["hits"]["hits"]
        if hits:
            srcs = [hit["_source"] for hit in hits]
            assert len(srcs) == 1, f"Multiple hits for {arxiv_id}: {srcs}"
            result = srcs[0]
        else:  # empty list
            result = {}

        return result

    def process_paper_info(self, paper_info: dict) -> str:
        authors_info = paper_info["authors"]
        authors_with_newline = []
        for name, affiliation in zip(
            authors_info["name"], authors_info["affiliations"]
        ):
            if len(affiliation) == 0:
                author = name
            else:
                author = f"{name} ({', '.join(affiliation)})"
            author_with_newline = f"\n\t- {author}"
            authors_with_newline.append(author_with_newline)

        fields_of_study = strip_list_string(paper_info["fieldsOfStudy"])

        # TODO: add keywords
        # keywords = self.generate_keywords(paper_info)

        # TODO: add citation(chicago style)
        # citation = self.generate_citation(paper_info)

        abstract_with_newlines = add_newlines(paper_info["abstract"], repl=".\n   ")
        information = f"""
### Information
- **Title**: {paper_info["title"]}
- **Citation count**: {paper_info["citationCount"] or "unknown"}
- **Publication**: *{paper_info["venue"]} ({paper_info["publicationDate"]}), {strip_list_string(paper_info["publicationTypes"])}*
- **Authors**: {"".join(authors_with_newline)}
- **Reference count**: {paper_info["referenceCount"] or "unknown"}
- **Fields of Study**: {fields_of_study}
- **arXiv url**: {paper_info["arxiv_url"]}
- **Abstract**:
    ```
    {abstract_with_newlines}
    ```
"""
        return information

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

        # Postprocess
        joined_queries = ", ".join(queries)
        formatted_contexts = "\n\n".join([f"```{context}```" for context in contexts])
        msg = f"{answer}\n\n- Queries: {joined_queries} \n\n- Contexts:\n {formatted_contexts}"

        return dict(
            queries=queries,
            contexts=contexts,
            answer=answer,
            usage_metadata=usage_metadata,
            msg=msg,
        )

    @D
    def chunk(self, arxiv_url: str):
        docs = PyPDFLoader(arxiv_url).load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )
        splits = text_splitter.split_documents(docs)
        return docs, splits
