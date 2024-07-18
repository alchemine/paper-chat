"""Retrieval agent builder"""

from typing_extensions import TypedDict
from pprint import pprint

from langchain import hub
from langchain.tools.retriever import create_retriever_tool
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
)

from langgraph.prebuilt import create_react_agent
from langgraph.graph import END, StateGraph, START
from paper_chat.core.llm import CHAT_LLM, EMBEDDINGS


class RetrievalAgentBuilder:
    def __init__(self, memory, summary: str, docs: list) -> None:
        # Prepare constants
        self._memory = memory
        self._summary = summary
        self._docs = docs
        self._text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )
        self._splits = self._text_splitter.split_documents(self._docs)

        # 2. Create agent
        vectorstore = FAISS.from_documents(self._splits, embedding=EMBEDDINGS)
        self._retriever = vectorstore.as_retriever(kwargs={"k": 5})


class ReactRetrievalAgentBuilder(RetrievalAgentBuilder):
    def build(self):
        retriever_tool = create_retriever_tool(
            self._retriever,
            "scientific_paper_retriever",
            "Searches and returns information from scientific papers",
        )

        # TODO: Is it best to include summary in system message?
        system_message = """[Role]
You are an assistant for question-answering tasks. 

[Instruction]
Answer the following questions based on the summary and retrieved contexts.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

[Constraints]
1. The sentences should be concrete, not abstract or vague.
2. If you don't know the answer, inform that you don't know and ask for more information.
3. Answer with Korean.

"""
        summary = """[Summary]
---
{summary}
---
"""

        return create_react_agent(
            CHAT_LLM,
            tools=[retriever_tool],
            checkpointer=self._memory,
            messages_modifier=system_message + summary,
        )


# Data model
class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )


# Data model
class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""

    binary_score: str = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )


# Data model
class GradeAnswer(BaseModel):
    """Binary score to assess answer addresses question."""

    binary_score: str = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        documents: list of documents
    """

    question: str
    generation: str
    documents: list[str]


# Post-processing
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


class SelfRetrievalAgentBuilder(RetrievalAgentBuilder):
    """
    # NOTE: Too slow
    # NOTE: Not good for chitchat
    GraphRecursionError: Recursion limit of 25 reachedwithout hitting a stop condition. You can increase the limit by setting the `recursion_limit` config key.
    Traceback:
    File "/home/vscode/.cache/pypoetry/virtualenvs/paper-chat-9TtSrW0h-py3.11/lib/python3.11/site-packages/streamlit/runtime/scriptrunner/script_runner.py", line 589, in _run_script
        exec(code, module.__dict__)
    File "/app/paper_chat/app.py", line 53, in <module>
        output = st.session_state["agent_executor"].stream(prompt)
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    File "/app/paper_chat/agents/retrieval_agent_executor.py", line 179, in stream
        for step in self._agent_executor.stream(
    File "/home/vscode/.cache/pypoetry/virtualenvs/paper-chat-9TtSrW0h-py3.11/lib/python3.11/site-packages/langgraph/pregel/__init__.py", line 1170, in stream
        raise GraphRecursionError(
    """

    def build(self):
        # 1. Prepare nodes
        self._rag_chain = self._get_rag_chain()

        self._retrieval_grader = self._get_retrieval_grader()
        self._hallucination_grader = self._get_hallucination_grader()
        self._answer_grader = self._get_answer_grader()
        self._question_rewriter = self._get_question_rewriter()

        # 2. Capture the flow in as a graph
        workflow = StateGraph(GraphState)

        # Define the nodes
        workflow.add_node("retrieve", self.retrieve)  # retrieve
        workflow.add_node("grade_documents", self.grade_documents)  # grade documents
        workflow.add_node("generate", self.generate)  # generatae
        workflow.add_node("transform_query", self.transform_query)  # transform_query

        # Build graph
        workflow.add_edge(START, "retrieve")
        workflow.add_edge("retrieve", "grade_documents")
        workflow.add_conditional_edges(
            "grade_documents",
            self.decide_to_generate,
            {
                "transform_query": "transform_query",
                "generate": "generate",
            },
        )
        workflow.add_edge("transform_query", "retrieve")
        workflow.add_conditional_edges(
            "generate",
            self.grade_generation_v_documents_and_question,
            {
                "not supported": "generate",
                "useful": END,
                "not useful": "transform_query",
            },
        )

        # Compile
        app = workflow.compile()

        # # Run
        # inputs = {"question": "Explain how the different types of agent memory work?"}
        # for output in app.stream(inputs):
        #     for key, value in output.items():
        #         # Node
        #         pprint(f"Node '{key}':")
        #         # Optional: print full state at each node
        #         # pprint.pprint(value["keys"], indent=2, width=80, depth=None)
        #     pprint("\n---\n")

        # # Final generation
        # pprint(value["generation"])

        return app

    def _get_rag_chain(self):
        # Generate
        # prompt = hub.pull("rlm/rag-prompt")

        system_prompt = """[Role]
You are an assistant for question-answering tasks. 

[Instruction]
Answer the following questions based on the summary and retrieved contexts.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

[Constraints]
1. The sentences should be concrete, not abstract or vague.
2. If you don't know the answer, inform that you don't know and ask for more information.
3. Answer with Korean.

[Summary]
{summary}

[Context]
{context}
"""
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{question}"),
            ]
        ).partial(summary=self._summary)

        # Chain
        rag_chain = prompt | CHAT_LLM | StrOutputParser()
        # generation = rag_chain.invoke({"context": docs, "question": question})
        # print(generation)
        return rag_chain

    def _get_retrieval_grader(self):
        structured_llm_grader = CHAT_LLM.with_structured_output(GradeDocuments)

        # Prompt
        system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
            It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
            If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
            Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
        grade_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                (
                    "human",
                    "Retrieved document: \n\n {document} \n\n User question: {question}",
                ),
            ]
        )
        retrieval_grader = grade_prompt | structured_llm_grader
        # question = "agent memory"
        # docs = retriever.get_relevant_documents(question)
        # doc_txt = docs[1].page_content
        # print(retrieval_grader.invoke({"question": question, "document": doc_txt}))
        return retrieval_grader

    def _get_hallucination_grader(self):
        # Hallucination grader
        structured_llm_grader = CHAT_LLM.with_structured_output(GradeHallucinations)

        # Prompt
        system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n 
            Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""
        hallucination_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                (
                    "human",
                    "Set of facts: \n\n {documents} \n\n LLM generation: {generation}",
                ),
            ]
        )

        hallucination_grader = hallucination_prompt | structured_llm_grader
        # hallucination_grader.invoke({"documents": docs, "generation": generation})
        return hallucination_grader

    def _get_answer_grader(self):
        # Answer grader
        structured_llm_grader = CHAT_LLM.with_structured_output(GradeAnswer)

        # Prompt
        system = """You are a grader assessing whether an answer addresses / resolves a question \n 
            Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question."""
        answer_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                (
                    "human",
                    "User question: \n\n {question} \n\n LLM generation: {generation}",
                ),
            ]
        )

        answer_grader = answer_prompt | structured_llm_grader
        # answer_grader.invoke({"question": question, "generation": generation})
        return answer_grader

    def _get_question_rewriter(self):
        # Question Re-writer
        # Prompt
        system = """You a question re-writer that converts an input question to a better version that is optimized \n 
            for vectorstore retrieval. Look at the input and try to reason about the underlying semantic intent / meaning."""
        re_write_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                (
                    "human",
                    "Here is the initial question: \n\n {question} \n Formulate an improved question.",
                ),
            ]
        )
        question_rewriter = re_write_prompt | CHAT_LLM | StrOutputParser()
        # question_rewriter.invoke({"question": question})
        return question_rewriter

    # Nodes
    def retrieve(self, state):
        """
        Retrieve documents

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, documents, that contains retrieved documents
        """
        print("---RETRIEVE---")
        question = state["question"]

        # Retrieval
        documents = self._retriever.get_relevant_documents(question)
        return {"documents": documents, "question": question}

    def generate(self, state):
        """
        Generate answer

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, generation, that contains LLM generation
        """
        print("---GENERATE---")
        question = state["question"]
        documents = state["documents"]

        # RAG generation
        generation = self._rag_chain.invoke(
            {"context": documents, "question": question}
        )
        return {"documents": documents, "question": question, "generation": generation}

    def grade_documents(self, state):
        """
        Determines whether the retrieved documents are relevant to the question.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Updates documents key with only filtered relevant documents
        """

        print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
        question = state["question"]
        documents = state["documents"]

        # Score each doc
        filtered_docs = []
        for d in documents:
            score = self._retrieval_grader.invoke(
                {"question": question, "document": d.page_content}
            )
            grade = score.binary_score
            if grade == "yes":
                print("---GRADE: DOCUMENT RELEVANT---")
                filtered_docs.append(d)
            else:
                print("---GRADE: DOCUMENT NOT RELEVANT---")
                continue
        return {"documents": filtered_docs, "question": question}

    def transform_query(self, state):
        """
        Transform the query to produce a better question.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Updates question key with a re-phrased question
        """

        print("---TRANSFORM QUERY---")
        question = state["question"]
        documents = state["documents"]

        # Re-write question
        better_question = self._question_rewriter.invoke({"question": question})
        return {"documents": documents, "question": better_question}

    # Edges
    def decide_to_generate(self, state):
        """
        Determines whether to generate an answer, or re-generate a question.

        Args:
            state (dict): The current graph state

        Returns:
            str: Binary decision for next node to call
        """

        print("---ASSESS GRADED DOCUMENTS---")
        state["question"]
        filtered_documents = state["documents"]

        if not filtered_documents:
            # All documents have been filtered check_relevance
            # We will re-generate a new query
            print(
                "---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---"
            )
            return "transform_query"
        else:
            # We have relevant documents, so generate answer
            print("---DECISION: GENERATE---")
            return "generate"

    def grade_generation_v_documents_and_question(self, state):
        """
        Determines whether the generation is grounded in the document and answers question.

        Args:
            state (dict): The current graph state

        Returns:
            str: Decision for next node to call
        """

        print("---CHECK HALLUCINATIONS---")
        question = state["question"]
        documents = state["documents"]
        generation = state["generation"]

        score = self._hallucination_grader.invoke(
            {"documents": documents, "generation": generation}
        )
        grade = score.binary_score

        # Check hallucination
        if grade == "yes":
            print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
            # Check question-answering
            print("---GRADE GENERATION vs QUESTION---")
            score = self._answer_grader.invoke(
                {"question": question, "generation": generation}
            )
            grade = score.binary_score
            if grade == "yes":
                print("---DECISION: GENERATION ADDRESSES QUESTION---")
                return "useful"
            else:
                print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
                return "not useful"
        else:
            pprint("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
            return "not supported"
