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
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.sqlite import SqliteSaver

class RetrievalAgentExecutor:
    def __init__(self, llm=CHAT_LLM) -> None:
        self._llm = llm
        self._text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        self._memory = SqliteSaver.from_conn_string(":memory:")
        self._config = {"configurable": {"thread_id": "default"}}

    def build(self, arxiv_url: str) -> None:
        # 1. Load
        loader = PyPDFLoader(arxiv_url)
        self._docs = loader.load()
        self._splits = self._text_splitter.split_documents(self._docs)
        
        # 2. Create agent
        vectorstore = FAISS.from_documents(self._splits, embedding=EMBEDDINGS)
        self._retriever = vectorstore.as_retriever(kwargs={"k": 5})
        retriever_tool = create_retriever_tool(self._retriever, "scientific_paper_retriever", "Searches and returns information from scientific papers")
        
        # added_system_message = "You are QA bot for scientific papers. Answer the following questions based on the summary and retrieved contexts.\nSummary: "
        # added_system_message += self._get_summary()
        # self._agent_executor = create_react_agent(self._llm, tools=[retriever_tool], checkpointer=self._memory, messages_modifier=added_system_message)
        self._agent_executor = create_react_agent(self._llm, tools=[retriever_tool], checkpointer=self._memory)

    def stream(self, prompt: str) -> dict:
        queries = []
        for step in self._agent_executor.stream({"messages": [HumanMessage(prompt)]}, config=self._config):
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
                    tool_calls = step["agent"]["messages"][0].additional_kwargs["tool_calls"]
                    queries = [eval(tool_call["function"]["arguments"])["query"] for tool_call in tool_calls]
            elif "tools" in step:
                contexts = [msg.content for msg in step["tools"]["messages"]]
        
        if not queries:
            contexts = []
        return dict(queries=queries, contexts=contexts, answer=answer, usage_metadata=usage_metadata)

    # def _get_summary(self):
    #     prompt_template = (
    #         "Write a concise summary of the following:\n"
    #         "{text}\n"
    #         "CONCISE SUMMARY:"
    #     )
    #     prompt = PromptTemplate.from_template(prompt_template)

    #     refine_template = (
    #         "Your job is to produce a final summary\n"
    #         "We have provided an existing summary up to a certain point: {existing_answer}\n"
    #         "We have the opportunity to refine the existing summary"
    #         "(only if needed) with some more context below.\n"
    #         "------------\n"
    #         "{text}\n"
    #         "------------\n"
    #         "Given the new context, refine the original summary in Italian"
    #         "If the context isn't useful, return the original summary."
    #     )
    #     refine_prompt = PromptTemplate.from_template(refine_template)
    #     chain = load_summarize_chain(
    #         llm=self._llm,
    #         chain_type="refine",
    #         question_prompt=prompt,
    #         refine_prompt=refine_prompt,
    #         return_intermediate_steps=True,
    #         input_key="input_documents",
    #         output_key="output_text",
    #     )
        
    #     result = chain.invoke(self._splits)["output_text"]
    #     return result
    
    # TODO: refine the summary
    # def _get_summary(self):
    #     map_prompt = hub.pull("rlm/map-prompt")
    #     map_chain = LLMChain(llm=CHAT_LLM, prompt=map_prompt)
        
    #     reduce_prompt = hub.pull("rlm/reduce-prompt")
    #     reduce_chain = LLMChain(llm=CHAT_LLM, prompt=reduce_prompt)
    #     combine_documents_chain = StuffDocumentsChain(
    #         llm_chain=reduce_chain, document_variable_name="doc_summaries"
    #     )

    #     # Combines and iteratively reduces the mapped documents
    #     reduce_documents_chain = ReduceDocumentsChain(
    #         # This is final chain that is called.
    #         combine_documents_chain=combine_documents_chain,
    #         # If documents exceed context for `StuffDocumentsChain`
    #         collapse_documents_chain=combine_documents_chain,
    #         # The maximum number of tokens to group documents into.
    #         token_max=4000,
    #     )

    #     # Combining documents by mapping a chain over them, then combining results
    #     map_reduce_chain = MapReduceDocumentsChain(
    #         # Map chain
    #         llm_chain=map_chain,
    #         # Reduce chain
    #         reduce_documents_chain=reduce_documents_chain,
    #         # The variable name in the llm_chain to put the documents in
    #         document_variable_name="docs",
    #         # Return the results of the map steps in the output
    #         return_intermediate_steps=False,
    #     )

    #     text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    #         chunk_size=1000, chunk_overlap=0
    #     )
    #     split_docs = text_splitter.split_documents(self._docs)
    #     result = map_reduce_chain.invoke(split_docs)
    #     summary = result["output_text"]
    #     print("summary:", summary)
    #     return summary