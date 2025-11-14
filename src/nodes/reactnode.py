import sys 
from src.exception.Execption import CustomException
from src.logger.logging import logging
from src.state.rag_state import RAGState

from typing import Optional,List
from langchain_core.documents import Document
from langchain_core.tools import Tool
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun

class RAGNodes:
    def __init__(self,retriever,llm):
        self.retriever = retriever
        self.llm = llm
        self._agent = None
    
    def retrieve_docs(self, state: RAGState) -> RAGState:
        try:
            docs = self.retriever.get_relevant_documents(state.question)
            logging.info("Retrieving docs")
            return RAGState(
                question=state.question,
                retrieved_docs=docs
            )
        except Exception as e:
            raise CustomException(e, sys)

    
    def _build_tools(self)->List[Tool]:
        def retriever_tool_fn(query:str)->str:
            docs:List[Document] = self.retriever.invoke(query)
            if not docs:
                return "No documents found"
            merged = []
            for i , d in enumerate(docs[:8],start=1):
                meta = d.metadata if hasattr(d,"metadata") else {}
                title = meta.get('title') or meta.get('source') or f"doc_{i}"
                merged.append(f"[{i}] {title} \n{d.page_content}")
            return "\n\n".join(merged)
        
        retriever_tool = Tool(
            name="retriever",
            description="Fetch passages from indexed vectorstore",
            func=retriever_tool_fn
        )
        wiki = WikipediaQueryRun(
            api_wrapper=WikipediaAPIWrapper(top_k_results=2,lang='en')
        )
        wikipedia_tool = Tool(
            name="wikipedia",
            description="search wikipedia for general knowledge",
            func=wiki.run
        )
        return [retriever_tool,wikipedia_tool]
    
    def _build_agent(self):
        tools = self._build_tools()
        system_prompt = (
            "You are a helpful RAG agent."
            "Prefer 'retriever' for user-provided docs: use 'wikipedia' for general knowledge."
            "Return only the final useful answer."
        )
        self._agent = create_react_agent(self.llm,tools=tools,prompt=system_prompt)

    def generate_answer(self,state:RAGState)->RAGState:
        if self._agent is None:
            self._build_agent()

        result = self._agent.invoke({'messages':[HumanMessage(content=state.question)]})
        messages = result.get("messages",[])
        answer :Optional[str] = None
        if messages:
            answer_msg = messages[-1]
            answer = getattr(answer_msg,'content',None)

        return RAGState(
            question=state.question,
            retrieved_docs=state.retrieved_docs,
            answer = answer or "Could not generate answer"
        )