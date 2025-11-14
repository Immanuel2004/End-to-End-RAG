import sys
from src.exception.Execption import CustomException
from src.logger.logging import logging

from langgraph.graph import END,StateGraph
from src.state.rag_state import RAGState
from src.nodes.nodes import RAGNodes

class GraphBuilder:
    def __init__(self,retriever,llm):
        self.nodes = RAGNodes()
        self.graph = None

    def build(self):
        try:
            builder = StateGraph(RAGState)
            
            builder.add_node('retriever',self.nodes.retrieve_docs)
            builder.add_node('responder',self.nodes.generate_answer)

            builder.set_entry_point('retriever')

            builder.add_edge('retriever','responder')
            builder.add_edge('responder',END)

            self.graph = builder.compile()
            logging.info("Builded the Graph successfully")
            return self.graph
        except Exception as e:
            raise CustomException(e,sys)
        
    def run(self,question:str)->dict:
        try:
            if self.graph is None:
                self.build()
            
            initial_state = RAGState(question=question)
            logging.info("Running the Graph builder")
            return self.graph.invoke(initial_state)
        except Exception as e:
            raise CustomException(e,sys)