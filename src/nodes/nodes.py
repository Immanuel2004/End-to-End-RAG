import sys
from src.exception.Execption import CustomException
from src.logger.logging import logging
from src.state.rag_state import RAGState

class RAGNodes:
    def __init__(self,retriever,llm):
        self.retriever = retriever
        self.llm = llm

    def retrieve_docs(self,state:RAGState)->RAGState:
        try:
            docs = self.retriever.invoke(state.question)
            logging.info("Retrieving docs")
            return RAGState(
                question=state.question,
                retrieved_docs=docs
            )
        except Exception as e:
            raise CustomException(e,sys)
        
    def generate_answer(self,state:RAGState)->RAGState:
        try:
            context = "\n\n".join([doc.page_content for doc in state.retrieved_docs])
            prompt = f"""Answer the questions based on context
                     context : {context}
                     question : {state.question}
            """
            response = self.invoke(prompt)
            logging.info("Generating Answers based on the context")
            return RAGState(
                question=state.question,
                retrieved_docs=state.retrieved_docs,
                answer=response.content
            )
        except Exception as e:
            raise CustomException(e,sys)