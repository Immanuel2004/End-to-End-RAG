import streamlit as st
from pathlib import Path
import sys
import time

sys.path.append(str(Path(__file__).parent))

from src.config.config import Config
from src.data_ingestion.document_processor import DocumentProcessor
from src.vectorstore.vectorstore import VectorStore
from src.graph_builder.graph_builder import GraphBuilder

st.set_page_config(
    page_title="RAG Search",
    layout="wide"
)

st.markdown("""
    <style>
    .stButton > button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        font-size: 16px;
        padding: 0.6em 1em;
    }
    .stTextInput>div>div>input {
        font-size: 16px;
        padding: 0.6em;
    }
    .stExpanderHeader {
        font-weight: bold;
        font-size: 16px;
    }
    .stTextArea textarea {
        font-size: 14px;
    }
    </style>
""", unsafe_allow_html=True)

def init_session_state():
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = None
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False
    if 'history' not in st.session_state:
        st.session_state.history = []

@st.cache_resource
def initialize_rag():
    try:
        llm = Config.get_llm()
        if llm is None:
            st.error("LLM not initialized")
            return None, 0

        doc_processor = DocumentProcessor(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP
        )

        urls = Config.DEFAULT_URLS
        documents = doc_processor.process_url(urls)
        if not documents:
            st.error("No documents loaded")
            return None, 0

        vector_store = VectorStore()
        vector_store.create_vectorstore(documents)
        retriever = vector_store.get_retriever()
        if retriever is None:
            st.error("Retriever not initialized")
            return None, 0

        graph_builder = GraphBuilder(retriever=retriever, llm=llm)
        graph_builder.build()

        return graph_builder, len(documents)

    except Exception as e:
        st.error(f"Failed to initialize: {str(e)}")
        return None, 0

def main():
    init_session_state()

    st.markdown("<h1 style='text-align: center;'>RAG Document Search</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Ask questions about the loaded documents</p>", unsafe_allow_html=True)

    if not st.session_state.initialized:
        with st.spinner("Loading system..."):
            rag_system, num_chunks = initialize_rag()
            if rag_system:
                st.session_state.rag_system = rag_system
                st.session_state.initialized = True
                st.success(f"System ready! ({num_chunks} document chunks loaded)")

    st.markdown("---")

    col1, col2 = st.columns([3, 1])

    with col1:
        with st.form("search_form"):
            question = st.text_input(
                "Enter your question:",
                placeholder="What would you like to know?"
            )
            submit = st.form_submit_button("Search")


            if submit and question:
                if st.session_state.rag_system:
                    with st.spinner("Searching..."):
                        start_time = time.time()
                        result = st.session_state.rag_system.run(question)
                        elapsed_time = time.time() - start_time

                        st.session_state.history.append({
                            'question': question,
                            'answer': result['answer'],
                            'time': elapsed_time
                        })

                        st.markdown("### Answer")
                        st.success(result['answer'])

                        with st.expander("Source Documents"):
                            for i, doc in enumerate(result['retrieved_docs'], 1):
                                st.text_area(
                                    f"Document {i}",
                                    doc.page_content[:500] + "...",
                                    height=120,
                                    disabled=True
                                )

                        st.caption(f"Response time: {elapsed_time:.2f} seconds")
        with col2:
            st.markdown("### Recent Searches")
            for item in reversed(st.session_state.history[-3:]):
                st.markdown(f"**Q:** {item['question']}")
                st.markdown(f"**A:** {item['answer'][:200]}...")
                st.caption(f"Time: {item['time']:.2f}s")
                st.markdown("---")

if __name__ == "__main__":
    main()
