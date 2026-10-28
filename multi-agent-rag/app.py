
import os
import tempfile

import streamlit as st
from supervisor_graph import build_vector_store, AgentState, app as supervisor_app


# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="ResearchAI | Multi-Agent RAG",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------
# 2. CUSTOM STYLING
# --------------------------------------------------

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    .stApp {
        background: #0b0d12;
        color: #e8eaf0;
        font-family: 'Inter', sans-serif;
    }

    [data-testid="stSidebar"] {
        background: #10131b;
        border-right: 1px solid #242936;
    }

    .block-container {
        max-width: 1150px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .hero {
        padding: 1.3rem 0;
    }

    .eyebrow {
        color: #a78bfa;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 2px;
        text-transform: uppercase;
    }

    .hero h1 {
        color: #f4f5fb;
        font-size: 2.6rem;
        font-weight: 700;
        letter-spacing: -1.5px;
        margin: 0.5rem 0;
    }

    .hero p {
        color: #969db0;
        font-size: 1rem;
    }

    .panel {
        background: #121620;
        border: 1px solid #272d3b;
        border-radius: 16px;
        padding: 1.2rem;
        min-height: 115px;
    }

    .panel-label {
        color: #929bb0;
        font-size: 0.85rem;
    }

    .panel-value {
        color: #f2f3f9;
        font-size: 1.3rem;
        font-weight: 600;
        margin-top: 0.5rem;
    }

    .agent {
        background: #141925;
        border: 1px solid #292f40;
        border-radius: 12px;
        padding: 1rem;
        margin: 0.6rem 0;
    }

    .agent-name {
        color: #e8eaf5;
        font-weight: 600;
    }

    .agent-description {
        color: #939bb0;
        font-size: 0.85rem;
        margin-top: 0.3rem;
    }

    div[data-testid="stChatMessage"] {
        background: #121620;
        border: 1px solid #252b39;
        border-radius: 14px;
        padding: 1rem;
    }

    .stButton button {
        border-radius: 10px;
        min-height: 2.5rem;
    }

    hr {
        border-color: #272d3b;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# 3. SESSION STATE
# --------------------------------------------------

defaults = {
    "pdf_path": None,
    "vector_store": None,
    "processed_pdf_name": None,
    "messages": [],
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# --------------------------------------------------
# 4. SIDEBAR AND PDF UPLOAD
# --------------------------------------------------

with st.sidebar:
    st.markdown("## ✦ ResearchAI")
    st.caption("MULTI-AGENT DOCUMENT INTELLIGENCE")
    st.divider()

    st.markdown("### Your workspace")
    st.markdown("📄 Document Library")
    st.markdown("💬 Ask your documents")
    st.markdown("🧠 Agent activity")

    st.divider()
    st.markdown("### Knowledge source")

    uploaded_pdf = st.file_uploader(
        "Upload a PDF",
        type=["pdf"],
        key="pdf_uploader",
    )

    if uploaded_pdf is not None:
        st.caption(f"Selected: {uploaded_pdf.name}")

        if st.button("Process document", use_container_width=True):
            try:
                temp_dir = os.path.join(
                    tempfile.gettempdir(),
                    "multi_agent_rag",
                )
                os.makedirs(temp_dir, exist_ok=True)

                pdf_path = os.path.join(
                    temp_dir,
                    uploaded_pdf.name,
                )

                with open(pdf_path, "wb") as pdf_file:
                    pdf_file.write(uploaded_pdf.getvalue())

                with st.spinner(
                    "Building the local PDF search index..."
                ):
                    vector_store = build_vector_store(pdf_path)

                st.session_state["pdf_path"] = pdf_path
                st.session_state["vector_store"] = vector_store
                st.session_state["processed_pdf_name"] = uploaded_pdf.name
                st.session_state["messages"] = []

                st.success("Document processed successfully!")

            except Exception as exc:
                st.error(f"Could not process the PDF: {exc}")

    if st.session_state["vector_store"] is not None:
        st.success(
            f"Ready: {st.session_state['processed_pdf_name']}"
        )
    else:
        st.info("Upload and process a PDF to begin.")

    st.divider()
    st.caption("LOCAL MODE · OLLAMA")
    st.caption("Inference and embeddings run locally.")


# --------------------------------------------------
# 5. DASHBOARD HEADER
# --------------------------------------------------

st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">Your personal research workspace</div>
        <h1>Understand your documents.</h1>
        <p>
            Ask questions, explore chapters, and see your AI agents
            work together.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# 6. DASHBOARD CARDS
# --------------------------------------------------

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        """
        <div class="panel">
            <div class="panel-label">AI architecture</div>
            <div class="panel-value">Multi-agent RAG</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        """
        <div class="panel">
            <div class="panel-label">Language model</div>
            <div class="panel-value">Llama 3.2 · 3B</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col3:
    st.markdown(
        """
        <div class="panel">
            <div class="panel-label">Vector search</div>
            <div class="panel-value">Local FAISS</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------
# 7. AGENT CARDS
# --------------------------------------------------

st.markdown("### ✦ Meet your agents")

agent_col1, agent_col2 = st.columns(2)

with agent_col1:
    st.markdown(
        """
        <div class="agent">
            <div class="agent-name">◈ Supervisor Agent</div>
            <div class="agent-description">
                Routes questions to the appropriate specialist.
            </div>
        </div>
        <div class="agent">
            <div class="agent-name">⌕ Chapter Expert</div>
            <div class="agent-description">
                Retrieves relevant passages to answer chapter questions.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with agent_col2:
    st.markdown(
        """
        <div class="agent">
            <div class="agent-name">▤ Metadata Agent</div>
            <div class="agent-description">
                Finds document details, including its author.
            </div>
        </div>
        <div class="agent">
            <div class="agent-name">✧ Topic Expert</div>
            <div class="agent-description">
                Answers questions about topics covered in the document.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------
# 8. DOCUMENT CHAT
# --------------------------------------------------

st.divider()
st.markdown("### 💬 Ask your documents")

if st.session_state["vector_store"] is None:
    st.info(
        "Upload a PDF in the sidebar and click "
        "**Process document** to start chatting."
    )

else:
    st.caption(
        f"Currently querying: {st.session_state['processed_pdf_name']}"
    )

    # Display previous conversation messages.
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input(
        "Ask a question about your document..."
    )

    if question:
        st.session_state["messages"].append(
            {"role": "user", "content": question}
        )

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            try:
                with st.spinner(
                    "Supervisor is coordinating your AI agents..."
                ):
                    initial_state: AgentState = {
                        "pdf_path": st.session_state["pdf_path"],
                        "question": question,
                        "route": "",
                        "answer": "",
                        "vector_store": st.session_state["vector_store"],
                        "handoff_count": 0,
                        "chapter_answer": "",
                        "metadata_answer": "",
                    }

                    result = supervisor_app.invoke(initial_state)
                    answer = result.get(
                        "answer",
                        "The workflow returned no answer.",
                    )

                st.markdown(answer)

                # Show the route selected by the supervisor.
                route = result.get("route", "")
                if route:
                    st.caption(f"Workflow route: {route}")

                st.session_state["messages"].append(
                    {"role": "assistant", "content": answer}
                )

            except Exception as exc:
                st.error(f"Could not process the question: {exc}")
                st.caption(
                    "Check that Ollama is running and the required "
                    "local models are installed."
                )

