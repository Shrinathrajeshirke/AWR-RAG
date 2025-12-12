import streamlit as st
import os
import uuid
from ingestor.rag_system_manager import RAGSystemManager
from ingestor.document_loader import DocumentProcessor
from config import MODEL_CHOICES
from utils.email_utils import generate_report_content

# --- configuration ---
TEMP_DIR = "temp_files"
if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR)

# --- Session state initialization ---
if 'rag_manager' not in st.session_state:
    st.session_state.rag_manager = None

if 'ingested_docs' not in st.session_state:
    st.session_state.ingested_docs = {}

if 'evaluation_history' not in st.session_state:
    st.session_state.evaluation_history = []

def initialize_rag_manager():
    """Initialize RAG manager without optional LangSmith credentials"""
    try:
        # Initialized without LangSmith parameters
        st.session_state.rag_manager = RAGSystemManager()
        return True
    except Exception as e:
        st.error(f"Failed to initialize RAGSystemManager: {e}")
        return False

def handle_ingestion(uploaded_files):
    """saves uploaded files and calls the RAGManager's indexing function."""
    doc_processor = DocumentProcessor()

    for uploaded_file in uploaded_files:
        doc_id = str(uuid.uuid4())[:8]
        file_path = os.path.join(TEMP_DIR, uploaded_file.name)
        file_content_bytes = uploaded_file.read()
        with open(file_path, "wb") as f:
            f.write(file_content_bytes)
        uploaded_file.seek(0)

        st.info(f"Processing {uploaded_file.name}...")
        try:
            chunks = doc_processor.load_and_split_document(
                file_path=file_path, 
                doc_id=doc_id, 
                filename=uploaded_file.name
            )
            st.info(f"Split into {len(chunks)} chunks. Indexing...")
            
            st.session_state.rag_manager.index_document_chunks(chunks)
            st.session_state.ingested_docs[doc_id] = uploaded_file.name
            st.success(f"✓ Indexed {uploaded_file.name} (ID: {doc_id})")
        except Exception as e:
            st.error(f"Failed to process {uploaded_file.name}: {e}")
            import traceback
            st.code(traceback.format_exc())
        finally:
            if os.path.exists(file_path):
                os.remove(file_path)

    st.success(f"Successfully ingested {len(uploaded_files)} document(s).")
    st.rerun()

def handle_query(query, selected_doc_ids, api_choice, api_key, model_name, 
                enable_eval, ground_truth, openai_eval_key, 
                recipient_email, report_format):
    """Executes the RAG query with optional evaluation and sends report via email."""
    if not selected_doc_ids:
        st.warning("Please select at least one document to query.")
        return
    
    result = st.session_state.rag_manager.answer_query(
        question=query,
        doc_ids=selected_doc_ids,
        api_choice=api_choice,
        api_key=api_key,
        model_name=model_name,
        enable_evaluation=enable_eval,
        ground_truth=ground_truth if ground_truth else None,
        openai_eval_key=openai_eval_key
    )

    st.markdown("---")
    st.markdown("### 🤖 RAG Answer")
    
    if result.get("error"):
        st.error(f"Query Failed: {result['answer']}")
        return

    final_answer = result["answer"]
    retrieved_contexts = result.get("contexts", [])
    
    st.success("Query successful!")
    st.markdown(f"**Answer:**\n{final_answer}")

    # Show retrieved context count
    st.caption(f"📚 Retrieved {result.get('retrieved_docs_count', 0)} relevant chunks")
    
    # Show evaluation scores if available
    if enable_eval and "evaluation" in result and not result.get("error"):
        st.success("Evaluation completed and recorded internally for tracking.")
        with st.expander("📊 View Evaluation Scores"):
            st.json(result["evaluation"])
    
    # Generate and send report if email is provided
    if recipient_email and recipient_email.strip():
        with st.spinner(f"Generating {report_format.upper()} report and sending email..."):
            try:
                # Import email utilities
                from utils.email_utils import generate_report_content, generate_pdf_report, send_email_report
                
                # Generate text and HTML content
                text_content, html_content = generate_report_content(
                    user_query=query,
                    answer_markdown=final_answer,
                    context=retrieved_contexts
                )
                
                subject = f"RAG Query Report - {query[:50]}..."
                
                if report_format == "text":
                    # Send plain text email
                    success = send_email_report(
                        to_email=recipient_email,
                        subject=subject,
                        body=text_content,
                        subtype='plain'
                    )
                    
                elif report_format == "html":
                    # Send HTML email
                    success = send_email_report(
                        to_email=recipient_email,
                        subject=subject,
                        body=html_content,
                        subtype='html'
                    )
                    
                elif report_format == "pdf":
                    # Generate PDF and send as attachment
                    pdf_path = os.path.join(TEMP_DIR, f"rag_report_{uuid.uuid4().hex[:8]}.pdf")
                    generate_pdf_report(
                        user_query=query,
                        answer_markdown=final_answer,
                        context=retrieved_contexts,
                        output_path=pdf_path
                    )
                    
                    success = send_email_report(
                        to_email=recipient_email,
                        subject=subject,
                        body="Please find the RAG query report attached as PDF.",
                        subtype='plain',
                        attachment_path=pdf_path
                    )
                    
                    # Clean up PDF file
                    if os.path.exists(pdf_path):
                        os.remove(pdf_path)
                
                if success:
                    st.success(f"Report sent successfully to {recipient_email} as {report_format.upper()}!")
                else:
                    st.warning(f"Report generated but email sending failed. Please check SMTP configuration in email_utils.py")
                    
            except Exception as e:
                st.error(f"Error generating/sending report: {e}")
                import traceback
                st.code(traceback.format_exc())
    
st.set_page_config(page_title="Multi-Document RAG with Evaluation", layout="wide")
st.title("📄 Multi-Document RAG with RAGAS Evaluation")

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Initialize RAG Manager section
    if st.session_state.rag_manager is None:
        if st.button("Initialize RAG Manager"):
            if initialize_rag_manager():
                st.success("✓ RAG Manager initialized!")
            else:
                st.error("Failed to initialize")

    st.divider()
    
    # Evaluation Settings
    with st.expander("📊 RAGAS Evaluation Info", expanded=False):
        st.markdown("""
        **RAGAS Metrics:**
        - **Faithfulness**: Is the answer factually consistent with context?
        - **Answer Relevancy**: Does the answer address the question?
        - **Context Precision**: Are relevant contexts ranked higher?
        - **Context Recall**: Are all relevant contexts retrieved?
        
        *Note: Precision & Recall require ground truth answers. RAGAS requires an **OpenAI API Key** for all metrics.*
        """)

# Main content
if st.session_state.rag_manager is None:
    st.warning("⚠️ Please initialize the RAG Manager from the sidebar first!")
    st.stop()

## Document ingestion section
with st.container():
    st.header("1. Ingest Documents")
    uploaded_files = st.file_uploader(
        "Upload reports/Documents (PDF, TXT, etc.)",
        type=["pdf", "txt", "docx", "html"],
        accept_multiple_files=True
    )

    if uploaded_files and st.button("Index Documents"):
        with st.spinner("Processing and Indexing Documents..."):
            handle_ingestion(uploaded_files)

# Display current ingested files
if st.session_state.ingested_docs:
    st.markdown("### Currently Indexed Documents")
    doc_map = {doc_id: filename for doc_id, filename in st.session_state.ingested_docs.items()}
    
    # Create a list of tuples for better DataFrame display
    doc_list = [(doc_id, filename) for doc_id, filename in doc_map.items()]

    st.dataframe(
        doc_list,
        column_order=["Document ID", "Filename"],
        column_config={
            0: st.column_config.TextColumn("Document ID", help="Unique 8-char identifier"),
            1: st.column_config.TextColumn("Filename", help="Original uploaded file name")
        },
        hide_index=True,
        use_container_width=True
    )
    
    # Add debug section
    with st.expander("🔍 Debug: Collection Statistics"):
        col_a, col_b = st.columns(2)
        
        with col_a:
            if st.button("Refresh Stats", key="refresh_stats"):
                stats = st.session_state.rag_manager.get_collection_stats()
                st.json(stats)
        
        with col_b:
            # Placeholder for log viewing (assuming 'logs' dir exists)
            st.caption("Log viewing placeholder (check 'logs/' directory)")
    
    st.divider()

# Query section
st.header("2. Configure & Query")

col1, col2 = st.columns(2)

with col1:
    api_choice = st.selectbox(
        "Select LLM provider",
        list(MODEL_CHOICES.keys())[:-1],
        key='api_choice_select'
    )

with col2:
    api_key = st.text_input(
        f"Enter {api_choice.upper()} API key",
        type="password",
        key='api_key_input'
    )

available_models = MODEL_CHOICES.get(api_choice, [])

if available_models:
    model_name = st.selectbox(
        f"Select model for {api_choice.upper()}",
        options=available_models,
        # Ensure 'default' key and api_choice key exists for index access
        index=available_models.index(MODEL_CHOICES['default'].get(api_choice, available_models[0])) if api_choice in MODEL_CHOICES['default'] and MODEL_CHOICES['default'][api_choice] in available_models else 0,
        key="model_name_select"
    )
else:
    model_name = st.text_input("Enter model name manually")

if st.session_state.ingested_docs:
    ingested_doc_ids = list(st.session_state.ingested_docs.keys())
    selected_doc_ids = st.multiselect(
        "Select document(s) for Query/Comparison (Select two or more to compare)",
        options=ingested_doc_ids,
        format_func=lambda x: st.session_state.ingested_docs[x],
        default=ingested_doc_ids,
        key='doc_select'
    )

    user_query = st.text_area("Ask your question or request a comparison", height=100)
    
    # Evaluation options
    enable_eval = st.checkbox("Enable RAGAS Evaluation", value=False)

    if enable_eval:
        col_eval1, col_eval2 = st.columns(2)
        
        with col_eval1:
            openai_eval_key = st.text_input(
                "OpenAI API Key (for RAGAS)",
                type="password",
                help="RAGAS requires OpenAI API for evaluation metrics (e.g., gpt-4 or gpt-3.5-turbo)",
                key='openai_eval_key'
            )
        
        with col_eval2:
            ground_truth = st.text_input(
                "Ground Truth Answer (optional)",
                help="Provide correct answer for context precision/recall metrics"
            )
    else:
        openai_eval_key = None
        ground_truth = None

    # Email report options
    st.markdown("---")
    st.subheader("📧 Email Report")
    
    col_email1, col_email2 = st.columns(2)
    
    with col_email1:
        recipient_email = st.text_input(
            "Recipient Email Address",
            placeholder="user@example.com",
            help="Enter email address to receive the report",
            key='recipient_email'
        )
    
    with col_email2:
        report_format = st.selectbox(
            "Report Format",
            options=["pdf", "html", "text"],
            index=0,
            help="Choose the format for the email report",
            key='report_format'
        )
    
    if recipient_email:
        st.info(f"📨 Report will be sent to: {recipient_email} as {report_format.upper()}")

    if st.button("🚀 Run RAG Query", type="primary"):
        if not api_key or not model_name or not user_query:
            st.error("Please ensure API key, model name and query are filled out.")
        elif enable_eval and not openai_eval_key:
             st.error("RAGAS Evaluation is enabled but the OpenAI API Key is missing.")
        else:
            with st.spinner(f"Running query with {api_choice.upper()}..."):
                handle_query(
                    query=user_query,
                    selected_doc_ids=selected_doc_ids,
                    api_choice=api_choice,
                    api_key=api_key,
                    model_name=model_name,
                    enable_eval=enable_eval,
                    ground_truth=ground_truth,
                    openai_eval_key=openai_eval_key,
                    recipient_email=recipient_email if 'recipient_email' in locals() else "",
                    report_format=report_format if 'report_format' in locals() else "pdf"
                )
else:
    st.warning("Please ingest documents in step 1 to enable querying.")
