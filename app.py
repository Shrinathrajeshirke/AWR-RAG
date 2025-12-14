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

if 'processed_files' not in st.session_state:
    st.session_state.processed_files = set()

def initialize_rag_manager():
    """Initialize RAG manager"""
    try:
        st.session_state.rag_manager = RAGSystemManager()
        return True
    except Exception as e:
        st.error(f"Failed to initialize RAGSystemManager: {e}")
        return False

def handle_ingestion(uploaded_files):
    """Auto-index uploaded files"""
    doc_processor = DocumentProcessor()

    for uploaded_file in uploaded_files:
        # Skip if already processed
        if uploaded_file.name in st.session_state.processed_files:
            continue
            
        doc_id = str(uuid.uuid4())[:8]
        file_path = os.path.join(TEMP_DIR, uploaded_file.name)
        
        # Read file content
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
            st.session_state.processed_files.add(uploaded_file.name)
            st.success(f"✅ Indexed {uploaded_file.name} (ID: {doc_id})")
        except Exception as e:
            st.error(f"Failed to process {uploaded_file.name}: {e}")
            import traceback
            st.code(traceback.format_exc())
        finally:
            if os.path.exists(file_path):
                os.remove(file_path)

def handle_query(query, selected_doc_ids, api_choice, api_key, model_name, 
                recipient_email, report_format, prompt_style, ragas_enabled):
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
        prompt_style=prompt_style,
        ragas_enabled=ragas_enabled
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
    
    # Show subtle indicator if RAGAS ran
    if ragas_enabled and result.get("evaluation_completed"):
        st.caption("📊 Quality evaluation completed (logged)")
    
    # Generate and send report if email is provided
    if recipient_email and recipient_email.strip():
        with st.spinner(f"Generating {report_format.upper()} report and sending email..."):
            try:
                from utils.email_utils import generate_report_content, generate_pdf_report, send_email_report
                
                subject = f"AWR Analysis Report - {query[:50]}..."
                
                if report_format == "text":
                    # Generate text content
                    text_content, _ = generate_report_content(
                        user_query=query,
                        answer_markdown=final_answer,
                        context=retrieved_contexts
                    )
                    
                    # Send with text as attachment
                    success = send_email_report(
                        to_email=recipient_email,
                        subject=subject,
                        body="Please find the AWR analysis report attached.",
                        subtype='plain',
                        attachment_content=text_content,
                        attachment_filename=f"awr_report_{uuid.uuid4().hex[:8]}.txt"
                    )
                    
                elif report_format == "html":
                    # Generate HTML content
                    _, html_content = generate_report_content(
                        user_query=query,
                        answer_markdown=final_answer,
                        context=retrieved_contexts
                    )
                    
                    # Send with HTML as attachment
                    success = send_email_report(
                        to_email=recipient_email,
                        subject=subject,
                        body="Please find the AWR analysis report attached.",
                        subtype='plain',
                        attachment_content=html_content,
                        attachment_filename=f"awr_report_{uuid.uuid4().hex[:8]}.html"
                    )
                    
                elif report_format == "pdf":
                    # Generate PDF file
                    pdf_path = os.path.join(TEMP_DIR, f"awr_report_{uuid.uuid4().hex[:8]}.pdf")
                    generate_pdf_report(
                        user_query=query,
                        answer_markdown=final_answer,
                        context=retrieved_contexts,
                        output_path=pdf_path
                    )
                    
                    success = send_email_report(
                        to_email=recipient_email,
                        subject=subject,
                        body="Please find the AWR analysis report attached as PDF.",
                        subtype='plain',
                        attachment_path=pdf_path
                    )
                    
                    # Clean up PDF file
                    if os.path.exists(pdf_path):
                        os.remove(pdf_path)
                
                if success:
                    st.success(f"✅ Report sent successfully to {recipient_email} as {report_format.upper()}!")
                else:
                    st.warning(f"⚠️ Email sending failed. Please check SMTP configuration.")
                    
            except Exception as e:
                st.error(f"Error generating/sending report: {e}")
                import traceback
                st.code(traceback.format_exc())


# ============================================
# MAIN APPLICATION UI
# ============================================

st.set_page_config(page_title="Oracle AWR RAG Application", layout="wide")
st.title("📊 Oracle AWR RAG Application")

# Auto-initialize RAG Manager if not initialized
if st.session_state.rag_manager is None:
    with st.spinner("🔄 Initializing RAG Manager..."):
        if initialize_rag_manager():
            st.success("✅ RAG Manager initialized successfully!")
        else:
            st.error("⚠️ Failed to initialize RAG Manager. Please check logs.")
            st.stop()

# Document ingestion section
st.header("1️⃣ Upload & Index Documents")
uploaded_files = st.file_uploader(
    "Upload AWR Reports (PDF, TXT, DOCX, HTML)",
    type=["pdf", "txt", "docx", "html"],
    accept_multiple_files=True,
    help="Documents will be automatically indexed upon upload"
)

# Auto-index on upload
if uploaded_files:
    # Check for new files
    new_files = [f for f in uploaded_files if f.name not in st.session_state.processed_files]
    
    if new_files:
        with st.spinner(f"🔄 Auto-indexing {len(new_files)} new document(s)..."):
            handle_ingestion(new_files)
        st.rerun()

# Display currently indexed documents
if st.session_state.ingested_docs:
    st.markdown("### 📚 Currently Indexed Documents")
    doc_list = [(doc_id, filename) for doc_id, filename in st.session_state.ingested_docs.items()]

    st.dataframe(
        doc_list,
        column_config={
            0: st.column_config.TextColumn("Document ID", help="Unique 8-char identifier"),
            1: st.column_config.TextColumn("Filename", help="Original uploaded file name")
        },
        hide_index=True,
        use_container_width=True
    )
    
    st.divider()

# Query section
st.header("2️⃣ Configure & Query")

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
        index=available_models.index(MODEL_CHOICES['default'].get(api_choice, available_models[0])) if api_choice in MODEL_CHOICES['default'] and MODEL_CHOICES['default'][api_choice] in available_models else 0,
        key="model_name_select"
    )
else:
    model_name = st.text_input("Enter model name manually")

if st.session_state.ingested_docs:
    ingested_doc_ids = list(st.session_state.ingested_docs.keys())
    selected_doc_ids = st.multiselect(
        "Select document(s) for Query/Comparison",
        options=ingested_doc_ids,
        format_func=lambda x: st.session_state.ingested_docs[x],
        default=ingested_doc_ids,
        help="Select two or more documents to compare",
        key='doc_select'
    )

    user_query = st.text_area(
        "Ask your question or request a comparison", 
        height=100,
        placeholder="e.g., 'Analyze this AWR report and identify performance issues'"
    )
    
    st.markdown("---")
    
    # Analysis Style Selection
    col_prompt1, col_prompt2 = st.columns([2, 1])

    with col_prompt1:
        prompt_style = st.selectbox(
            "🎨 Analysis Style",
            options=["Standard", "Detailed Step-by-Step", "Issue-Focused"],
            index=0,
            help="Choose how the AI analyzes the documents"
        )

    with col_prompt2:
        st.markdown("#### Style Guide")
        if prompt_style == "Standard":
            st.info("📝 Balanced analysis with clear structure")
        elif prompt_style == "Detailed Step-by-Step":
            st.info("🔍 Deep dive with reasoning at each step")
        elif prompt_style == "Issue-Focused":
            st.info("🎯 Executive summary with prioritized issues")

    st.markdown("---")
    
    # RAGAS Toggle - Option 1: Simple Toggle (Recommended)
    col_ragas1, col_ragas2 = st.columns([3, 1])

    with col_ragas1:
        ragas_enabled = st.toggle(
            "📊 Enable Background Quality Evaluation (RAGAS)",
            value=True,  # Default ON
            help="Evaluates answer quality in background. Scores are logged for internal tracking."
        )

    with col_ragas2:
        if ragas_enabled:
            st.success("✅ Active")
        else:
            st.info("⏸️ Disabled")

    if ragas_enabled:
        st.caption("ℹ️ Quality metrics will be logged for internal analysis. No impact on response time.")
    
    st.markdown("---")
    
    # Email report options
    st.subheader("📧 Email Report (Optional)")
    
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

    # Run Query Button
    if st.button("🚀 Run RAG Query", type="primary", use_container_width=True):
        if not api_key or not model_name or not user_query:
            st.error("⚠️ Please ensure API key, model name and query are filled out.")
        else:
            with st.spinner(f"Running query with {api_choice.upper()}..."):
                handle_query(
                    query=user_query,
                    selected_doc_ids=selected_doc_ids,
                    api_choice=api_choice,
                    api_key=api_key,
                    model_name=model_name,
                    recipient_email=recipient_email if recipient_email else "",
                    report_format=report_format if recipient_email else "pdf",
                    prompt_style=prompt_style,
                    ragas_enabled=ragas_enabled
                )
else:
    st.warning("⚠️ Please upload and index documents in step 1 to enable querying.")

# Footer
st.markdown("---")
st.caption("Oracle AWR RAG Application | Powered by LangChain, Qdrant & Advanced LLMs")