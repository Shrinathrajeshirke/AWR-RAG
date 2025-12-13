# 📊 Oracle AWR RAG Application

A powerful Retrieval-Augmented Generation (RAG) system designed specifically for Oracle Database Administrators to analyze AWR (Automatic Workload Repository) reports using advanced AI models. Upload multiple AWR reports, ask questions in natural language, and receive intelligent insights with actionable recommendations.

**🚀 Live Demo**: [https://awr-rag.streamlit.app/](https://awr-rag.streamlit.app/)

---

## 🌟 Key Features

### 🔍 **Multi-Document Analysis**
- Upload and index multiple AWR reports simultaneously
- Compare performance across different time periods
- Track trends and identify degradation patterns
- Cross-reference metrics from multiple databases

### 🤖 **AI-Powered Insights**
- **Three Analysis Styles**:
  - **Standard**: Balanced analysis with clear structure
  - **Detailed Step-by-Step**: Deep dive with reasoning at each step
  - **Issue-Focused**: Executive summary with prioritized issues
- Natural language query interface
- Root cause analysis with evidence-based reasoning
- Actionable recommendations with priority and effort estimates

### 🎯 **Advanced RAG Capabilities**
- Semantic search using vector embeddings (Qdrant)
- Context-aware retrieval with document filtering
- Support for multiple LLM providers:
  - **OpenAI**: GPT-4o, GPT-4-turbo, GPT-3.5-turbo
  - **Groq**: Llama-3.1-8b-instant (ultra-fast inference)
  - **HuggingFace**: Zephyr-7b, Mixtral-8x7B

### 📊 **RAGAS Evaluation Framework**
- **Faithfulness**: Factual consistency with context
- **Answer Relevancy**: Question-answer alignment
- **Context Precision**: Retrieval quality assessment
- **Context Recall**: Completeness of retrieved information

### 📧 **Report Generation & Distribution**
- Export analysis in multiple formats:
  - **PDF**: Professional reports with formatting
  - **HTML**: Interactive web-based reports
  - **Text**: Plain text for easy sharing
- Email delivery with SMTP support
- Customizable report templates

### 🔐 **Enterprise-Ready**
- Secure API key management
- Environment variable support
- Comprehensive logging and debugging
- Collection statistics and monitoring

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.11** or higher
- pip (Python package manager)
- Git

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/Shrinathrajeshirke/AWR-RAG.git
cd AWR-RAG
```

2. **Create a virtual environment**
```bash
python create -p venv python==3.11 -y

# Windows
venv\Scripts\activate

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**

Create a `.env` file in the project root:

```env
# SMTP Configuration (for email reports)
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=your_email@gmail.com
SMTP_PASSWORD=your_app_password

# Vector Database (optional - default is in-memory)
QDRANT_LOCATION=:memory:

# Embedding Model (optional - default is all-MiniLM-L6-v2)
EMBEDDING_MODEL_NAME=all-MiniLM-L6-v2
```

5. **Run the application**
```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

---

## 📖 Usage Guide

### Step 1: Initialize the System

The RAG Manager initializes automatically when you start the app. You'll see:
```
✅ RAG Manager initialized successfully!
```

### Step 2: Upload AWR Reports

1. Navigate to **"1️⃣ Ingest Documents"**
2. Click **"Upload reports/Documents"**
3. Select one or more AWR reports (PDF, TXT, DOCX, HTML)
4. Click **"📥 Index Documents"**
5. Wait for processing - you'll see progress for each file:
   - Document chunking
   - Vector embedding generation
   - Indexing to Qdrant

### Step 3: Configure Query Settings

1. **Select LLM Provider**: 
   - OpenAI (best quality)
   - Groq (fastest)
   - HuggingFace (open-source)

2. **Enter API Key**: Provide your API key for the selected provider
   - OpenAI: https://platform.openai.com/api-keys
   - Groq: https://console.groq.com/keys
   - HuggingFace: https://huggingface.co/settings/tokens

3. **Select Model**: Choose from available models
   - GPT-4o (recommended for complex analysis)
   - Llama-3.1-8b (fastest, good quality)

4. **Select Documents**: Choose which indexed documents to query

5. **Choose Analysis Style**:
   - **Standard**: Best for general queries
   - **Detailed Step-by-Step**: For learning and deep analysis
   - **Issue-Focused**: For executive summaries and incident reports

### Step 4: Ask Questions

Example queries:

**Performance Analysis:**
```
"Analyze this AWR report and identify performance issues"
"What are the top 3 bottlenecks in this database?"
"Identify all metrics that are outside normal ranges"
```

**Wait Events:**
```
"What are the top wait events and how can we resolve them?"
"Explain the 'db file sequential read' wait event"
"Why is 'latch: shared pool' consuming so much time?"
```

**SQL Tuning:**
```
"Which SQL statements need optimization?"
"Analyze the top SQL by elapsed time"
"Are there any full table scans that should be avoided?"
```

**Comparison:**
```
"Compare the CPU usage between these two reports"
"What improved after the tuning changes?"
"Show me the performance trend over these time periods"
```

**Recommendations:**
```
"What immediate actions should be taken?"
"Provide a prioritized list of tuning recommendations"
"What are the quick wins for performance improvement?"
```

### Step 5: Review Results

- View AI-generated analysis with structured insights
- Check retrieved context count (shows how many relevant chunks were found)
- Review evaluation scores (if enabled)
- Download or email reports in your preferred format

---

## 🏗️ Project Structure

```
AWR-RAG/
│
├── 📁 ingestor/
│   ├── document_loader.py        # Document loading and chunking
│   ├── rag_system_manager.py     # Core RAG logic, vector store, agents
│   └── __init__.py
│
├── 📁 llm_api/
│   ├── api_manager.py             # LLM provider management (OpenAI, Groq, HF)
│   └── __init__.py
│
├── 📁 utils/
│   ├── logger.py                  # Centralized logging configuration
│   ├── exception.py               # Custom exception handling
│   ├── email_utils.py             # Email sending and report generation
│   └── __init__.py
│
├── 📁 temp_files/                 # Temporary file storage (auto-created)
├── 📁 logs/                       # Application logs (auto-created)
│
├── app.py                         # Streamlit UI application (main entry point)
├── config.py                      # Configuration settings
├── requirements.txt               # Python dependencies
├── .env                          # Environment variables (create this, not in git)
├── .gitignore                    # Git ignore rules
└── README.md                     # This file
```

---

## ⚙️ Configuration

### Core Settings (`config.py`)

```python
# Text Chunking
CHUNK_SIZE = 1000           # Characters per chunk
CHUNK_OVERLAP = 200         # Overlap between chunks

# Embeddings
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # Sentence transformers model

# Vector Database
QDRANT_COLLECTION_NAME = "multi_doc_comparison_rag"
QDRANT_LOCATION = ":memory:"  # Use ":memory:" or file path for persistence

# LLM Models
MODEL_CHOICES = {
    "openai": [
        "gpt-4o",                    # Latest GPT-4 optimized
        "gpt-4-turbo",              # Fast GPT-4 variant
        "gpt-3.5-turbo-0125"        # Cost-effective option
    ],
    "groq": [
        "llama-3.1-8b-instant"      # Ultra-fast Llama model
    ],
    "huggingface": [
        "HuggingFaceH4/zephyr-7b-beta",           # General purpose
        "mistralai/Mixtral-8x7B-Instruct-v0.1"   # Powerful mixture-of-experts
    ]
}

# RAGAS Evaluation Metrics
RAGAS_METRICS = [
    "faithfulness",         # Factual consistency
    "answer_relevancy",     # Question alignment
    "context_precision",    # Retrieval quality
    "context_recall"        # Completeness
]
```

### Environment Variables (`.env`)

```env
# Email Configuration (Optional)
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=your_email@gmail.com
SMTP_PASSWORD=your_app_password

# Vector Store (Optional - defaults from config.py)
QDRANT_LOCATION=:memory:
QDRANT_COLLECTION_NAME=multi_doc_comparison_rag

# Embedding Model (Optional)
EMBEDDING_MODEL_NAME=all-MiniLM-L6-v2
```

---

## 📊 Analysis Styles Explained

### 🟢 Standard Analysis
**Best for**: Quick reviews and general performance assessment

**What you get**:
- Key metrics analysis (CPU, DB Time, Wait Events)
- Issue identification with severity levels
- Root cause analysis with evidence
- Solutions with priority and effort estimates
- Clear structure with document citations

**Example Output**:
```
Step 1: Key Metrics Analysis
- CPU Usage: 85% (Warning: Above 80% threshold)
- DB Time: 15,234 seconds
- Top Wait Event: db file sequential read (45% of DB time)

Step 2: Issue Identification
🔴 CRITICAL: High CPU usage
🟡 WARNING: Excessive sequential reads

Step 3: Root Cause & Solutions
...
```

---

### 🔵 Detailed Step-by-Step
**Best for**: Learning, training, or detailed investigations

**What you get**:
- Systematic 4-step analysis process
- Metric extraction with explanations
- Threshold comparisons against Oracle best practices
- Reasoning shown at each step
- Educational format explaining the "why"

**Example Output**:
```
STEP 1 - Metric Extraction:
I'm examining the AWR report to identify key performance indicators...
- CPU usage: 85% (extracted from Load Profile section)
- DB Time: 15,234s (total database time during snapshot)
...

STEP 2 - Threshold Comparison:
Comparing metrics against Oracle best practices:
- CPU: 85% vs 80% threshold → EXCEEDS (⚠️ Warning)
...
```

---

### 🟣 Issue-Focused
**Best for**: Management reports and incident response

**What you get**:
- Executive summary with health score
- Categorized issues (Critical/Warning/Info)
- Metric comparison tables
- Top 3 prioritized action items
- Professional report format

**Example Output**:
```
### 📊 EXECUTIVE SUMMARY
Database health score: 6.5/10. Two critical issues requiring immediate 
attention: high CPU utilization and excessive hard parsing.

### 🔴 CRITICAL ISSUES
**Issue:** Hard Parsing Storm
- **Metric:** Parse ratio = 90% (Expected: <10%)
- **Root Cause:** SQL without bind variables
- **Impact:** 92% CPU consumption
- **Solution:** Implement bind variables in application
- **Priority:** High | **Effort:** Medium

### 🎯 TOP 3 ACTION ITEMS
1. Enable cursor_sharing=FORCE (Expected: 40% CPU reduction)
2. Add index on ORDERS.ORDER_DATE (Expected: 30% I/O reduction)
3. Increase shared_pool_size to 2GB (Expected: reduce latch waits)
```

---

## 🔧 Advanced Features

### 🎓 RAGAS Evaluation

Assess the quality of RAG responses using standardized metrics:

**How to enable**:
1. Expand **"📊 Advanced: RAGAS Evaluation (Optional)"**
2. Check **"Enable RAGAS Evaluation"**
3. Provide **OpenAI API key** (required - RAGAS uses GPT-4 for evaluation)
4. Optionally provide **Ground Truth Answer** for precision/recall metrics
5. Run your query

**Interpretation**:
- **Faithfulness (0-1)**: Higher = more factually consistent with context
  - >0.8: Excellent, no hallucinations
  - 0.6-0.8: Good, minor inconsistencies
  - <0.6: Poor, contains unsupported claims

- **Answer Relevancy (0-1)**: Higher = better addresses the question
  - >0.8: Highly relevant
  - 0.6-0.8: Mostly relevant
  - <0.6: Off-topic or incomplete

- **Context Precision (0-1)**: Requires ground truth
  - Measures if relevant chunks are ranked higher

- **Context Recall (0-1)**: Requires ground truth
  - Measures if all relevant information was retrieved

### 📧 Email Reports

**Setup for Gmail**:
1. Enable 2-Factor Authentication on your Google account
2. Generate App Password:
   - Go to https://myaccount.google.com/apppasswords
   - Select "Mail" and "Other (Custom name)"
   - Copy the 16-character password
3. Add to `.env`:
   ```env
   SMTP_USERNAME=your_email@gmail.com
   SMTP_PASSWORD=abcd efgh ijkl mnop  # Your app password
   ```

**How to use**:
1. Enter recipient email address
2. Choose format: PDF, HTML, or Text
3. Run your query
4. Report is automatically generated and sent

**Report Contents**:
- Query asked
- AI-generated answer
- Retrieved context sources
- Evaluation scores (if enabled)
- Professional formatting

### 🔄 Multi-Document Comparison

**When to use**:
- Compare before/after tuning changes
- Track performance degradation over time
- Analyze weekly/monthly trends
- Validate optimization efforts

**How to use**:
1. Index multiple AWR reports
2. Select 2 or more documents from the list
3. Ask comparison questions:
   ```
   "Compare CPU usage across these reports"
   "What metrics improved between these periods?"
   "Show me the trend in wait events"
   ```

**What you get**:
- Side-by-side metric comparison
- Identification of improvements and degradations
- New issues that appeared
- Issues that were resolved
- Trend analysis with recommendations

---

## 🚢 Deployment

### Streamlit Cloud (Easiest - FREE)

**Already deployed at**: https://awr-rag.streamlit.app/

To deploy your own instance:

1. **Push code to GitHub**
```bash
git add .
git commit -m "Initial commit"
git push origin main
```

2. **Deploy on Streamlit Cloud**
   - Go to https://share.streamlit.io/
   - Click "New app"
   - Select your repository: `Shrinathrajeshirke/AWR-RAG`
   - Main file: `app.py`
   - Click "Advanced settings" → "Secrets"
   - Add your secrets:
     ```toml
     SMTP_SERVER = "smtp.gmail.com"
     SMTP_PORT = 587
     SMTP_USERNAME = "your_email@gmail.com"
     SMTP_PASSWORD = "your_app_password"
     ```
   - Click "Deploy"

3. **Your app will be live at**: `https://your-app-name.streamlit.app/`


## 📦 Dependencies

### Core Libraries

| Library | Version | Purpose |
|---------|---------|---------|
| `streamlit` | Latest | Web UI framework |
| `langchain` | Latest | RAG orchestration |
| `langchain_openai` | Latest | OpenAI integration |
| `langchain_groq` | Latest | Groq integration |
| `langchain_huggingface` | Latest | HuggingFace integration |
| `qdrant-client` | Latest | Vector database |
| `sentence-transformers` | Latest | Embedding generation |
| `ragas` | Latest | RAG evaluation |
| `reportlab` | >=4.0.0 | PDF generation |
| `python-dotenv` | Latest | Environment variables |
| `markdown` | Latest | Markdown to HTML |

### Complete Installation

```bash
pip install -r requirements.txt
```

See `requirements.txt` for the full list.

---

## 🐛 Troubleshooting

### Issue: "reportlab not found" error
**Solution**:
```bash
pip uninstall reportlab
pip install reportlab --no-cache-dir
```

---

### Issue: Email sending fails
**Possible causes**:
1. SMTP credentials not set in `.env`
2. Using regular password instead of App Password (Gmail)
3. Firewall blocking port 587

**Solution**:
- Verify credentials in `.env` file
- For Gmail: Use App Password (https://myaccount.google.com/apppasswords)
- Test SMTP connection:
  ```python
  import smtplib
  server = smtplib.SMTP("smtp.gmail.com", 587)
  server.starttls()
  server.login("your_email@gmail.com", "your_app_password")
  print("✅ Connection successful!")
  ```

---

### Issue: "No documents retrieved" error
**Possible causes**:
1. Documents not indexed properly
2. Document IDs don't match filter
3. Query not matching any content

**Solution**:
- Check "🔍 Debug: Collection Statistics" to verify indexing
- Re-upload and index documents
- Try a broader query
- Check logs in `logs/` directory

---

### Issue: LLM API errors
**Common errors**:
- `Invalid API key`: Check your API key is correct
- `Rate limit exceeded`: Wait or upgrade plan
- `Model not found`: Select a different model

**Solution**:
```bash
# Test API key
curl https://api.openai.com/v1/models \
  -H "Authorization: Bearer YOUR_API_KEY"
```

---

### Issue: Out of memory (Qdrant)
**Solution**: Use file-based storage instead of `:memory:`

In `config.py`:
```python
QDRANT_LOCATION = "./qdrant_storage"  # Instead of ":memory:"
```

---

### Issue: Slow embedding generation
**Solution**: Use GPU acceleration or smaller model

For GPU (if available):
```python
# In config.py
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # Lightweight model
```

---

## 🛣️ Roadmap

### ✅ Phase 1: Core RAG System (Completed)
- [x] Multi-document upload and indexing
- [x] Vector search with Qdrant
- [x] Multiple LLM provider support
- [x] Three analysis styles
- [x] RAGAS evaluation
- [x] Report generation (PDF/HTML/Text)
- [x] Email delivery

### 🚧 Phase 2: Enhanced Analytics (In Progress)
- [ ] Structured AWR parsing (extract tables, SQL IDs)
- [ ] Automated anomaly detection
- [ ] Pattern matching for common issues
- [ ] Threshold-based alerting
- [ ] Visualization of metrics (charts/graphs)

### 📅 Phase 3: Knowledge Base (Planned)
- [ ] Index Oracle documentation
- [ ] Historical solutions database
- [ ] Best practices library
- [ ] Custom playbook integration
- [ ] Community-contributed patterns

### 🔮 Phase 4: Multi-Agent System (Future)
- [ ] Specialized diagnostic agents
- [ ] SQL optimization agent
- [ ] Capacity planning agent
- [ ] Report generation agent
- [ ] Agent orchestration with LangGraph

### 🌐 Phase 5: Enterprise Features (Future)
- [ ] Real-time database connectivity
- [ ] Integration with monitoring tools (Grafana, Prometheus)
- [ ] Scheduled report generation
- [ ] Multi-tenant support
- [ ] Role-based access control
- [ ] Audit logging

---

## 🤝 Contributing

Contributions are welcome! Whether it's bug reports, feature requests, or code contributions.

### How to Contribute

1. **Fork the repository**
   ```bash
   git clone https://github.com/Shrinathrajeshirke/AWR-RAG.git
   ```

2. **Create a feature branch**
   ```bash
   git checkout -b feature/AmazingFeature
   ```

3. **Make your changes**
   - Follow PEP 8 style guidelines
   - Add comments and docstrings
   - Update tests if applicable

4. **Commit your changes**
   ```bash
   git commit -m 'Add some AmazingFeature'
   ```

5. **Push to the branch**
   ```bash
   git push origin feature/AmazingFeature
   ```

6. **Open a Pull Request**
   - Describe your changes
   - Reference any related issues
   - Wait for review

### Development Setup

```bash
# Clone and setup
git clone https://github.com/Shrinathrajeshirke/AWR-RAG.git
cd AWR-RAG
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt

# Run tests (if available)
pytest tests/

# Run the app
streamlit run app.py
```

## 👥 Authors

- **Shrinath Rajeshirke** - *Creator & Maintainer*
  - GitHub: [@Shrinathrajeshirke](https://github.com/Shrinathrajeshirke)
  - Repository: [AWR-RAG](https://github.com/Shrinathrajeshirke/AWR-RAG)

---

## 🙏 Acknowledgments

- **Oracle Corporation** - For the AWR reporting framework
- **LangChain** - For the powerful RAG orchestration tools
- **Anthropic** - For Claude AI capabilities
- **Streamlit** - For the amazing UI framework
- **Qdrant** - For the efficient vector database
- **HuggingFace** - For open-source models and embeddings
- **OpenAI** - For GPT models
- **Groq** - For ultra-fast inference
- Open-source community for various libraries

---

## 📧 Support

- **Issues**: [GitHub Issues](https://github.com/Shrinathrajeshirke/AWR-RAG/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Shrinathrajeshirke/AWR-RAG/discussions)
- **Live Demo**: [awr-rag.streamlit.app](https://awr-rag.streamlit.app/)

---

## 📈 Project Stats

![GitHub stars](https://img.shields.io/github/stars/Shrinathrajeshirke/AWR-RAG?style=social)
![GitHub forks](https://img.shields.io/github/forks/Shrinathrajeshirke/AWR-RAG?style=social)
![GitHub watchers](https://img.shields.io/github/watchers/Shrinathrajeshirke/AWR-RAG?style=social)

---

**⭐ Star this repo** if you find it helpful!

**Made with ❤️ for DBAs by DBAs**
