# 🛠️ Complete Setup Guide

## 📋 Prerequisites

- **Python 3.8+** (Python 3.9-3.11 recommended)
- **Git** for cloning the repository
- **OpenAI API account** with credits
- **Internet connection** for package installation

## 🚀 Step-by-Step Setup

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd RAG-Demo-for-Interns
```

### 2. Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv rag_demo_env

# Activate virtual environment
# On Windows:
rag_demo_env\Scripts\activate
# On macOS/Linux:
source rag_demo_env/bin/activate
```

### 3. Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt

# Or install minimal version (basic RAG only)
pip install langchain langchain-openai langchain-chroma tiktoken chromadb openai numpy scikit-learn python-dotenv jupyter
```

### 4. Set Up API Keys

```bash
# Copy the example environment file
cp .env.example .env

# Edit the .env file with your API keys
# Use any text editor (notepad, vim, vscode, etc.)
```

**Required API Keys:**

#### OpenAI API Key (Required)
1. Go to https://platform.openai.com/api-keys
2. Create a new API key
3. Add to `.env` file: `OPENAI_API_KEY=your-key-here`

#### Optional API Keys:
- **Gemini**: https://makersuite.google.com/app/apikey
- **Hugging Face**: https://huggingface.co/settings/tokens

### 5. Add Sample Documents (Optional)

Create a `sample_documents` folder and add your PDF/Word files:

```bash
mkdir sample_documents
# Copy your PDF/Word files to this folder
```

### 6. Verify Setup

```bash
python test_setup.py
```

This will check:
- ✅ Python version compatibility
- ✅ All packages installed
- ✅ API keys configured
- ✅ Basic functionality working

## 🎯 Running the Demos

### Option 1: Interactive Jupyter Notebook (Recommended for Learning)

```bash
jupyter notebook Comprehensive_RAG_Demo.ipynb
```

**Features:**
- Step-by-step learning experience
- All code embedded in notebook
- Progressive difficulty levels
- Interactive exploration

### Option 2: Advanced Embedding Analysis

```bash
python Enhanced_RAG_Demo_with_Embeddings.py
```

**Features:**
- Deep embedding visualization
- Direct database querying
- Interactive command-line interface
- Mathematical analysis

### Option 3: Complete Multi-Modal Demo

```bash
python multimodal_rag_demo.py
```

**Features:**
- All three RAG approaches
- Image processing capabilities
- Real document handling
- Production-ready code

## 🔧 Configuration Options

### Environment Variables (.env file)

```bash
# Required
OPENAI_API_KEY=your-openai-api-key

# Optional Model Configuration
DEFAULT_EMBEDDING_MODEL=text-embedding-ada-002
DEFAULT_LLM_MODEL=gpt-3.5-turbo
DEFAULT_VISION_MODEL=gpt-4o

# Optional RAG Configuration
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
RETRIEVAL_K=5
```

### Custom Document Folder

By default, the system looks for documents in:
1. `sample_documents/` folder
2. `EmbeddingDocs/` folder

You can add your own PDF/Word files to either folder.

## 🐳 Docker Setup (Alternative)

If you prefer Docker:

```bash
# Build the Docker image
docker build -t rag-demo .

# Run the container
docker run -p 8888:8888 -v $(pwd):/workspace rag-demo
```

## 🧪 Testing Individual Components

### Test OpenAI Connection Only
```python
from openai import OpenAI
client = OpenAI()
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=10
)
print(response.choices[0].message.content)
```

### Test Embeddings Only
```python
from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings()
result = embeddings.embed_query("test")
print(f"Embedding dimensions: {len(result)}")
```

### Test Document Loading Only
```python
from langchain_community.document_loaders import PyPDFLoader
loader = PyPDFLoader("sample_documents/your_file.pdf")
docs = loader.load()
print(f"Loaded {len(docs)} pages")
```

## 📊 System Requirements

### Minimum Requirements
- **RAM**: 4GB
- **Storage**: 2GB free space
- **Internet**: Stable connection for API calls

### Recommended Requirements
- **RAM**: 8GB+
- **Storage**: 5GB free space
- **CPU**: Multi-core processor
- **Internet**: High-speed connection

## 🔍 Verification Checklist

Before running demos, ensure:

- [ ] Python 3.8+ installed
- [ ] Virtual environment activated
- [ ] All packages installed (`pip list` shows langchain, openai, etc.)
- [ ] `.env` file exists with OPENAI_API_KEY
- [ ] `test_setup.py` passes all critical tests
- [ ] Internet connection working
- [ ] OpenAI account has sufficient credits

## 💰 Cost Estimation

### OpenAI API Costs (Approximate)
- **Basic RAG Demo**: $0.01-0.05
- **Document RAG Demo**: $0.05-0.20
- **Multi-Modal RAG Demo**: $0.10-0.50
- **Complete Learning Session**: $0.25-1.00

### Cost Optimization Tips
- Use `gpt-3.5-turbo` instead of `gpt-4` for text generation
- Limit document size for initial testing
- Use smaller chunk sizes to reduce embedding costs
- Set reasonable limits on retrieval count (k=3-5)

## 🆘 Getting Help

If you encounter issues:

1. **Check the troubleshooting guide**: `docs/TROUBLESHOOTING.md`
2. **Run the test script**: `python test_setup.py`
3. **Check the logs**: Look for error messages in terminal output
4. **Verify API keys**: Ensure they're correctly set in `.env`
5. **Update packages**: `pip install --upgrade -r requirements.txt`

## 🎓 Next Steps

Once setup is complete:

1. **Start with Basic RAG** - Understand fundamental concepts
2. **Progress to Document RAG** - Learn real-world applications
3. **Explore Multi-Modal RAG** - See cutting-edge capabilities
4. **Experiment with your own data** - Apply to your use cases
5. **Read the learning path**: `docs/LEARNING_PATH.md`

Happy learning! 🚀
