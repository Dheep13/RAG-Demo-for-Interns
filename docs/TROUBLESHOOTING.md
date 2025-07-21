# 🚨 Troubleshooting Guide

## 🔧 Common Issues and Solutions

### 1. Python Version Issues

#### Problem: "Python version not supported"
```
Error: Python 3.7.x - Requires Python 3.8+
```

**Solutions:**
- **Windows**: Download Python 3.9+ from python.org
- **macOS**: `brew install python@3.9` or download from python.org
- **Linux**: `sudo apt update && sudo apt install python3.9`

#### Problem: Multiple Python versions
```
Error: Command 'python' not found
```

**Solutions:**
- Try `python3` instead of `python`
- Use `py -3.9` on Windows
- Check with `python --version` and `python3 --version`

### 2. Package Installation Issues

#### Problem: "No module named 'langchain'"
```
ModuleNotFoundError: No module named 'langchain'
```

**Solutions:**
```bash
# Ensure you're in the right environment
pip install -r requirements.txt

# If that fails, install individually:
pip install langchain langchain-openai langchain-chroma

# Check if packages are installed:
pip list | grep langchain
```

#### Problem: "Microsoft Visual C++ 14.0 is required" (Windows)
```
Error: Microsoft Visual C++ 14.0 is required
```

**Solutions:**
- Install Visual Studio Build Tools
- Or install pre-compiled packages: `pip install --only-binary=all -r requirements.txt`

#### Problem: "Failed building wheel for chromadb"
```
Error: Failed building wheel for chromadb
```

**Solutions:**
```bash
# Try installing with conda instead
conda install -c conda-forge chromadb

# Or use pre-compiled version
pip install --upgrade pip setuptools wheel
pip install chromadb --no-cache-dir
```

### 3. API Key Issues

#### Problem: "OPENAI_API_KEY not found"
```
Error: OPENAI_API_KEY not found
```

**Solutions:**
1. Check if `.env` file exists in project root
2. Verify the file contains: `OPENAI_API_KEY=your-actual-key`
3. No spaces around the `=` sign
4. No quotes around the key (unless part of the key)
5. Restart your terminal/IDE after creating `.env`

#### Problem: "Invalid API key"
```
Error: Incorrect API key provided
```

**Solutions:**
1. Verify key is correct (copy-paste from OpenAI dashboard)
2. Check if key has been revoked
3. Ensure your OpenAI account has credits
4. Try creating a new API key

#### Problem: "Rate limit exceeded"
```
Error: Rate limit reached for requests
```

**Solutions:**
1. Wait a few minutes and try again
2. Upgrade your OpenAI plan
3. Reduce the number of API calls in demos
4. Use smaller chunk sizes

### 4. Document Processing Issues

#### Problem: "No documents found"
```
Error: sample_documents folder not found
```

**Solutions:**
1. Create the folder: `mkdir sample_documents`
2. Add PDF/Word files to the folder
3. Or use the built-in sample data (basic RAG will still work)

#### Problem: "Error loading PDF"
```
Error: PdfReadError: EOF marker not found
```

**Solutions:**
1. Check if PDF file is corrupted
2. Try a different PDF file
3. Ensure PDF is not password-protected
4. Update pypdf: `pip install --upgrade pypdf`

#### Problem: "Error loading Word document"
```
Error: PackageNotFoundError: Package not found
```

**Solutions:**
1. Install docx2txt: `pip install docx2txt`
2. Ensure file has .docx extension (not .doc)
3. Try opening the file in Word to verify it's not corrupted

### 5. Memory and Performance Issues

#### Problem: "Out of memory" errors
```
Error: CUDA out of memory / RAM exhausted
```

**Solutions:**
1. Reduce chunk size: `CHUNK_SIZE=500`
2. Process fewer documents at once
3. Reduce retrieval count: `k=3` instead of `k=10`
4. Close other applications
5. Use a machine with more RAM

#### Problem: Slow performance
```
Demo is running very slowly
```

**Solutions:**
1. Check internet connection (API calls need good connection)
2. Use smaller documents for testing
3. Reduce the number of chunks processed
4. Use `gpt-3.5-turbo` instead of `gpt-4`

### 6. Jupyter Notebook Issues

#### Problem: "Jupyter not found"
```
Error: 'jupyter' is not recognized
```

**Solutions:**
```bash
pip install jupyter
# Or
pip install notebook

# Then try:
jupyter notebook
```

#### Problem: Kernel issues
```
Error: Kernel died, restarting
```

**Solutions:**
1. Restart the kernel: Kernel → Restart
2. Clear all outputs: Cell → All Output → Clear
3. Check for memory issues (see section 5)
4. Restart Jupyter entirely

#### Problem: "Module not found in notebook"
```
ModuleNotFoundError in Jupyter but works in terminal
```

**Solutions:**
```bash
# Install in the correct kernel
python -m ipykernel install --user --name=rag_demo_env

# Or install packages in notebook:
!pip install langchain
```

### 7. Network and Firewall Issues

#### Problem: "Connection timeout"
```
Error: HTTPSConnectionPool timeout
```

**Solutions:**
1. Check internet connection
2. Try different network (mobile hotspot)
3. Check corporate firewall settings
4. Use VPN if behind restrictive firewall

#### Problem: "SSL Certificate error"
```
Error: SSL: CERTIFICATE_VERIFY_FAILED
```

**Solutions:**
```bash
# Update certificates
pip install --upgrade certifi

# Or bypass SSL (not recommended for production)
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
```

### 8. Environment and Path Issues

#### Problem: "Command not found"
```
Error: 'python' is not recognized
```

**Solutions:**
1. Add Python to PATH during installation
2. Use full path: `C:\Python39\python.exe`
3. Use Python Launcher: `py -3.9`

#### Problem: Virtual environment issues
```
Error: Virtual environment not activating
```

**Solutions:**
```bash
# Windows
venv\Scripts\activate.bat

# PowerShell
venv\Scripts\Activate.ps1

# macOS/Linux
source venv/bin/activate

# Check if activated (should show (venv) in prompt)
```

## 🧪 Diagnostic Commands

### Check Python and Package Versions
```bash
python --version
pip --version
pip list | grep -E "(langchain|openai|chromadb)"
```

### Test Individual Components
```bash
# Test Python imports
python -c "import langchain; print('LangChain OK')"
python -c "import openai; print('OpenAI OK')"
python -c "import chromadb; print('ChromaDB OK')"

# Test API connection
python -c "from openai import OpenAI; print(OpenAI().models.list().data[0].id)"
```

### Check Environment
```bash
# Check if .env file exists and has content
cat .env  # Linux/Mac
type .env  # Windows

# Check environment variables
python -c "import os; print(os.getenv('OPENAI_API_KEY', 'NOT_FOUND'))"
```

## 🆘 Getting Additional Help

### 1. Run the Test Script
```bash
python test_setup.py
```
This will identify most common issues automatically.

### 2. Enable Debug Mode
Add to your `.env` file:
```
LOG_LEVEL=DEBUG
```

### 3. Check System Resources
```bash
# Check available memory
python -c "import psutil; print(f'RAM: {psutil.virtual_memory().percent}%')"

# Check disk space
python -c "import shutil; print(f'Disk: {shutil.disk_usage('.').free // (1024**3)} GB free')"
```

### 4. Create Minimal Test Case
```python
# Save as test_minimal.py
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

load_dotenv()
embeddings = OpenAIEmbeddings()
result = embeddings.embed_query("test")
print(f"Success! Embedding dimensions: {len(result)}")
```

### 5. Contact Support
If all else fails:
- 📧 Email: [your-support-email]
- 💬 Slack: #ai-learning-help
- 🐛 GitHub Issues: Create an issue with:
  - Your operating system
  - Python version
  - Error message (full traceback)
  - Output of `test_setup.py`

## 📋 Pre-Demo Checklist

Before running any demo, verify:

- [ ] `python test_setup.py` passes all critical tests
- [ ] `.env` file exists with valid OPENAI_API_KEY
- [ ] Virtual environment is activated (if using one)
- [ ] Internet connection is stable
- [ ] OpenAI account has sufficient credits
- [ ] No other resource-intensive applications running

## 🔄 Reset Instructions

If everything is broken and you want to start fresh:

```bash
# 1. Delete virtual environment
rm -rf rag_demo_env  # Linux/Mac
rmdir /s rag_demo_env  # Windows

# 2. Create new virtual environment
python -m venv rag_demo_env
source rag_demo_env/bin/activate  # Linux/Mac
rag_demo_env\Scripts\activate  # Windows

# 3. Reinstall everything
pip install --upgrade pip
pip install -r requirements.txt

# 4. Verify setup
python test_setup.py
```

Remember: Most issues are related to environment setup, API keys, or package versions. The test script will catch 90% of problems! 🚀
