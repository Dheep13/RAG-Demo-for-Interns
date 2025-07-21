"""
Setup Verification Script for RAG Demo
=====================================

This script tests if everything is properly installed and configured
for the RAG demo. Run this before starting the main demos.

Usage: python test_setup.py
"""

import os
import sys
from pathlib import Path

def test_python_version():
    """Test if Python version is compatible."""
    print("🐍 Testing Python version...")
    
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} - Compatible")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} - Requires Python 3.8+")
        return False

def test_package_imports():
    """Test if all required packages can be imported."""
    print("\n📦 Testing package imports...")
    
    packages = [
        ("langchain", "Core LangChain framework"),
        ("langchain_openai", "OpenAI integration"),
        ("langchain_chroma", "ChromaDB integration"),
        ("langchain_community", "Community loaders"),
        ("openai", "OpenAI Python client"),
        ("tiktoken", "OpenAI tokenizer"),
        ("chromadb", "Vector database"),
        ("numpy", "Numerical computing"),
        ("sklearn", "Machine learning utilities"),
        ("dotenv", "Environment management"),
    ]
    
    optional_packages = [
        ("pypdf", "PDF processing"),
        ("docx2txt", "Word document processing"),
        ("fitz", "PyMuPDF for image extraction"),
        ("jupyter", "Jupyter notebook support"),
    ]
    
    all_passed = True
    
    # Test required packages
    for package, description in packages:
        try:
            __import__(package)
            print(f"✅ {package} - {description}")
        except ImportError:
            print(f"❌ {package} - {description} (REQUIRED)")
            all_passed = False
    
    # Test optional packages
    print("\n📦 Testing optional packages...")
    for package, description in optional_packages:
        try:
            __import__(package)
            print(f"✅ {package} - {description}")
        except ImportError:
            print(f"⚠️ {package} - {description} (Optional, some features may not work)")
    
    return all_passed

def test_environment_file():
    """Test if .env file exists and has required keys."""
    print("\n🔑 Testing environment configuration...")
    
    env_file = Path(".env")
    if not env_file.exists():
        print("⚠️ .env file not found")
        print("   Copy .env.example to .env and add your API keys")
        return False
    
    # Load environment variables
    try:
        from dotenv import load_dotenv
        load_dotenv()
        
        # Check for OpenAI API key
        if os.getenv("OPENAI_API_KEY"):
            print("✅ OPENAI_API_KEY found")
            api_key_valid = True
        else:
            print("❌ OPENAI_API_KEY not found in .env file")
            api_key_valid = False
        
        # Check optional keys
        optional_keys = ["GEMINI_API_KEY", "HF_TOKEN"]
        for key in optional_keys:
            if os.getenv(key):
                print(f"✅ {key} found (optional)")
            else:
                print(f"⚠️ {key} not found (optional)")
        
        return api_key_valid
        
    except Exception as e:
        print(f"❌ Error loading .env file: {e}")
        return False

def test_openai_connection():
    """Test OpenAI API connection."""
    print("\n🤖 Testing OpenAI API connection...")
    
    try:
        from openai import OpenAI
        
        client = OpenAI()
        
        # Test with a simple completion
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Say 'API test successful!'"}],
            max_tokens=10
        )
        
        if response.choices[0].message.content:
            print("✅ OpenAI API connection successful")
            print(f"   Response: {response.choices[0].message.content}")
            return True
        else:
            print("❌ OpenAI API returned empty response")
            return False
            
    except Exception as e:
        print(f"❌ OpenAI API connection failed: {e}")
        print("   Check your API key and internet connection")
        return False

def test_sample_documents():
    """Test if sample documents are available."""
    print("\n📄 Testing sample documents...")
    
    # Check for sample documents
    sample_dirs = ["sample_documents", "EmbeddingDocs"]
    documents_found = False
    
    for dir_name in sample_dirs:
        sample_dir = Path(dir_name)
        if sample_dir.exists():
            pdf_files = list(sample_dir.glob("*.pdf"))
            word_files = list(sample_dir.glob("*.docx")) + list(sample_dir.glob("*.doc"))
            
            if pdf_files or word_files:
                print(f"✅ Found documents in {dir_name}/:")
                for file in pdf_files + word_files:
                    print(f"   📄 {file.name}")
                documents_found = True
                break
    
    if not documents_found:
        print("⚠️ No sample documents found")
        print("   The demo will work with built-in sample text")
        print("   Add PDF/Word files to 'sample_documents/' for document processing demos")
    
    return True  # Not required for basic functionality

def test_basic_rag_functionality():
    """Test basic RAG functionality."""
    print("\n🧪 Testing basic RAG functionality...")
    
    try:
        from langchain.schema import Document
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain_openai import OpenAIEmbeddings
        from langchain_chroma import Chroma
        
        # Create a simple document
        doc = Document(page_content="This is a test document for RAG functionality.")
        
        # Test text splitting
        splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
        chunks = splitter.split_documents([doc])
        
        # Test embeddings
        embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
        
        # Test vector store creation
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=None
        )
        
        # Test similarity search
        results = vectorstore.similarity_search("test", k=1)
        
        if results:
            print("✅ Basic RAG functionality working")
            print(f"   Created {len(chunks)} chunks")
            print(f"   Vector store created successfully")
            print(f"   Similarity search returned {len(results)} results")
            return True
        else:
            print("❌ Basic RAG functionality failed - no search results")
            return False
            
    except Exception as e:
        print(f"❌ Basic RAG functionality failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 RAG Demo Setup Verification")
    print("=" * 50)
    
    tests = [
        ("Python Version", test_python_version),
        ("Package Imports", test_package_imports),
        ("Environment File", test_environment_file),
        ("OpenAI Connection", test_openai_connection),
        ("Sample Documents", test_sample_documents),
        ("Basic RAG Functionality", test_basic_rag_functionality),
    ]
    
    results = {}
    critical_failures = 0
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            results[test_name] = test_func()
            if not results[test_name] and test_name in ["Python Version", "Package Imports", "OpenAI Connection"]:
                critical_failures += 1
        except Exception as e:
            print(f"❌ {test_name} failed with error: {e}")
            results[test_name] = False
            if test_name in ["Python Version", "Package Imports", "OpenAI Connection"]:
                critical_failures += 1
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 SETUP VERIFICATION SUMMARY")
    print("=" * 50)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\n📈 Overall: {passed}/{total} tests passed")
    
    if critical_failures == 0:
        print("\n🎉 Setup verification successful!")
        print("\n🚀 You're ready to run the RAG demos:")
        print("   1. jupyter notebook Comprehensive_RAG_Demo.ipynb")
        print("   2. python Enhanced_RAG_Demo_with_Embeddings.py")
        print("   3. python multimodal_rag_demo.py")
    else:
        print(f"\n❌ {critical_failures} critical issues found!")
        print("\n🔧 Fix these issues before running the demos:")
        if not results.get("Python Version", True):
            print("   • Upgrade to Python 3.8 or higher")
        if not results.get("Package Imports", True):
            print("   • Run: pip install -r requirements.txt")
        if not results.get("OpenAI Connection", True):
            print("   • Check your OPENAI_API_KEY in .env file")
        
        print("\n💡 For detailed help, see docs/TROUBLESHOOTING.md")

if __name__ == "__main__":
    main()
