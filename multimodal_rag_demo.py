"""
Comprehensive RAG Demo - All Methods Combined
============================================

This comprehensive RAG implementation demonstrates multiple approaches:

1. BASIC TEXT RAG: Simple text embedding with sample data
2. DOCUMENT RAG: PDF and Word document processing
3. MULTI-MODAL RAG: Text + Images with GPT-4 Vision

Perfect for segmenting into Jupyter notebook sections for intern training.

Features:
- Sample text data embedding
- PDF and Word document text extraction
- Image extraction from PDFs
- GPT-4 Vision image descriptions
- Combined text + image embeddings
- Interactive Q&A with source attribution

Author: AI Learning Demo
Date: 2024
"""

import os
import sys
from pathlib import Path
import base64
from io import BytesIO

# Load environment variables
def load_env_file():
    """Load environment variables from .env file"""
    code_snippets_path = Path(__file__).parent / "Code Snippets"
    env_file = code_snippets_path / ".env"
    
    if env_file.exists():
        with open(env_file, 'r') as f:
            for line in f:
                if '=' in line and not line.startswith('#'):
                    key, value = line.strip().split('=', 1)
                    value = value.strip("'\"")
                    os.environ[key] = value
        print("✅ Environment variables loaded from .env file")
    else:
        print("❌ .env file not found")

load_env_file()

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.messages import HumanMessage

class ComprehensiveRAGDemo:
    """
    Comprehensive RAG implementation with multiple approaches:
    1. Basic text RAG with sample data
    2. Document RAG with PDF/Word files
    3. Multi-modal RAG with text + images
    """

    def __init__(self):
        """Initialize comprehensive RAG system."""
        print("🚀 Initializing Comprehensive RAG Demo...")

        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY not found")

        # Initialize models
        self.embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        # Use the latest vision model from the documentation
        self.vision_llm = ChatOpenAI(model="gpt-4o", max_tokens=1024)

        # Text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

        self.vectorstore = None
        self.qa_chain = None
        self.processed_content = []

        print("✅ Comprehensive RAG Demo initialized!")
        print("🔧 Supports: Basic Text, Document Processing, Multi-Modal")

    # ==========================================
    # SECTION 1: BASIC TEXT RAG
    # ==========================================

    def demo_basic_text_rag(self):
        """Demonstrate basic RAG with sample text data."""
        print("\n" + "="*60)
        print("📚 SECTION 1: BASIC TEXT RAG DEMO")
        print("="*60)

        # Sample company documents
        sample_docs = [
            """
            Company Policy: Remote Work Guidelines

            Our company supports flexible remote work arrangements. Employees can work from home
            up to 3 days per week with manager approval. Remote work days must be scheduled in advance.

            Equipment: The company provides laptops and necessary software for remote work.
            Communication: Daily check-ins via Slack are required for remote workers.
            Productivity: Remote workers must maintain the same productivity standards as office workers.
            """,

            """
            Employee Benefits Overview

            Health Insurance: Full medical, dental, and vision coverage provided.
            Vacation Policy: 20 days of paid vacation per year, plus 10 sick days.
            Professional Development: $2000 annual budget for training and conferences.
            Retirement: 401k with 4% company matching.
            Wellness: Free gym membership and mental health support.
            """,

            """
            IT Security Guidelines

            Password Requirements: Minimum 12 characters with special characters.
            VPN: Required for all remote connections to company systems.
            Software Updates: Automatic updates must be enabled on all devices.
            Data Protection: No company data on personal devices without encryption.
            Incident Reporting: Security incidents must be reported within 1 hour.
            """,

            """
            Meeting Room Booking System

            Conference rooms can be booked through the company portal.
            Maximum booking duration: 4 hours per session.
            Cancellation: Must cancel at least 2 hours in advance.
            Equipment: All rooms have projectors, whiteboards, and video conferencing.
            Catering: Can be arranged through HR for meetings over 2 hours.
            """
        ]

        print(f"� Processing {len(sample_docs)} sample documents...")

        # Convert to Document objects
        documents = [Document(page_content=doc) for doc in sample_docs]

        # Create knowledge base
        self._create_basic_knowledge_base(documents)

        # Demo questions
        demo_questions = [
            "How many days can I work from home?",
            "What's our vacation policy?",
            "What are the password requirements?",
            "How do I book a meeting room?"
        ]

        print("\n🎪 Basic RAG Demo Questions:")
        for i, question in enumerate(demo_questions, 1):
            print(f"\n{i}. {question}")
            result = self.query(question)
            print(f"💡 Answer: {result['answer']}")
            print(f"📖 Sources: {len(result['source_documents'])} chunks")

        print("\n✅ Basic Text RAG demonstration complete!")

    def _create_basic_knowledge_base(self, documents):
        """Create knowledge base from basic documents."""
        # Split documents
        chunks = self.text_splitter.split_documents(documents)
        print(f"🔪 Created {len(chunks)} chunks")

        # Create vector store
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=None
        )

        # Create QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True
        )

        print("✅ Basic knowledge base created!")

    # ==========================================
    # SECTION 2: DOCUMENT RAG (PDF/WORD)
    # ==========================================

    def demo_document_rag(self):
        """Demonstrate document RAG with PDF and Word files."""
        print("\n" + "="*60)
        print("📄 SECTION 2: DOCUMENT RAG DEMO")
        print("="*60)

        # Load documents from folder
        documents = self.load_documents_from_folder()

        if not documents:
            print("❌ No documents found in EmbeddingDocs folder")
            return

        # Create knowledge base
        self.create_document_knowledge_base(documents)

        # Demo questions based on loaded documents
        print("\n🎪 Document RAG Demo Questions:")

        # Check what documents we have and ask relevant questions
        doc_names = [doc.metadata.get('source', '') for doc in documents]

        if any('attention' in name.lower() for name in doc_names):
            print("\n📄 Questions about 'Attention is All You Need' paper:")
            questions = [
                "What is the Transformer architecture?",
                "How does self-attention work?",
                "What are the advantages of Transformers over RNNs?"
            ]
            for q in questions:
                result = self.query(q)
                print(f"❓ {q}")
                print(f"💡 {result['answer'][:200]}...")
                print()

        if any('boomi' in name.lower() for name in doc_names):
            print("\n📄 Questions about Boomi document:")
            questions = [
                "What is Boomi used for?",
                "How does Boomi integration work?"
            ]
            for q in questions:
                result = self.query(q)
                print(f"❓ {q}")
                print(f"💡 {result['answer'][:200]}...")
                print()

        print("✅ Document RAG demonstration complete!")

    def load_pdf_document(self, file_path):
        """Load a PDF document (text only)."""
        try:
            from langchain_community.document_loaders import PyPDFLoader

            print(f"📄 Loading PDF: {file_path.name}")
            loader = PyPDFLoader(str(file_path))
            documents = loader.load()

            for doc in documents:
                doc.metadata.update({
                    "source": file_path.name,
                    "type": "PDF_Text",
                    "content_type": "text"
                })

            print(f"   ✅ Loaded {len(documents)} pages")
            return documents

        except Exception as e:
            print(f"❌ Error loading PDF: {e}")
            return []

    def load_word_document(self, file_path):
        """Load a Word document."""
        try:
            from langchain_community.document_loaders import Docx2txtLoader

            print(f"📄 Loading Word document: {file_path.name}")
            loader = Docx2txtLoader(str(file_path))
            documents = loader.load()

            for doc in documents:
                doc.metadata.update({
                    "source": file_path.name,
                    "type": "Word_Text",
                    "content_type": "text"
                })

            print(f"   ✅ Loaded Word document")
            return documents

        except Exception as e:
            print(f"❌ Error loading Word document: {e}")
            return []

    def create_document_knowledge_base(self, documents):
        """Create knowledge base from documents."""
        print(f"🔪 Processing {len(documents)} documents...")

        # Split documents
        chunks = self.text_splitter.split_documents(documents)
        print(f"📄 Created {len(chunks)} chunks")

        # Create vector store
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=None
        )

        # Create QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 5}),
            return_source_documents=True
        )

        print("✅ Document knowledge base created!")

    # ==========================================
    # SECTION 3: MULTI-MODAL RAG (TEXT + IMAGES)
    # ==========================================

    def extract_images_from_pdf(self, pdf_path):
        """Extract images from PDF and convert to base64."""
        try:
            import fitz  # PyMuPDF
            
            print(f"🖼️  Extracting images from: {pdf_path.name}")
            doc = fitz.open(pdf_path)
            images = []
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                image_list = page.get_images()
                
                for img_index, img in enumerate(image_list):
                    try:
                        # Get image data
                        xref = img[0]
                        pix = fitz.Pixmap(doc, xref)
                        
                        if pix.n - pix.alpha < 4:  # GRAY or RGB
                            # Convert to PNG bytes
                            img_data = pix.tobytes("png")
                            
                            # Convert to base64
                            img_base64 = base64.b64encode(img_data).decode()
                            
                            images.append({
                                "base64": img_base64,
                                "page": page_num + 1,
                                "index": img_index,
                                "source": pdf_path.name
                            })
                        
                        pix = None  # Free memory
                        
                    except Exception as e:
                        print(f"   ⚠️  Error extracting image {img_index} from page {page_num + 1}: {e}")
                        continue
            
            doc.close()
            print(f"   ✅ Extracted {len(images)} images")
            return images
            
        except ImportError:
            print("❌ PyMuPDF not installed. Install with: pip install PyMuPDF")
            return []
        except Exception as e:
            print(f"❌ Error extracting images from {pdf_path.name}: {e}")
            return []
    
    def describe_image(self, image_base64, source_info):
        """Generate description of image using GPT-4o Vision (latest format)."""
        try:
            print(f"   🔍 Analyzing image from {source_info}...")

            # Use the latest OpenAI format for vision
            message = HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": "Describe this image in detail. Focus on any text, diagrams, charts, figures, tables, or important visual elements that might be relevant for answering questions about the document. Include any mathematical formulas, architectural diagrams, or technical illustrations you see."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_base64}",
                            "detail": "high"  # Use high detail for better analysis
                        }
                    }
                ]
            )

            response = self.vision_llm.invoke([message])
            description = response.content

            print(f"   ✅ Generated description ({len(description)} chars)")
            return description

        except Exception as e:
            print(f"   ❌ Error describing image: {e}")
            return f"Image from {source_info} (description failed)"
    
    def load_pdf_with_images(self, file_path):
        """Load PDF with both text and images."""
        documents = []
        
        # Load text content
        try:
            from langchain_community.document_loaders import PyPDFLoader
            
            print(f"📄 Loading PDF text: {file_path.name}")
            loader = PyPDFLoader(str(file_path))
            text_docs = loader.load()
            
            for doc in text_docs:
                doc.metadata.update({
                    "source": file_path.name,
                    "type": "PDF_Text",
                    "content_type": "text"
                })
            
            documents.extend(text_docs)
            print(f"   ✅ Loaded {len(text_docs)} text pages")
            
        except Exception as e:
            print(f"❌ Error loading PDF text: {e}")
        
        # Extract and process images
        images = self.extract_images_from_pdf(file_path)
        
        for img in images:
            # Generate description
            description = self.describe_image(
                img["base64"], 
                f"{img['source']} (page {img['page']})"
            )
            
            # Create document for image description
            img_doc = Document(
                page_content=f"Image from page {img['page']}: {description}",
                metadata={
                    "source": file_path.name,
                    "type": "PDF_Image",
                    "content_type": "image",
                    "page": img["page"],
                    "image_base64": img["base64"]  # Store for potential display
                }
            )
            
            documents.append(img_doc)
        
        return documents
    
    def load_word_document(self, file_path):
        """Load Word document (text only for now)."""
        try:
            from langchain_community.document_loaders import Docx2txtLoader
            
            print(f"📄 Loading Word document: {file_path.name}")
            loader = Docx2txtLoader(str(file_path))
            documents = loader.load()
            
            for doc in documents:
                doc.metadata.update({
                    "source": file_path.name,
                    "type": "Word_Text",
                    "content_type": "text"
                })
            
            print(f"   ✅ Loaded Word document")
            return documents
            
        except Exception as e:
            print(f"❌ Error loading Word document: {e}")
            return []
    
    def load_documents_from_folder(self, folder_path="EmbeddingDocs"):
        """Load all documents with multi-modal processing."""
        folder = Path(folder_path)
        
        if not folder.exists():
            print(f"❌ Folder {folder_path} not found")
            return []
        
        print(f"📁 Loading documents with multi-modal processing from: {folder}")
        all_documents = []
        
        # Find files
        pdf_files = list(folder.glob("*.pdf"))
        word_files = list(folder.glob("*.docx")) + list(folder.glob("*.doc"))
        
        print(f"   Found {len(pdf_files)} PDF files and {len(word_files)} Word files")
        
        # Process PDF files with images
        for pdf_file in pdf_files:
            docs = self.load_pdf_with_images(pdf_file)
            all_documents.extend(docs)
        
        # Process Word files
        for word_file in word_files:
            docs = self.load_word_document(word_file)
            all_documents.extend(docs)
        
        self.processed_content = all_documents
        print(f"📚 Total content pieces loaded: {len(all_documents)}")
        
        # Show content breakdown
        content_types = {}
        for doc in all_documents:
            content_type = doc.metadata.get('type', 'Unknown')
            content_types[content_type] = content_types.get(content_type, 0) + 1
        
        print("📊 Content breakdown:")
        for content_type, count in content_types.items():
            print(f"   {content_type}: {count}")
        
        return all_documents
    
    def create_knowledge_base(self, documents=None):
        """Create vector store from multi-modal documents."""
        if documents is None:
            documents = self.processed_content
        
        if not documents:
            print("❌ No documents to process")
            return
        
        print(f"🔪 Processing {len(documents)} content pieces...")
        
        # Split documents into chunks
        chunks = self.text_splitter.split_documents(documents)
        print(f"📄 Created {len(chunks)} chunks")
        
        # Create vector store
        print("🧮 Creating embeddings for multi-modal content...")
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=None
        )
        
        # Create QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 5}),
            return_source_documents=True
        )
        
        print("✅ Multi-modal knowledge base created!")
    
    def query(self, question):
        """Ask a question using multi-modal RAG."""
        if self.qa_chain is None:
            raise ValueError("No knowledge base created!")
        
        print(f"\n❓ Question: {question}")
        print("🔍 Searching multi-modal knowledge base...")
        
        result = self.qa_chain.invoke({"query": question})
        
        return {
            "answer": result["result"],
            "source_documents": result["source_documents"]
        }
    
    def interactive_demo(self):
        """Run interactive multi-modal demo."""
        print("\n" + "="*60)
        print("🎯 MULTI-MODAL RAG DEMO (TEXT + IMAGES)")
        print("="*60)
        
        # Load documents
        documents = self.load_documents_from_folder()
        
        if not documents:
            print("❌ No documents loaded.")
            return
        
        # Create knowledge base
        self.create_knowledge_base(documents)
        
        print("\n💡 Try questions about both text and visual content:")
        print("   - What is shown in the diagrams?")
        print("   - Describe any charts or figures")
        print("   - What does the architecture look like?")
        print("   - Any visual elements in the documents?")
        
        print("\n🎪 Ask your questions (type 'quit' to exit):")
        
        while True:
            question = input("\n❓ Your question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                break
            
            if not question:
                continue
            
            try:
                result = self.query(question)
                print(f"\n💡 Answer: {result['answer']}")
                
                # Show sources with content type
                sources = result['source_documents']
                print(f"\n📖 Sources ({len(sources)} pieces):")
                for i, doc in enumerate(sources, 1):
                    source = doc.metadata.get('source', 'Unknown')
                    content_type = doc.metadata.get('type', 'Unknown')
                    snippet = doc.page_content[:100].replace('\n', ' ').strip()
                    print(f"   {i}. [{content_type}] {source}: {snippet}...")
                
            except Exception as e:
                print(f"❌ Error: {e}")

    def run_comprehensive_demo(self):
        """Run all three RAG demonstrations in sequence."""
        print("🌟 Welcome to the Comprehensive RAG Demo!")
        print("This demo shows three different RAG approaches:")
        print("1. 📚 Basic Text RAG with sample data")
        print("2. 📄 Document RAG with PDF/Word files")
        print("3. 🖼️ Multi-Modal RAG with text + images")

        choice = input("\nWhich demo would you like to run? (1/2/3/all): ").strip().lower()

        if choice in ['1', 'all']:
            self.demo_basic_text_rag()
            if choice == 'all':
                input("\nPress Enter to continue to Document RAG...")

        if choice in ['2', 'all']:
            self.demo_document_rag()
            if choice == 'all':
                input("\nPress Enter to continue to Multi-Modal RAG...")

        if choice in ['3', 'all']:
            self.demo_multimodal_rag()

        print("\n🎓 COMPREHENSIVE RAG DEMO COMPLETE!")
        print("="*50)
        print("✅ You've seen three different RAG approaches")
        print("✅ Basic text embedding and retrieval")
        print("✅ Document processing (PDF/Word)")
        print("✅ Multi-modal capabilities (text + images)")
        print("✅ Perfect for intern training and demonstrations!")

    def demo_multimodal_rag(self):
        """Demonstrate multi-modal RAG with images."""
        print("\n" + "="*60)
        print("🖼️ SECTION 3: MULTI-MODAL RAG DEMO")
        print("="*60)

        # Load documents with images
        documents = self.load_documents_from_folder()

        if not documents:
            print("❌ No documents found in EmbeddingDocs folder")
            return

        # Create multi-modal knowledge base
        self.create_knowledge_base(documents)

        # Demo questions focusing on visual content
        print("\n🎪 Multi-Modal RAG Demo Questions:")

        visual_questions = [
            "What diagrams or figures are shown in the documents?",
            "Describe any architectural illustrations or charts",
            "What visual elements help explain the concepts?",
            "Are there any mathematical formulas or equations shown?"
        ]

        for question in visual_questions:
            try:
                result = self.query(question)
                print(f"\n❓ {question}")
                print(f"💡 Answer: {result['answer']}")

                # Show if any image sources were used
                sources = result['source_documents']
                image_sources = [s for s in sources if s.metadata.get('type', '').endswith('_Image')]
                if image_sources:
                    print(f"🖼️ Used {len(image_sources)} image descriptions in answer")

            except Exception as e:
                print(f"❌ Error: {e}")

        print("\n✅ Multi-Modal RAG demonstration complete!")

    def query(self, question):
        """Ask a question using the current RAG setup."""
        if self.qa_chain is None:
            raise ValueError("No knowledge base created!")

        result = self.qa_chain.invoke({"query": question})

        return {
            "answer": result["result"],
            "source_documents": result["source_documents"]
        }

    def load_documents_from_folder(self, folder_path="EmbeddingDocs"):
        """Load all documents with multi-modal processing."""
        folder = Path(folder_path)

        if not folder.exists():
            print(f"❌ Folder {folder_path} not found")
            return []

        print(f"📁 Loading documents with multi-modal processing from: {folder}")
        all_documents = []

        # Find files
        pdf_files = list(folder.glob("*.pdf"))
        word_files = list(folder.glob("*.docx")) + list(folder.glob("*.doc"))

        print(f"   Found {len(pdf_files)} PDF files and {len(word_files)} Word files")

        # Process PDF files with images
        for pdf_file in pdf_files:
            docs = self.load_pdf_with_images(pdf_file)
            all_documents.extend(docs)

        # Process Word files
        for word_file in word_files:
            docs = self.load_word_document(word_file)
            all_documents.extend(docs)

        self.processed_content = all_documents
        print(f"📚 Total content pieces loaded: {len(all_documents)}")

        # Show content breakdown
        content_types = {}
        for doc in all_documents:
            content_type = doc.metadata.get('type', 'Unknown')
            content_types[content_type] = content_types.get(content_type, 0) + 1

        print("📊 Content breakdown:")
        for content_type, count in content_types.items():
            print(f"   {content_type}: {count}")

        return all_documents

    def create_knowledge_base(self, documents=None):
        """Create vector store from multi-modal documents."""
        if documents is None:
            documents = self.processed_content

        if not documents:
            print("❌ No documents to process")
            return

        print(f"🔪 Processing {len(documents)} content pieces...")

        # Split documents into chunks
        chunks = self.text_splitter.split_documents(documents)
        print(f"📄 Created {len(chunks)} chunks")

        # Create vector store
        print("🧮 Creating embeddings for multi-modal content...")
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=None
        )

        # Create QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 5}),
            return_source_documents=True
        )

        print("✅ Multi-modal knowledge base created!")


def main():
    """Main function."""
    print("🌟 Welcome to the Multi-Modal RAG Demo!")
    print("This demo processes both TEXT and IMAGES from your documents!")
    
    try:
        demo = ComprehensiveRAGDemo()
        demo.run_comprehensive_demo()
        
        print("\n🎓 MULTI-MODAL RAG CAPABILITIES DEMONSTRATED:")
        print("✅ Text extraction and processing")
        print("✅ Image extraction from PDFs")
        print("✅ AI-powered image description")
        print("✅ Combined text + image search")
        print("✅ Source attribution for both content types")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Install PyMuPDF for image extraction: pip install PyMuPDF")


if __name__ == "__main__":
    main()
