"""
Enhanced RAG Demo with Embedding Visualization
==============================================

This enhanced version includes:
1. Embedding visualization and analysis
2. Direct vector database querying
3. Similarity score explanations
4. Database statistics
5. Interactive exploration features

Perfect for intern training to understand how RAG works under the hood!
"""

import os
import sys
from pathlib import Path
import base64
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load environment variables
def load_env_file():
    """Load environment variables from .env file"""
    code_snippets_path = Path("Code Snippets")
    env_file = code_snippets_path / ".env"
    
    if env_file.exists():
        with open(env_file, 'r') as f:
            for line in f:
                if '=' in line and not line.startswith('#'):
                    key, value = line.strip().split('=', 1)
                    value = value.strip("'\"")
                    os.environ[key] = value
        print("✅ Environment variables loaded")
        return True
    else:
        print("❌ .env file not found")
        return False

load_env_file()

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.messages import HumanMessage

class EnhancedRAGDemo:
    """
    Enhanced RAG implementation with embedding visualization and database analysis.
    """
    
    def __init__(self):
        """Initialize enhanced RAG system."""
        print("🚀 Initializing Enhanced RAG Demo with Embedding Analysis...")
        
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY not found")
        
        # Initialize models
        self.embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
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
        
        print("✅ Enhanced RAG Demo initialized!")
        print("🔧 Features: Embedding Analysis, Direct DB Queries, Similarity Scores")
    
    def show_embedding_details(self, sample_texts):
        """Show detailed embedding analysis for educational purposes."""
        print("\n🔍 EMBEDDING ANALYSIS:")
        print("=" * 60)
        
        embeddings_data = []
        
        for i, text in enumerate(sample_texts, 1):
            print(f"\n📄 Text {i}: '{text[:100]}...'")
            
            # Generate embedding
            embedding = self.embeddings.embed_query(text)
            embeddings_data.append(embedding)
            
            print(f"   📊 Embedding dimensions: {len(embedding)}")
            print(f"   📈 Value range: [{min(embedding):.4f}, {max(embedding):.4f}]")
            print(f"   🔢 First 10 values: {[round(x, 4) for x in embedding[:10]]}")
            print(f"   📏 Vector magnitude: {np.linalg.norm(embedding):.4f}")
        
        # Calculate similarity matrix
        if len(embeddings_data) > 1:
            print(f"\n🔗 COSINE SIMILARITY MATRIX:")
            print("-" * 50)
            
            similarity_matrix = cosine_similarity(embeddings_data)
            
            for i in range(len(sample_texts)):
                for j in range(i + 1, len(sample_texts)):
                    similarity = similarity_matrix[i][j]
                    print(f"Text {i+1} ↔ Text {j+1}: {similarity:.4f}")
                    if similarity > 0.8:
                        print("   ✅ Very similar!")
                    elif similarity > 0.6:
                        print("   ⚠️ Moderately similar")
                    else:
                        print("   ❌ Different topics")
        
        print("\n💡 Key Insights:")
        print("   • Higher cosine similarity = more semantically similar")
        print("   • RAG uses this similarity to find relevant chunks")
        print("   • Each dimension captures different semantic features")
    
    def query_database_directly(self, query_text, k=3, show_details=True):
        """Query the vector database directly with detailed analysis."""
        if self.vectorstore is None:
            print("❌ No vector database available. Create knowledge base first.")
            return None
        
        if show_details:
            print(f"\n🔍 DIRECT DATABASE QUERY: '{query_text}'")
            print("=" * 60)
            
            # Show query embedding details
            query_embedding = self.embeddings.embed_query(query_text)
            print(f"📊 Query embedding dimensions: {len(query_embedding)}")
            print(f"📈 Query embedding range: [{min(query_embedding):.4f}, {max(query_embedding):.4f}]")
            print(f"🔢 Query embedding preview: {[round(x, 4) for x in query_embedding[:5]]}...")
        
        # Search with similarity scores
        try:
            results = self.vectorstore.similarity_search_with_score(query_text, k=k)
            
            if show_details:
                print(f"\n📊 Top {k} Similar Chunks:")
                print("-" * 50)
                
                for i, (doc, score) in enumerate(results, 1):
                    print(f"\n{i}. 🎯 Similarity Score: {score:.4f}")
                    
                    # Interpret score
                    if score < 0.5:
                        interpretation = "🟢 Excellent match"
                    elif score < 1.0:
                        interpretation = "🟡 Good match"
                    elif score < 1.5:
                        interpretation = "🟠 Moderate match"
                    else:
                        interpretation = "🔴 Poor match"
                    
                    print(f"   {interpretation}")
                    print(f"   📄 Source: {doc.metadata.get('source', 'Unknown')}")
                    print(f"   📝 Type: {doc.metadata.get('type', 'Unknown')}")
                    print(f"   📖 Content: {doc.page_content[:150]}...")
                
                print(f"\n💡 Similarity Score Guide:")
                print("   🟢 0.0-0.5: Excellent semantic match")
                print("   🟡 0.5-1.0: Good semantic match")
                print("   🟠 1.0-1.5: Moderate semantic match")
                print("   🔴 1.5+: Poor semantic match")
            
            return results
            
        except Exception as e:
            print(f"❌ Error querying database: {e}")
            return None
    
    def show_database_statistics(self):
        """Show comprehensive database statistics."""
        if self.vectorstore is None:
            print("❌ No vector database available.")
            return
        
        print("\n📊 VECTOR DATABASE STATISTICS:")
        print("=" * 50)
        
        try:
            # Get collection info
            collection = self.vectorstore._collection
            count = collection.count()
            
            print(f"📈 Total vectors stored: {count}")
            print(f"🤖 Embedding model: {self.embeddings.model}")
            print(f"📏 Vector dimensions: 1536 (OpenAI ada-002)")
            print(f"💾 Database type: ChromaDB (in-memory)")
            print(f"🔪 Chunk size: {self.text_splitter._chunk_size} characters")
            print(f"🔗 Chunk overlap: {self.text_splitter._chunk_overlap} characters")
            
            # Analyze content by type
            if count > 0:
                sample_docs = self.vectorstore.similarity_search("sample", k=min(20, count))
                
                content_types = {}
                sources = set()
                total_chars = 0
                
                for doc in sample_docs:
                    content_type = doc.metadata.get('type', 'Unknown')
                    source = doc.metadata.get('source', 'Unknown')
                    
                    content_types[content_type] = content_types.get(content_type, 0) + 1
                    sources.add(source)
                    total_chars += len(doc.page_content)
                
                print(f"\n📄 Content Analysis (sample of {len(sample_docs)} chunks):")
                for content_type, count in content_types.items():
                    print(f"   {content_type}: {count} chunks")
                
                print(f"\n📚 Sources: {len(sources)} unique documents")
                for source in sorted(sources):
                    print(f"   📖 {source}")
                
                avg_chunk_size = total_chars / len(sample_docs)
                print(f"\n📏 Average chunk size: {avg_chunk_size:.0f} characters")
                
        except Exception as e:
            print(f"❌ Error getting database stats: {e}")
    
    def compare_query_similarities(self, queries):
        """Compare how different queries perform against the database."""
        print("\n🧪 QUERY SIMILARITY COMPARISON:")
        print("=" * 60)
        
        for i, query in enumerate(queries, 1):
            print(f"\n{i}. Query: '{query}'")
            results = self.query_database_directly(query, k=1, show_details=False)
            
            if results:
                best_score = results[0][1]
                source = results[0][0].metadata.get('source', 'Unknown')
                content_preview = results[0][0].page_content[:100]
                
                print(f"   🎯 Best score: {best_score:.4f}")
                print(f"   📄 From: {source}")
                print(f"   📖 Content: {content_preview}...")
                
                # Quality assessment
                if best_score < 0.5:
                    print("   ✅ Excellent - will generate accurate answer")
                elif best_score < 1.0:
                    print("   ⚠️ Good - should generate relevant answer")
                elif best_score < 1.5:
                    print("   🟠 Moderate - may generate generic answer")
                else:
                    print("   ❌ Poor - likely to hallucinate or give irrelevant answer")
    
    def demo_embedding_concepts(self):
        """Interactive demo of embedding concepts."""
        print("\n🎓 EMBEDDING CONCEPTS DEMONSTRATION:")
        print("=" * 60)
        
        # Test different types of text similarity
        test_cases = [
            {
                "title": "Semantic Similarity",
                "texts": [
                    "Employees can work from home",
                    "Staff are allowed to work remotely",
                    "The sky is blue today"
                ]
            },
            {
                "title": "Technical Similarity", 
                "texts": [
                    "Neural networks use backpropagation",
                    "Deep learning models train with gradient descent",
                    "I like pizza for dinner"
                ]
            }
        ]
        
        for case in test_cases:
            print(f"\n📚 {case['title']} Test:")
            print("-" * 30)
            self.show_embedding_details(case['texts'])
    
    def interactive_exploration(self):
        """Interactive exploration of the RAG system."""
        print("\n🎮 INTERACTIVE RAG EXPLORATION:")
        print("=" * 50)
        print("Commands:")
        print("  'stats' - Show database statistics")
        print("  'query <text>' - Query database directly")
        print("  'compare' - Compare multiple queries")
        print("  'embeddings' - Show embedding concepts")
        print("  'quit' - Exit")
        
        while True:
            command = input("\n🎯 Enter command: ").strip().lower()
            
            if command == 'quit':
                print("👋 Thanks for exploring RAG!")
                break
            elif command == 'stats':
                self.show_database_statistics()
            elif command.startswith('query '):
                query_text = command[6:]
                self.query_database_directly(query_text)
            elif command == 'compare':
                queries = [
                    "remote work policy",
                    "vacation days",
                    "password requirements",
                    "completely unrelated topic"
                ]
                self.compare_query_similarities(queries)
            elif command == 'embeddings':
                self.demo_embedding_concepts()
            else:
                print("❌ Unknown command. Try 'stats', 'query <text>', 'compare', 'embeddings', or 'quit'")


def main():
    """Main demo function."""
    print("🌟 Enhanced RAG Demo with Embedding Visualization!")
    print("This demo shows you exactly how RAG works under the hood.")
    
    try:
        # Initialize demo
        demo = EnhancedRAGDemo()
        
        # Create a simple knowledge base for demonstration
        sample_docs = [
            Document(page_content="""
            Company Remote Work Policy: Employees can work from home up to 3 days per week 
            with manager approval. Remote work days must be scheduled in advance.
            Equipment is provided by the company including laptops and software.
            """),
            Document(page_content="""
            Employee Benefits: Health insurance covers medical, dental, and vision.
            Vacation policy provides 20 days paid time off plus 10 sick days annually.
            Professional development budget is $2000 per employee per year.
            """),
            Document(page_content="""
            IT Security Requirements: Passwords must be minimum 12 characters with special characters.
            VPN is required for all remote connections. Software updates must be automatic.
            Security incidents must be reported within 1 hour of discovery.
            """)
        ]
        
        print("\n📚 Creating sample knowledge base...")
        
        # Create vector store
        chunks = demo.text_splitter.split_documents(sample_docs)
        demo.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=demo.embeddings,
            persist_directory=None
        )
        
        # Create QA chain
        demo.qa_chain = RetrievalQA.from_chain_type(
            llm=demo.llm,
            chain_type="stuff",
            retriever=demo.vectorstore.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True
        )
        
        print("✅ Knowledge base created!")
        
        # Show embedding analysis
        sample_texts = [chunk.page_content for chunk in chunks[:3]]
        demo.show_embedding_details(sample_texts)
        
        # Show database statistics
        demo.show_database_statistics()
        
        # Demo query comparison
        demo.compare_query_similarities([
            "remote work policy",
            "vacation benefits", 
            "password security",
            "unrelated topic"
        ])
        
        # Interactive exploration
        demo.interactive_exploration()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Make sure your OpenAI API key is set in the .env file")


if __name__ == "__main__":
    main()
