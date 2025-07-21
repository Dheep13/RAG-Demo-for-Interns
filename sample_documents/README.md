# 📄 Sample Documents

This folder contains sample documents for the RAG demo. You can add your own PDF and Word documents here for testing.

## 📁 Supported File Types

- **PDF files** (`.pdf`) - Processed with PyPDF
- **Word documents** (`.docx`, `.doc`) - Processed with docx2txt

## 🎯 How to Use

1. **Add your documents** to this folder
2. **Run the demos** - they will automatically detect and process files here
3. **Ask questions** about your document content

## 📋 Sample Documents Included

The demo works with any documents you add here. For testing purposes, you might want to include:

- Company policies
- Technical documentation
- Research papers
- User manuals
- Any text-heavy documents

## 🔍 What Happens During Processing

### **Text Extraction:**
- PDF: Extracts text from each page
- Word: Extracts all text content
- Creates metadata for source tracking

### **Chunking:**
- Splits documents into 1000-character chunks
- 200-character overlap between chunks
- Preserves context across chunk boundaries

### **Embedding:**
- Each chunk becomes a 1536-dimensional vector
- Uses OpenAI's text-embedding-ada-002 model
- Enables semantic similarity search

### **Multi-Modal (PDFs only):**
- Extracts images from PDF pages
- Uses GPT-4o Vision to describe images
- Creates searchable descriptions of visual content

## 💡 Tips for Best Results

### **Document Quality:**
- Use clear, well-formatted documents
- Avoid heavily image-based PDFs (unless using multi-modal)
- Ensure text is selectable in PDFs

### **Document Size:**
- Start with smaller documents (< 50 pages) for testing
- Larger documents work but take longer to process
- Consider splitting very large documents

### **Content Types:**
- **Great**: Technical docs, policies, manuals, research papers
- **Good**: Reports, presentations (if text-heavy)
- **Limited**: Image-heavy documents, scanned PDFs

## 🧪 Testing Your Documents

After adding documents, test with questions like:

### **General Questions:**
- "What is this document about?"
- "Summarize the main points"
- "What are the key requirements?"

### **Specific Questions:**
- "How do I configure X?"
- "What are the security guidelines?"
- "What is the process for Y?"

### **Multi-Modal Questions (for PDFs with images):**
- "What diagrams are shown?"
- "Describe the architecture"
- "What do the charts illustrate?"

## 🔧 Troubleshooting

### **Document Not Loading:**
- Check file format (PDF/DOCX supported)
- Ensure file is not corrupted
- Verify file permissions

### **Poor Search Results:**
- Try more specific questions
- Use keywords from the document
- Check if document text is extractable

### **Processing Errors:**
- Check document size (very large files may timeout)
- Ensure document is not password-protected
- Try with a simpler document first

## 📊 Processing Statistics

After processing, you'll see:
- Number of pages/chunks created
- Content type breakdown
- Source attribution
- Embedding statistics

## 🎓 Educational Value

Using your own documents helps interns understand:
- How RAG works with real-world content
- The importance of document quality
- How chunking affects retrieval
- The relationship between questions and results

## 🔄 Updating Documents

To add new documents:
1. Copy files to this folder
2. Restart the demo
3. Documents will be automatically reprocessed

To remove documents:
1. Delete files from this folder
2. Restart the demo
3. Vector database will be rebuilt

---

**Happy document processing! 📚**

*The quality of your RAG system depends heavily on the quality and relevance of your documents.*
