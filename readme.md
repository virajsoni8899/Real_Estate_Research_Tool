# 🏠 Real Estate RAG Tool

A powerful **Retrieval-Augmented Generation (RAG)** system built with LangChain for real estate market analysis and mortgage rate insights.

## 🚀 Features

- **Web Content Processing**: Automatically scrapes and processes real estate articles from various sources
- **Intelligent Document Chunking**: Splits large documents into manageable chunks for better retrieval
- **Vector Database Storage**: Uses ChromaDB for efficient similarity search and document retrieval
- **AI-Powered Q&A**: Leverages Groq's LLaMA model for generating accurate answers with source citations
- **Source Attribution**: Provides proper citations for all generated answers

## 📋 Prerequisites

- Python 3.8+
- Groq API key
- Internet connection for web scraping

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone <your-repository-url>
   cd real-estate-rag-tool
   ```

2. **Install required packages**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   Create a `.env` file in the project root:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   ```

## 📦 Dependencies

```
langchain
langchain-community
langchain-chroma
langchain-groq
langchain-huggingface
chromadb
python-dotenv
unstructured
```

## 🏗️ Project Structure

```
real-estate-rag-tool/
├── rag.py                 # Main RAG implementation
├── requirements.txt       # Python dependencies
├── .env                  # Environment variables (create this)
├── resources/
│   └── vectorstore/      # ChromaDB storage (auto-created)
└── README.md            # This file
```

## 🔧 Configuration

### Key Parameters

- **CHUNK_SIZE**: `1000` - Size of document chunks for processing
- **EMBEDDING_MODEL**: `"Alibaba-NLP/gte-base-en-v1.5"` - HuggingFace embedding model
- **LLM_MODEL**: `"llama3-8b-8192"` - Groq language model
- **COLLECTION_NAME**: `"real_estate"` - ChromaDB collection name

### Customization

You can modify these parameters in `rag.py`:

```python
CHUNK_SIZE = 800  # Reduce for shorter chunks
EMBEDDING_MODEL = "your-preferred-embedding-model"
```

## 🚀 Usage

### Basic Usage

```python
from rag import process_urls, generate_answer

# Process real estate articles
urls = [
    "https://www.example.com/mortgage-rates-article",
    "https://www.example.com/housing-market-analysis"
]

# Build knowledge base
process_urls(urls)

# Ask questions
answer, sources = generate_answer("What are the current 30-year mortgage rates?")
print(f"Answer: {answer}")
print(f"Sources: {sources}")
```

### Running the Script

```bash
python rag.py
```

## 📊 Example Output

```
initialize component
load data
splitting text
add doc to vectordb
answers: The 30-year fixed mortgage rate was 6.72% for the week ending Dec. 19.
sources: https://www.cnbc.com/2024/12/20/why-mortgage-rates-jumped-despite-fed-interest-rate-cut.html
```

## 🔍 How It Works

1. **Data Ingestion**: Web articles are scraped using `UnstructuredURLLoader`
2. **Text Processing**: Documents are split into chunks using `RecursiveCharacterTextSplitter`
3. **Embedding Creation**: Text chunks are converted to embeddings using HuggingFace models
4. **Vector Storage**: Embeddings are stored in ChromaDB for fast retrieval
5. **Query Processing**: User queries are embedded and matched against stored documents
6. **Answer Generation**: Groq's LLaMA model generates answers based on retrieved context

## 🎯 Use Cases

- **Mortgage Rate Analysis**: Get current rates and trends
- **Housing Market Insights**: Analyze market conditions and predictions
- **Real Estate Research**: Quick answers from multiple sources
- **Investment Decisions**: Data-driven real estate investment insights

## ⚠️ Troubleshooting

### Common Issues

1. **Access Denied Errors**
   - Some websites block automated requests
   - Try adding user-agent headers or use alternative URLs

2. **Token Length Warnings**
   - Reduce `CHUNK_SIZE` to 800 or lower
   - This optimizes embedding model performance

3. **Model Deprecation**
   - Update to latest supported Groq models
   - Check [Groq documentation](https://console.groq.com/docs/models)

### Performance Tips

- Use smaller chunk sizes for better embedding quality
- Process fewer URLs at once for faster initialization
- Consider using local documents for consistent access

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **LangChain** for the RAG framework
- **Groq** for fast LLM inference
- **ChromaDB** for vector storage
- **HuggingFace** for embedding models

## 📞 Support

For questions or issues:
- Open an issue on GitHub
- Check the [troubleshooting section](#-troubleshooting)
- Review the [LangChain documentation](https://python.langchain.com/)

---

**Made with ❤️ for Real Estate Intelligence**