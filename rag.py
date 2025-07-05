from uuid import uuid4
from dotenv import load_dotenv
from pathlib import Path
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQAWithSourcesChain
from prompt import prompt, example_prompt
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
load_dotenv()

CHUNK_SIZE = 1000
EMBEDDING_MODEL = "Alibaba-NLP/gte-base-en-v1.5"
VECTORSTORE_DIR = Path(__file__).parent / 'resources/vectorstore'
COLLECTION_NAME = 'real_estate'

llm = None
vector_store = None


def initialize_components():
    global llm, vector_store

    if llm is None:
        llm = ChatGroq(model='llama3-8b-8192', temperature=0.9, max_tokens=500)

    if vector_store is None:
        ef = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={'trust_remote_code': True}
        )
        # Create directory if it doesn't exist
        VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)

        vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=ef,
            persist_directory=str(VECTORSTORE_DIR)
        )


def process_urls(input_urls):
    global vector_store
    yield 'initialize component'
    initialize_components()

    # Handle collection deletion and recreation
    try:
        vector_store.delete_collection()
    except:
        pass  # Collection might not exist yet

    # Recreate the vector store after deletion

    ef = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'trust_remote_code': True}
    )
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=ef,
        persist_directory=str(VECTORSTORE_DIR)
    )

    yield  "loading data"
    loader = UnstructuredURLLoader(
        urls=input_urls,
        headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
    )
    data = loader.load()

    yield  "spliting the data"
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", ' '],
        chunk_size=CHUNK_SIZE
    )
    docs = text_splitter.split_documents(data)

    yield  "adding document to vectordb"
    uuids = [str(uuid4()) for _ in range(len(docs))]
    vector_store.add_documents(documents=docs, ids=uuids)

    yield  "done adding data into vector database"
def generate_answer(query):
    global vector_store
    if vector_store is None:
        raise RuntimeError("You must process the URLs first before asking a question.")

    # Get top 4 relevant docs from vector store
    docs = vector_store.similarity_search(query, k=4)

    # Load chain with custom prompt
    chain = load_qa_with_sources_chain(
        llm=llm,
        prompt=prompt,
        document_prompt=example_prompt
    )

    result = chain.invoke({
        "input_documents": docs,
        "question": query
    })

    answer = result.get("answer") or result.get("output_text") or "No answer found."
    sources = result.get("sources", "")
    return answer, sources



if __name__ == "__main__":
    urls = [
        "https://www.cnbc.com/2024/12/21/how-the-federal-reserves-rate-policy-affects-mortgages.html",
        "https://www.cnbc.com/2024/12/20/why-mortgage-rates-jumped-despite-fed-interest-rate-cut.html"
    ]
    process_urls(urls)
    answer, sources = generate_answer("tell me what was the 30year fixed mortage rate along with the date?")
    print(f'answers: {answer}')
    print(f'sources: {sources}')