from dotenv import load_dotenv

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import ReadTheDocsLoader
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore


load_dotenv()

def ingest_docs():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    loader = ReadTheDocsLoader(path = "langchain-docs/api.python.langchain.com/en/latest", encoding="utf-8")
    #loader = ReadTheDocsLoader("test", encoding="utf-8")
    try:
        raw_documents = loader.load()
    except Exception as e:
        print(f"Error loading documents: {e}")
        return
    print(f"Loaded {len(raw_documents)} documents")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)
    documents = text_splitter.split_documents(raw_documents)
    for doc in documents:
        #print(f"doc: {doc}" + '\n')
        new_url = doc.metadata["source"]
        new_url =new_url.replace("langchain.com/docs/", "https:/")
        doc.metadata.update({"source": new_url})
    print(f"Adding to Pinecone {len(documents)} documents")
    batch_size = 100
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        PineconeVectorStore.from_documents(
            batch,
            embedding=embeddings,
            index_name="langchain-doc-index",
        )

def ingest_docs_2() -> None:
    from langchain_community.document_loaders import FireCrawlLoader

    langchain_docs_base_urls= [
        "https://python.langchain.com/docs/integrations/chat/",
        "https://python.langchain.com/docs/integrations/llms/",
        "https://python.langchain.com/docs/integrations/memory/",
        "https://python.langchain.com/docs/integrations/stores/",
        "https://python.langchain.com/docs/integrations/text_embedding/",
        "https://python.langchain.com/docs/integrations/document_loaders/",
        "https://python.langchain.com/docs/integrations/document_transformers/",
        "https://python.langchain.com/docs/integrations/verctorstores/",
        "https://python.langchain.com/docs/integrations/retrievers/",
        "https://python.langchain.com/docs/integrations/tools/",
        "https://python.langchain.com/docs/integrations/llm_caching/",
        "https://python.langchain.com/docs/integrations/graphs/",
        "https://python.langchain.com/docs/integrations/callbacks/",
        "https://python.langchain.com/docs/integrations/chat_loaders/",
        "https://python.langchain.com/docs/integrations/concepts/"
    ]
    for url in langchain_docs_base_urls:
        print(f"firecrawling: {url=}")
        loader = FireCrawlLoader(
            url=url,
            mode="crawl",
            params = {
                #"crawlerOptions": {"limit": 5 },
                #"pageOptions": {"onlyMainContent": True},
                #"wait_until_done": True
            },
        )
        docs = loader.load()
        print(f"Going to add {len(docs)} documents to Pinecone")
        print(f"Document loaded: {url} to vectorstore")
        PineconeVectorStore.from_documents(
            docs,
            embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
            index_name="firecrawl-index",
        )

if __name__ == "__main__":
    print("Loading documents...")
    #ingest_docs()
    ingest_docs_2()
    #llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)


