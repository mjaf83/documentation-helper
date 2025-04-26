from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
#from langchain_ollama import ChatOllama
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()

INDEX_NAME = "langchain-doc-index"

def run_llm(query: str):
    embeddings = OpenAIEmbeddings(model ="text-embedding-3-small")
    docsearch = PineconeVectorStore(
        index_name=INDEX_NAME,
        embedding = embeddings
    )  
    chat = ChatOpenAI(
        model="gpt-4.1-nano",
        temperature=0, 
        verbose=True
    )

    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    stuff_documents_chain = create_stuff_documents_chain(chat, retrieval_qa_chat_prompt)
    qa = create_retrieval_chain(
        retriever=docsearch.as_retriever(),
        combine_docs_chain=stuff_documents_chain
    )

    result = qa.invoke(input = {"input": query})
    return result



if __name__ == "__main__":
    res = run_llm("What is a Lanchain chain?")
    print(res["answer"])
