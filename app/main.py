import os
from dotenv import load_dotenv
from redis import Redis as RedisClient
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Redis
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")


def connect_to_redis(url="redis://localhost:6379"):
    return RedisClient.from_url(url)


def clear_index(client, index_name):
    try:
        client.ft(index_name).dropindex(delete_documents=True)
        print("Önceki index silindi.")
    except Exception as e:
        print(f"Index yok ya da silinemedi: {e}")


def load_pdf(path):
    loader = PyPDFLoader(path)
    return loader.load()


def split_documents(docs, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n", ".", " ", ""]
    )
    return splitter.split_documents(docs)


def create_vectorstore(docs, embedding_model, redis_url, index_name):
    return Redis.from_documents(
        documents=docs,
        embedding=embedding_model,
        redis_url=redis_url,
        index_name=index_name
    )


def build_rag_chain(vectorstore, groq_api_key):
    retriever = vectorstore.as_retriever()

    llm = ChatGroq(
        model="deepseek-r1-distill-llama-70b",
        temperature=0,
        api_key=groq_api_key
    )

    system_prompt = (
        "Use the given context to answer the question. "
        "If you don't know the answer, say you don't know. "
        "Context: {context}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, question_answer_chain)


if __name__ == "__main__":
    pdf_name = "my_pdf"
    redis_url = "redis://localhost:6379"
    index_name = f"{pdf_name}_index"
    pdf_path = rf"..\data\{pdf_name}.pdf"

    client = connect_to_redis(redis_url)
    clear_index(client, index_name)

    docs = load_pdf(pdf_path)
    chunks = split_documents(docs)

    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vectorstore = create_vectorstore(chunks, embedding_model, redis_url, index_name)

    chain = build_rag_chain(vectorstore, groq_api_key)
    result = chain.invoke({"input": "Umut Şahin kız arkadaşının ismi ne?"})

    print("-----" * 20)
    print(result["answer"])
