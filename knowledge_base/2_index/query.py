import argparse

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    args = parser.parse_args()
    query = args.query

    embeddings = OllamaEmbeddings(model="qwen3-embedding:0.6b", dimensions=1024)
    store = Chroma(embedding_function=embeddings, persist_directory="./out")
    query_text = f"Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: {query}"
    results = store.similarity_search(query=query_text, k=3)
    for doc in results:
        print(f"* {doc.page_content} [{doc.metadata}]")


if __name__ == "__main__":
    main()
