import shutil
from pathlib import Path
from time import time

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


def main():
    in_path = Path("../1_replace/out")
    in_data = []
    out_dir = "./out"

    shutil.rmtree(out_dir, ignore_errors=True)

    for in_file in in_path.glob("*.txt"):
        with open(in_file, "r") as f:
            contents = f.read()
            in_data.append((in_file, contents))

    docs = []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, chunk_overlap=160, add_start_index=True
    )
    for in_file, contents in in_data:
        filename = in_file.name
        subj = in_file.stem

        chunks = text_splitter.create_documents(
            [contents], metadatas=[{"file": filename, "subj": subj}]
        )

        chunks = [c for c in chunks if len(c.page_content.strip()) >= 100]

        for idx, chunk in enumerate(chunks):
            start = chunk.metadata["start_index"]
            chunk.metadata.update(
                {
                    "chunk": idx,
                    "total_chunks": len(chunks),
                    "start_char": start,
                    "end_char": start + len(chunk.page_content),
                }
            )

        docs.extend(chunks)

    start = time()
    embeddings = OllamaEmbeddings(model="qwen3-embedding:0.6b", dimensions=1024)
    chroma = Chroma(embedding_function=embeddings, persist_directory=out_dir)

    print("generating embeddings...")
    batch_size = 32
    for i in range(0, len(docs), batch_size):
        print(f"processing batch {i}–{i + batch_size}...")
        batch = docs[i : i + batch_size]
        try:
            chroma.add_documents(batch)
        except Exception as e:
            print(f"batch failed {i}–{i + batch_size}: {e}")
    print(f"done in {time() - start} s")


if __name__ == "__main__":
    main()
