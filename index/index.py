import hashlib
import sys
from argparse import ArgumentParser
from pathlib import Path

import chromadb
import structlog
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

log = structlog.get_logger()


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def process(docs_path, chroma):
    disk_state = {
        str(p.relative_to(docs_path)): file_hash(p) for p in docs_path.rglob("*.txt")
    }
    stored = chroma.get(include=["metadatas"])
    db_state = {m["file"]: m["file_hash"] for m in stored["metadatas"]}

    to_add = [s for s in disk_state if s not in db_state]
    to_update = [
        s for s in disk_state if s in db_state and db_state[s] != disk_state[s]
    ]
    to_delete = [s for s in db_state if s not in disk_state]

    if not (to_add or to_update or to_delete):
        log.info("nothing to do, bye")
        return

    log.info(
        "processing files in directory",
        to_add=len(to_add),
        to_update=len(to_update),
        to_delete=len(to_delete),
    )

    log.info("deleting old entries...")

    for source in to_update + to_delete:
        chroma._collection.delete(where={"file": source})

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, chunk_overlap=160, add_start_index=True
    )

    log.info("inserting new and updated entries...")

    for source in to_add + to_update:
        log.info("processing file", name=source)

        source_path = docs_path / Path(source)
        subj = source_path.stem

        with open(source_path, 'r') as f:
            contents = f.read()

        chunks = text_splitter.create_documents(
            [contents], metadatas=[{"file": source, "subj": subj, "file_hash": disk_state[source]}]
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

        ids = [f"{disk_state[source]}:{i}" for i in range(len(chunks))]

        batch_size = 32
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            ids_batch = ids[i:i+batch_size]
            chroma.add_documents(batch, ids=ids_batch)

    log.info("processing complete")


def main():
    parser = ArgumentParser()
    parser.add_argument("--chroma-host", action="store", type=str, required=True)
    parser.add_argument("--chroma-port", action="store", type=int, required=True)
    parser.add_argument("--docs", action="store", type=str, required=True)
    args = parser.parse_args()

    docs_path = Path(args.docs)
    if not docs_path.is_dir():
        log.error("cannot access docs directory", path=args.docs)
        return 1

    client = chromadb.HttpClient(host=args.chroma_host, port=args.chroma_port)
    embeddings = OllamaEmbeddings(model="qwen3-embedding:0.6b", dimensions=1024)
    chroma = Chroma(client=client, embedding_function=embeddings)

    process(docs_path, chroma)

    return 0


if __name__ == "__main__":
    status = main()
    sys.exit(status)
