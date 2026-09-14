import json
import sys
from argparse import ArgumentParser

import requests
from langchain_ollama import ChatOllama
from openevals.llm import create_llm_as_judge
from openevals.prompts import (
    CORRECTNESS_PROMPT,
    RAG_GROUNDEDNESS_PROMPT,
    RAG_RETRIEVAL_RELEVANCE_PROMPT,
)


class Judges:
    def __init__(self, llm):
        self.correctness = create_llm_as_judge(
            judge=llm, prompt=CORRECTNESS_PROMPT, feedback_key="correctness"
        )
        self.groundedness = create_llm_as_judge(
            judge=llm, prompt=RAG_GROUNDEDNESS_PROMPT, feedback_key="groundedness"
        )
        self.relevance = create_llm_as_judge(
            judge=llm,
            prompt=RAG_RETRIEVAL_RELEVANCE_PROMPT,
            feedback_key="retrieval_relevance",
        )


class Client:
    def __init__(self, host, port):
        self._host = host
        self._port = port

    def query(self, query):
        r = requests.post(
            f"http://{self._host}:{self._port}/query", data=json.dumps({"query": query})
        )

        resp = r.json()
        return resp


def evaluate_known(client, judges, q, a):
    resp = client.query(q)
    answer = resp["response"].rpartition("\n")[2]

    docs = list(s["chunk_text"] for s in resp["sources"])
    context = {"documents": docs}

    correctness = judges.correctness(inputs=q, outputs=answer, reference_outputs=a)
    groundedness = judges.groundedness(context=context, outputs=answer)
    relevance = judges.relevance(inputs={"question": q}, context=context)

    return {
        "q": q,
        "expected": a,
        "actual": answer,
        "correctness": correctness,
        "groundedness": groundedness,
        "relevance": relevance,
    }


def evaluate_unknown(client, q):
    resp = client.query(q)
    answer = resp["response"].rpartition("\n")[2]

    is_success = resp["is_success"]

    return {
        "q": q,
        "actual": answer,
        "is_unsuccessful": not is_success
    }


def run_test(client, questions):
    judges = Judges(ChatOllama(model="gemma3:4b"))

    known_results = []
    unknown_results = []

    for entry in questions["known_questions"]:
        res = evaluate_known(client, judges, entry["q"], entry["a"])
        known_results.append(res)

    for entry in questions["unknown_questions"]:
        res = evaluate_unknown(client, entry)
        unknown_results.append(res)

    for r in known_results:
        print(json.dumps(r))

    for r in unknown_results:
        print(json.dumps(r))


def main():
    parser = ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    with open("golden_questions.json", "r") as f:
        questions = json.load(f)
        run_test(Client(args.host, args.port), questions)

    return 0


if __name__ == "__main__":
    status = main()
    sys.exit(status)
