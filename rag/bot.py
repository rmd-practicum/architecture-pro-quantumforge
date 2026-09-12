import asyncio
import os
import sys
from argparse import ArgumentParser

from langchain_chroma import Chroma
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import (
    create_stuff_documents_chain,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)


class Bot:
    def __init__(self):
        self.embeddings = OllamaEmbeddings(
            model="qwen3-embedding:0.6b", dimensions=1024
        )
        self.store = Chroma(
            embedding_function=self.embeddings,
            persist_directory="../knowledge_base/2_index/out",
        )
        self.retriever = VectorStoreRetriever(vectorstore=self.store)
        self.llm = OllamaLLM(model="gemma3:4b", num_ctx=8192)

        self.system_prompt = (
            "You are an assistant who first thinks and then answers. ALWAYS output your thinking steps. "
            "Use the given context to answer the question. "
            "If you don't know the answer, your answer must be: I don't know. "
            "Use one sentence and keep the answer concise. "
            "Example: "
            "Q: Who discovered Heart of Raitioss? "
            "A: Heart of Raitioss was discovered by Tekreir miners. "
            "Context: {context}"
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                ("human", "{input}"),
            ]
        )
        question_answer_chain = create_stuff_documents_chain(self.llm, prompt)
        self.chain = create_retrieval_chain(self.retriever, question_answer_chain)

    def handle(self, q):
        return self.chain.invoke({"input": q})


def run_telegram(bot: Bot):
    lock = asyncio.Lock()

    async def on_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text(
            "What do you want to learn about the lore of Liakramar?"
        )

    async def on_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
        q = update.message.text
        async with lock:
            await update.message.chat.send_action(ChatAction.TYPING)
            res = await asyncio.to_thread(bot.handle, q)
        await update.message.reply_text(res["answer"])

    app = Application.builder().token(os.environ["TELEGRAM_TOKEN"]).build()
    app.add_handler(CommandHandler("start", on_start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_message))
    app.run_polling()


def run_repl(bot: Bot):
    while True:
        try:
            query = input("query: ")
            res = bot.handle(query)
            print(res["answer"])
        except EOFError:
            break


def main():
    parser = ArgumentParser()
    parser.add_argument("mode")
    args = parser.parse_args()

    bot = Bot()

    if args.mode == "repl":
        return run_repl(bot)
    elif args.mode == 'telegram':
        return run_telegram(bot)

    raise RuntimeError(f"unknown mode: {args.mode}")


if __name__ == "__main__":
    status = main()
    sys.exit(status)
