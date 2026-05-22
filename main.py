from retriever import FAISSRetriever
from generator import QAGenerator


class RAGSystem:

    def __init__(self):

        print("Initializing retriever...")
        self.retriever = FAISSRetriever()

        print("Loading index...")
        self.retriever.load("my_vnu_index")

        print("Initializing generator...")
        self.generator = QAGenerator()

    def ask(
        self,
        question
    ):

        print("\nRetrieving documents...")

        retrieved_docs = self.retriever.search(
            query=question,
            top_n=15,
            top_k=3
        )

        print("\nTop retrieved contexts:\n")

        for idx, doc in enumerate(retrieved_docs):

            print(f"========== Context {idx+1} ==========")
            print(doc["text"][:500])
            print()

        print("Generating answer...\n")

        answer = self.generator.get_answer(
            question=question,
            retrieved_docs=retrieved_docs
        )

        return answer
 

if __name__ == "__main__":

    rag = RAGSystem()

    while True:

        question = input("\nQuestion: ")

        if question.lower() in ["exit", "quit"]:
            break

        answer = rag.ask(question)

        print("\nFinal Answer:")
        print(answer)