from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever  # Or change if your file is named differently

model = OllamaLLM(model="llama3.2")

template = """
You are an expert in answering questions about a pizza restaurant.

Here are some relevant reviews: {reviews}

Here is the question to answer: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

while True:
    print("\n\n--------------------------")
    question = input("Ask your question (q to quit): ")
    print("\n\n")
    if question.lower() == "q":
        break

    # ✅ Use get_relevant_documents, not function call
    docs = retriever.get_relevant_documents(question)
    reviews = [doc.page_content for doc in docs]

    print("Reviews:", reviews)
    print("Question:", question)

    result = chain.invoke({
        "reviews": reviews,
        "question": question
    })

    print("Answer:", result if isinstance(result, str) else getattr(result, 'content', result))
