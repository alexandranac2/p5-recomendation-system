from langchain_openai import ChatOpenAI


def analyse_promt(queryString):
    llm = ChatOpenAI(model="gpt-4o-mini")

    response = llm.invoke(
        "can you get the intension of the following query: "
        + queryString
        + "Give me sinonyms for the product? As output plase give me just the array of synonyms. No other text or explanation. no ```json or ``` or anything else. Just the array of synonyms.Prepare this for similarity serch in FAISS vector store. So you need to give me the array of synonyms that will be used for similarity search in FAISS vector store."
    )
    print(f"🔍 Analyse Promt Response: {response.content}")
    print(f"📝 LLM Response (type: {type(response.content).__name__}): {response.content}")
    return response.content
