from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import Optional, Dict, List
import os
from dotenv import load_dotenv

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    print("❌ Error: OPENAI_API_KEY not set")
    print("   Create .env file with: OPENAI_API_KEY=sk-your-key")
    exit(1)


class understand_promt(BaseModel):
    """What the LLM extracts from user query"""

    intent: str = Field(description="search, gift, comparison, specific_need")
    category: Optional[str] = None
    price_range: Optional[Dict[str, float]] = None
    attributes: Optional[Dict[str, str]] = None
    use_case: Optional[str] = None
    product: str = Field(description="The product that the user is searching for")


def analyse_promt(queryString):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    response = llm.with_structured_output(understand_promt).invoke(
        "can you get the intension of the following query: "
        + queryString
    )
    print(f"🔍 Analyse Promt Response: {response}")

    return response
