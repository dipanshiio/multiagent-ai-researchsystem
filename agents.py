
from langchain.agents import create_agent
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from tools import web_search, scrape_url
from dotenv import load_dotenv

load_dotenv()

# Model setup
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    max_retries=2,
    timeout=30,
)

# 1st agent: Search
def build_search_agent():
    return create_agent(
        model=llm,
        tools=[web_search],
    )

# 2nd agent: Reader
def build_reader_agent():
    return create_agent(
        model=llm,
        tools=[scrape_url],
    )

# Writer chain
writer_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a research writer. Be clear, factual, structured, "
        "and concise. Do not repeat information."
    ),
    (
        "human",
        """Write a research report on the topic below.

Topic: {topic}

Research:
{research}

Use this structure:
1. Introduction
2. Key Findings (3-5 points)
3. Conclusion
4. Sources (include URLs present in the research)

Use only supported facts. Keep the report under 500 words."""
    ),
])

writer_chain = writer_prompt | llm | StrOutputParser()

# Critic chain
critic_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a constructive research critic. Be specific and concise."
    ),
    (
        "human",
        """Review this research report:

{report}

Use this format:

Score: X/10

Strengths:
- ...

Areas to Improve:
- ...

One line verdict:
...

Keep the review under 120 words."""
    ),
])

critic_chain = critic_prompt | llm | StrOutputParser()
