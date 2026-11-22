from dotenv import load_dotenv
from deepagents import create_deep_agent
from langchain.messages import HumanMessage

load_dotenv()

EXAMPLE_QUERY = "How do I stream intermediate tool results from a subagent?"

baseline_agent = create_deep_agent(
    model="google_genai:gemini-3.5-flash-lite",    
    tools=[],
    system_prompt=(
        "You are a helpful LangChain documentation assistant. "
        "Answer questions about LangChain APIs and patterns."
    ),
)

if __name__ == "__main__":
    result = baseline_agent.invoke(
        {"messages": [HumanMessage(content=EXAMPLE_QUERY)]}
    )
    print(result["messages"][-1].text)
