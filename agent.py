import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.agent_toolkits.load_tools import load_tools
# 1. FIXED IMPORT: Changed from langgraph to langchain.agents
from langchain.agents import create_agent

# Load environment variables from .env
load_dotenv()

# Initialize Groq LLM with LLaMA 3
llm = ChatGroq(
    model="llama-3.1-8b-instant", 
    temperature=0,
    groq_api_key=os.getenv("GROQ_API_KEY")
)

# Load basic mathematical tools
tools = load_tools(["llm-math"], llm=llm)

# 2. FIXED INSTANTIATION: Use explicit keyword arguments for the new function
agent_executor = create_agent(
    model=llm, 
    tools=tools,
    system_prompt="You are a helpful mathematical assistant."
)

if __name__ == "__main__":
    query = "What is 25 multiplied by 48?"
    print(f"User: {query}\n")

    # Run the agent
    messages = agent_executor.invoke({"messages": [("user", query)]})

    # Print the final response from the agent
    final_response = messages["messages"][-1].content
    print(f"Agent Response: {final_response}")
