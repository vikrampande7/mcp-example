import asyncio
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from mcp_use import MCPAgent, MCPClient

async def run_agent():
    # Load environment variables
    load_dotenv()    

    # Load configuration dictionary and Environment variables
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

    config = "mcp_servers.json"

    # Create MCPClient from configuration dictionary
    client = MCPClient.from_config_file(config)

    # Create LLM
    # llm = ChatOpenAI(model="gpt-4o")
    # llm = ChatGroq(model="llama3-8b-8192")
    llm = ChatGroq(model="qwen-qwq-32b")
    
    # Create agent with the client
    agent = MCPAgent(llm=llm, client=client, max_steps=30, memory_enabled=True)

    # Run the query
    print(f"Welcome to the MCP Agent!")
    print(f"You can ask me anything about the website. Type 'exit' to quit. Type 'clear' to clear the memory.")

    try:
        while True:
            # Get user input
            user_input = input("\n> ")

            if user_input.lower() == "exit":
                print("Exiting the MCP Agent...")
                break
            elif user_input.lower() == "clear":
                agent.clear_conversation_history()
                print("Conversation history cleared.")
                continue

            try:
                # Run the agent
                result = await agent.run(user_input)
                print(f"\nAssistant: {result}")
            except Exception as e:
                print(f"An error occurred: {e}")
            
    finally:
        # Close the MCPClient
        if client and client.sessions:
            await client.close_all_sessions()

                
if __name__ == "__main__":
    asyncio.run(run_agent())