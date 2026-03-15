"""
Interactive test — chat with the Orchestrator in the terminal.
Type messages, see how it routes to tools.
Type 'quit' to exit.
"""

from dotenv import load_dotenv
load_dotenv()

from src.agents.orchestrator.orchestrator import Orchestrator

agent = Orchestrator()

print("=" * 50)
print("ORCHESTRATOR TEST (type 'quit' to exit)")
print("=" * 50)

while True:
    user_input = input("\nYou: ").strip()
    if user_input.lower() in ("quit", "exit", "q"):
        break

    print("\nThinking...")
    response = agent.chat(user_input)
    print(f"\nAssistant: {response}")