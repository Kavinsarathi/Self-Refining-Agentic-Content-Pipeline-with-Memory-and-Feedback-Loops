from graph import linkedin_graph_automation
import asyncio

def main(user_input):
    response = asyncio.run(linkedin_graph_automation(user_input=user_input))
    return response

if __name__ == '__main__':
    # user_input = input()
    user_input = "AI Agent"
    main(user_input)
