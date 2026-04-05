from config import state_history_path
from tool_func import start_node,checks_session_status,manual_login,session_login,Agent_calling,State,load_chat_history,store_chat_history
from langgraph.graph import StateGraph,START,END
import asyncio

async def linkedin_graph_automation(user_input):
    graph = StateGraph(State)

    graph.add_node('Start',start_node)
    graph.add_node('manual_login',manual_login)
    graph.add_node('agent_calling',Agent_calling)
    graph.add_node('session_login',session_login)
    graph.add_node('store_history',store_chat_history)

    graph.add_edge(START,'Start')
    graph.set_entry_point('Start')
    graph.add_conditional_edges('Start',checks_session_status)
    graph.add_edge('manual_login','agent_calling')
    graph.add_edge('manual_login',END)
    graph.add_edge('agent_calling','session_login')
    graph.add_edge('session_login','store_history')
    graph.add_edge('store_history',END)

    graph = graph.compile()

    
    initial_state = {
        "user_query":user_input,
        "chat_history":await load_chat_history(state_history_path),
        "content":'',
        'content_quality':''
    }

    response = await graph.ainvoke(initial_state)
    return response
