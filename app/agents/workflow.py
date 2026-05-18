from langgraph.graph import StateGraph,START,END
from app.agents.agent import (extraction_node,decision_making_function,reasoning_node,irrelevant_node,filter_node,AgentState,should_call_tools)
from langgraph.prebuilt import ToolNode
from app.agents.tools import search_places,search_hotels
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()

graph=StateGraph(AgentState)
graph.add_node("extractor",extraction_node)
graph.add_node("reasoning",reasoning_node)
graph.add_node("irrelevant",irrelevant_node)
graph.add_node("filter",filter_node)
graph.add_conditional_edges("extractor",decision_making_function,
                            {"irrelevant_query":"irrelevant",
                             "relevant_query":"reasoning"})
tool_node=ToolNode([search_places,search_hotels])
graph.add_node("tool_node",tool_node)
graph.add_conditional_edges("reasoning", should_call_tools,{"tool_node":"tool_node","filtering":"filter"})
graph.add_edge("tool_node","reasoning")

graph.set_entry_point("extractor")
graph.add_edge("irrelevant",END)
graph.add_edge("filter",END)
travel_app=graph.compile(checkpointer=memory)
travel_app.get_graph().draw_mermaid_png(output_file_path="agent_graph.png")
# print("going to invoke the agent")

# #result=app.invoke({"messages":["I want to make 1 day trip to kochi with family of 4 members and the budget is 10000.Suggest me best travel plan"]})
# from rich.console import Console
# from rich.markdown import Markdown
# console = Console()

# result = app.invoke({
#     "messages": [
#         "I want to make 2 days trip to calicut  with family of 4 members and the budget is 10000.Suggest me best travel plan"
#     ]
# })

# markdown = Markdown(result["final_response"])

# console.print(markdown)
