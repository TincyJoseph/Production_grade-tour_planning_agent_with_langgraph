from typing import TypedDict,Annotated,List,Optional,Literal,Union
from operator import add as add_messages
from langchain_core.messages import BaseMessage
from langchain.messages import SystemMessage,HumanMessage,AIMessage
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from operator import add
import os
from app.agents.tools import search_places,search_hotels
from langgraph.graph import END
from langchain_ollama import ChatOllama
_=load_dotenv()


class AgentState(TypedDict):
    messages:Annotated[List[BaseMessage],add]
    destination:Optional[str]=""
    budget:Optional[int]=0
    days:Optional[int]=0
    no_of_people:Optional[int]=0

    final_response:Optional[str]
    is_relevant:Optional[bool]
    missing_fields:List[str]
    llm_calls:Optional[int]
    
class Extraction(TypedDict):
    is_relevant:bool
    destination:str
    budget:int
    days:int
    no_of_people:int
    missing_fields:List[str]
    
filter_llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)
llm=ChatOllama(
    model="qwen2.5:7b-instruct",
    temperature=0)
llm_with_structure=filter_llm.with_structured_output(Extraction)
llm_with_tools= filter_llm.bind_tools([search_places,search_hotels])

def relevancy_check_node(state:AgentState)->AgentState:
    relevant_words={
    "trip",
    "travel",
    "tour",
    "vacation",
    "holiday",
    "hotel",
    "flight",
    "destination",
    "budget",
    "days",
    "honeymoon",
    "itinerary",
    "resort",
    "beach",
    "mountain",
    "booking",
    "stay"
}
    return {"is_relevant":any(word in state["messages"][-1] for word in relevant_words)}
def is_relevant_function(state:AgentState)->str:
    if state["is_relevant"]:
        return "relevant"
    return "irrelevant"



async def extraction_node(state:AgentState):
    """_summary_

    Args:
        state (AgentState): _description_
    """
    prompt1=[SystemMessage(content="You are a helping assistant.Your task is to Extract destination,number of days,number of people and budget for the tour from the query if they exist else mark the corresponing as None.Return the missing fields also")]+state["messages"]

    result=await llm_with_structure.ainvoke(prompt1)

    print(f"data extracted======= {result}")
    return {
    "destination": result.get("destination"),
    "budget": result.get("budget"),
    "days": result.get("days"),
    "missing_fields": result.get("missing_fields"),
    "no_of_people": result.get("no_of_people"),
    "llm_calls": state.get("llm_calls", 0) + 1
}
     

async def reasoning_node(state: AgentState) -> AgentState:
    """
    Analyze the trip details and decide whether tool calling is needed.
    If required, directly call the appropriate tools.
    """

    print("----inside reasoning node-------")

    messages = [
        SystemMessage(
            content="""
You are the reasoning node of a travel planning agent.

Your job:
1. Decide whether external information is needed.
2. If hotel/place information is needed, DIRECTLY call the appropriate tool.
3. Never explain which tool should be called in text.
4. Actually call the tool.
5. If tool results are already available in the conversation, do NOT call tools again.
"""
        ),

        HumanMessage(
            content=f"""
Trip Details:

Destination: {state.get('destination')}
Budget: {state.get('budget')}
Number of People: {state.get('no_of_people')}
Days: {state.get('days')}
"""
        )
    ]

    # include previous conversation/tool results
    messages.extend(state["messages"])

    result = await llm_with_tools.ainvoke(messages)

    return {"messages": [result], "llm_calls": state.get("llm_calls", 0) + 1}

def should_call_tools(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    print(f"Last message is {last_message}")
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tool_node"
    return "filtering"


# def asking_node(state:AgentState):
#     prompt3=f"Your task is to check the {state['missing fields']} and ask the user to enter the missing fields."

#     result3=llm.invoke(HumanMessage(content=prompt3))

#     return result3
def irrelevant_node(state:AgentState)->str:
    return {"final_response":AIMessage(content="I am a travel planning assistant.Ask me travel related queries")}

# def filter_node(state:AgentState)->AgentState:
#     """Analyse the current state messages and summarize the best 3 options to stay and best places to visit.Don't call any tool"""
#     print("inside filter node------------------------------")
#     final_response=filter_llm.invoke([SystemMessage(content="Analyse the current state messages and summarize the best 3 options to stay and best places to visit.Don't call any tool")]+state["messages"])
#     return({"final_response":final_response})

from langchain_core.messages import ToolMessage, AIMessage

async def filter_node(state: AgentState) -> AgentState:
    print("inside filter node------------------------------")

    cleaned_messages = []

    for msg in state["messages"]:

        # Skip AI messages that requested tools
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            continue

        # Keep tool outputs as normal text context
        if isinstance(msg, ToolMessage):
            cleaned_messages.append(
                HumanMessage(content=f"Tool Result: {msg.content}")
            )
            continue

        cleaned_messages.append(msg)

    final_response = await filter_llm.ainvoke(
        [
            SystemMessage(
                content="""
                Summarize the travel plan from the available information.

                Return:
                - Best 3 stay options
                - Best places to visit

                Do not call tools.
                """
            )
        ] + cleaned_messages
    )

    return {"final_response": final_response, "llm_calls": state.get("llm_calls", 0) + 1}
