from fastapi import FastAPI, HTTPException
from langgraph import graph

from app.models.schemas import ChatRequest,ChatResponse
from app.agents.workflow import graph
import os
from dotenv import load_dotenv
load_dotenv()
DB_URI = os.getenv("DB_URI")
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

app = FastAPI(
    title="AI Travel Planner"
)


@app.get("/")
async def root():
    return {"message": "AI Travel Planner API is running"}



@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):

    try:
        async with AsyncPostgresSaver.from_conn_string(DB_URI) as checkpointer:
            await checkpointer.setup()
            travel_app=graph.compile(checkpointer=checkpointer)
            config = {
            "configurable": {
                "thread_id": request.thread_id
            }
            }

            result = await travel_app.ainvoke(
            {
                "messages": [request.query]
            },
            config=config
         )

        print("-----result from agent------------------", result)

        return ChatResponse(
            response=result["final_response"].content
        )

    except Exception as e:
        return ChatResponse(
            response=str(e)
        )
