import sys
from pathlib import Path
import uuid
sys.path.append(str(Path(__file__).resolve().parents[2]))

import streamlit as st
from app.agents import workflow as wf 


st.set_page_config(
    page_title="AI Travel Planner",
    page_icon="✈️",
    layout="centered"
)

st.title("✈️ AI Travel Planner")

st.write("Plan your trip using AI")

user_query = st.text_area(
    "Enter your travel request",
    placeholder="I want a 1 day trip to Kochi with family of 4 and budget 10000"
)

if st.button("Generate Travel Plan"):

    if user_query.strip() == "":
        st.warning("Please enter your travel request")
    else:
        if "thread_id" not in st.session_state:
            st.session_state.thread_id = str(uuid.uuid4())
        config = {"configurable": { "thread_id": st.session_state.thread_id}
}
        with st.spinner("Planning your trip..."):

            result = wf.travel_app.invoke({
                "messages": [user_query]
            },config=config)

            final_response = result["final_response"].content

            st.markdown(final_response)