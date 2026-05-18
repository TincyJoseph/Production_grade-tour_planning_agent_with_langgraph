import streamlit as st
import requests

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

        with st.spinner("Planning your trip..."):

            response = requests.post(
                "http://127.0.0.1:8000/plan-trip",
                json={
                    "query": user_query
                }
            )

            data = response.json()

            st.markdown(data["final_response"])