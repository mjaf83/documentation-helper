from backend.core import run_llm
import streamlit as st


st.header("Documentation Helper Bot")

def create_source_string(source_urls: set[str]) -> str:
    if not source_urls:
        return ""
    sources_list = list(source_urls)
    sources_list.sort()
    sources_string  = "sources:\n"
    for i, sources in enumerate(sources_list):
        sources_string += f"{i+1}. {sources}\n"
    return sources_string

prompt = st.text_input("Prompt:", placeholder="Enter your question here:")

if "user_prompt_history" not in st.session_state:
    st.session_state["user_prompt_history"] = []
if "chat_result_history" not in st.session_state:
    st.session_state["chat_result_history"] = []

if prompt:
    with st.spinner("Loading..."):
        response = run_llm(query=prompt)
        sources = set([doc.metadata["source"] for doc in response["source_documents"]])
        formatted_response = f"{response['result']}\n\n {create_source_string(sources)}"
        st.session_state["user_prompt_history"].append(prompt)
        st.session_state["chat_result_history"].append(formatted_response)

if st.session_state["chat_result_history"]:
    for generated_response, user_query in zip(st.session_state["chat_result_history"], st.session_state["user_prompt_history"]):
        st.chat_message("user").write(user_query)
        st.chat_message("assistant").write(generated_response)
