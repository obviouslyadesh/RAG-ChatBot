import streamlit as st
import requests
import os
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

API = "http://127.0.0.1:8000"

st.title("🚀 RAG SaaS Chatbot")

# Upload section
st.subheader("Upload Documents")
file = st.file_uploader("Upload PDF")

if file:
    res = requests.post(
        f"{API}/upload",
        files={"file": file.getvalue()}
    )
    st.success(res.json())

# Chat section
st.subheader("Chat")

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    st.chat_message(m["role"]).markdown(m["content"])

user = st.chat_input("Ask something...")

if user:
    st.chat_message("user").markdown(user)

    res = requests.post(
        f"{API}/chat",
        json={"question": user}
    )

    answer = res.json()["answer"]

    st.chat_message("assistant").markdown(answer)

    st.session_state.messages.append({"role": "user", "content": user})
    st.session_state.messages.append({"role": "assistant", "content": answer})