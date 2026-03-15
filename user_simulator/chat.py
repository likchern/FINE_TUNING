import streamlit as st
import requests

API_URL = "http://localhost:8000/chat"

st.set_page_config(page_title="Qwen Chat Tester", layout="centered")
st.title("Qwen SFT Chat")

#####################################
# Initialize conversation state
#####################################

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "system",
            "content": (
                "你是一个医学健康问答助手。"
                "请直接给出清晰、简洁、最终的回答，"
                "不要展示思考过程。"
            )
        }
    ]

#####################################
# Render full chat history
#####################################

for msg in st.session_state.messages:
    if msg["role"] == "system":
        continue

    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

#####################################
# Chat input
#####################################

user_input = st.chat_input("请输入你的问题")

if user_input:
    # Append user message
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    # Show user message immediately
    with st.chat_message("user"):
        st.markdown(user_input)

    # Call server
    with st.chat_message("assistant"):
        with st.spinner("思考中..."):
            response = requests.post(
                API_URL,
                json={
                    "messages": st.session_state.messages,
                    "max_new_tokens": 2048,
                    "temperature": 0.2
                },
                timeout=120
            )

            reply = response.json()["reply"]
            st.markdown(reply)

    # Append assistant reply
    st.session_state.messages.append({
        "role": "assistant",
        "content": reply
    })
