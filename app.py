import os
import gc
import tempfile
import uuid
import pandas as pd

from local_model import LocalMistralLLM
from dotenv import load_dotenv
import streamlit as st

# Load environment variables
load_dotenv()

# Session state for chat & model
if "id" not in st.session_state:
    st.session_state.id = uuid.uuid4()
    st.session_state.file_cache = {}
    st.session_state.model = None

# Initialize local Mistral-7B model
if st.session_state.model is None:
    model_path = "/Users/ravikuruganthy/myApps/models/mistral/mistral-7b-instruct-v0.1.Q4_K_M.gguf"
    st.session_state.model = LocalMistralLLM(model_path=model_path, n_ctx=4096, n_threads=4)

def reset_chat():
    st.session_state.messages = []
    st.session_state.context = None
    gc.collect()

def process_github_repo(github_url):
    """
    Placeholder function to process GitHub repo.
    This function should download & extract repo content.
    """
    return "Summary of repo", "Repo structure", "Repo content"

# Sidebar UI
with st.sidebar:
    st.header(f"Add your GitHub repository!")

    github_url = st.text_input("Enter GitHub repository URL", placeholder="GitHub URL")
    load_repo = st.button("Load Repository")

    if github_url and load_repo:
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                st.write("Processing repository...")
                repo_name = github_url.split('/')[-1]
                file_key = f"{st.session_state.id}-{repo_name}"

                if file_key not in st.session_state.file_cache:
                    summary, tree, content = process_github_repo(github_url)

                    content_path = os.path.join(temp_dir, f"{repo_name}_content.md")
                    with open(content_path, "w", encoding="utf-8") as f:
                        f.write(content)

                    st.session_state.file_cache[file_key] = content

                st.success("Repository processed! Ready to chat.")
        except Exception as e:
            st.error(f"Error: {e}")
            st.stop()

col1, col2 = st.columns([6, 1])
with col1:
    st.header(f"Chat with GitHub using Mistral </>")

with col2:
    st.button("Clear ↺", on_click=reset_chat)

# Initialize chat history
if "messages" not in st.session_state:
    reset_chat()

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask about the repo..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        try:
            repo_name = github_url.split('/')[-1]
            file_key = f"{st.session_state.id}-{repo_name}"
            repo_content = st.session_state.file_cache.get(file_key)

            if repo_content is None:
                st.error("Load a repository first!")
                st.stop()

            # Generate a response from Mistral-7B
            query_prompt = f"Repo Content:\n{repo_content}\n\nQuestion: {prompt}\n\nAnswer:"
            full_response = st.session_state.model.generate(query_prompt)

            message_placeholder.markdown(full_response)

        except Exception as e:
            st.error(f"Error processing query: {e}")
            full_response = "An error occurred."
            message_placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})