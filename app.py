from dotenv import load_dotenv
import streamlit as st
from utils import *
from urllib.parse import urlparse

def is_valid_url(url):
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False

load_dotenv()  

st.set_page_config(page_title="Website Chatbot", page_icon="🤖", layout="centered")

if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'chat_active' not in st.session_state:
    st.session_state.chat_active = False

st.header("Website Chatbot", divider="rainbow")

with st.sidebar:
    st.title("Website Chatbot")
    
if st.sidebar.button("Already trained Websites"):
    names = all_name_spaces()

    if len(names) == 0:
        st.sidebar.write("No websites trained yet")
    else:
        for k in names:
            st.sidebar.write(k.capitalize())

st.sidebar.markdown("---")

st.session_state['website_name'] = st.sidebar.text_input("Enter Your Company Name" , key="name")

st.session_state['Website_url'] = st.sidebar.text_input("Enter Website URL" , key="url" )

load_button = st.sidebar.button("Learn the Website")

if load_button:
    if st.session_state['Website_url'] != "" or st.session_state['website_name'] != "":
        if is_valid_url(st.session_state['Website_url']):
            with st.spinner("Loading website details..."):
                st.session_state.chat_active = True
        else:
            st.error("Please enter a valid URL")
        if not namespace_exists(st.session_state['website_name'].lower()):
            vector_store = get_vectorstore_from_url(st.session_state['Website_url'] , st.session_state['website_name'].lower())
            st.sidebar.success("Data Pushed to Pinecone Successfully")
        else:
            st.sidebar.write("Correct ")
            st.sidebar.success("Data Pushed to Pinecone Successfully")
        
    else:
        st.sidebar.error("Opps!! Please provide the website url")

st.session_state['running'] = False

st.write("Note: The information provided will reflect what is available on the website.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

user_input = st.chat_input("Type your message here...")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.write(user_input)

    response = get_context_retriever_chain(user_input ,st.session_state['website_name'].lower())
    st.session_state.messages.append({"role": "assistant", "content": response.content})
    with st.chat_message("assistant"):
        st.write(response.content)

st.sidebar.markdown("---")
st.sidebar.markdown("Built with Streamlit ❤️")