import streamlit as st
import json
import os

USER_FILE = "users.json"

def load_users():
    if not os.path.exists(USER_FILE):
        with open(USER_FILE, 'w') as f:
            json.dump({}, f)
    with open(USER_FILE, 'r') as f:
        return json.load(f)

def login():
    st.title("Login Page")

    username = st.text_input("Username")
    password = st.text_input("Password", type='password')

    if st.button("Login"):
        users = load_users()
        if username in users and users[username] == password:
            st.session_state.logged_in = True
            st.session_state.username = username
            st.success("Login successful!")
            st.session_state.current_page = "Recommendation"
        else:
            st.error("Invalid credentials, please try again.")
