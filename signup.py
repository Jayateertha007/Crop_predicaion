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

def save_users(users):
    with open(USER_FILE, 'w') as f:
        json.dump(users, f, indent=4)

def signup():
    st.title("Signup Page")

    new_user = st.text_input("Create Username")
    new_pass = st.text_input("Create Password", type="password")

    if st.button("Signup"):
        users = load_users()
        if new_user in users:
            st.error("Username already exists.")
        else:
            users[new_user] = new_pass
            save_users(users)
            st.success("Signup successful! Please login.")
            st.session_state.current_page = "login.py"
