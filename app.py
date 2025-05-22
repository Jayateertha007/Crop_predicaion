import streamlit as st

# Set this as the VERY FIRST Streamlit command (only once)
st.set_page_config(page_title="Crop Recommendation App")

from home import home
from login import login
from signup import signup
from recommendation import recommendation

def navbar():
    st.sidebar.title("Navigation")
    choice = st.sidebar.radio("Go to", ["Home", "Login", "Signup", "Recommendation"])

    # Add logout button if logged in
    if st.session_state.get("logged_in", False):
        if st.sidebar.button("Logout"):
            # Clear session states related to login
            for key in ["logged_in", "username"]:
                if key in st.session_state:
                    del st.session_state[key]
            # Set page to Home after logout
            st.session_state.current_page = "Home"
            # No st.experimental_rerun() here — rely on Streamlit's auto rerun

    # Only update current_page if user changed it manually
    # But don't override if logout changed it to "Home"
    if st.session_state.get("current_page") != "Home" or choice != "Home":
        st.session_state.current_page = choice

def main():
    if "current_page" not in st.session_state:
        st.session_state.current_page = "Home"

    navbar()

    page = st.session_state.current_page

    if page == "Home":
        home()
    elif page == "Login":
        login()
    elif page == "Signup":
        signup()
    elif page == "Recommendation":
        if st.session_state.get('logged_in', False):
            recommendation()
        else:
            st.error("Please log in first!")

if __name__ == "__main__":
    main()
