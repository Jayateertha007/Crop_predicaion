import streamlit as st





def home():
    st.title("Welcome to Crop Recommendation System")  
    st.write("""
    This web application helps farmers and agricultural enthusiasts make informed decisions about what crops to plant based on various environmental factors such as:

    - Nitrogen (N)
    - Phosphorus (P)
    - Potassium (K)
    - Temperature
    - Humidity
    - pH value
    - Rainfall

    The goal of this system is to provide personalized crop recommendations based on the given data inputs. 

    #### Features:
    - **User-Friendly Interface:** Easy-to-use form to input your farm's environmental data.
    - **Accurate Recommendations:** Our model recommends the best crops for your soil and climate conditions.
    - **Login System:** Secure login for access to crop recommendations.
    
    #### How It Works:
    1. **Login:** First, login with your credentials.
    2. **Recommendation:** Once logged in, provide your farm's environmental data such as soil nutrients and climate conditions.
    3. **Result:** Get the recommended crop and its corresponding image based on your inputs.
    
    Whether you're a small-scale farmer or involved in large-scale agriculture, our system helps you make data-driven decisions for better yields.


    """)
    st.write("Click below to proceed to login.")  
    if st.button("Go to Login"):
        st.session_state.page = "login"
