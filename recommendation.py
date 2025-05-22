import streamlit as st
import numpy as np
import pickle


# Function to load the trained model
def load_model():
    with open('crop_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    return model

# Crop recommendation form
def recommendation_form(model):
    st.title("Crop Recommendation System")

    # Input form for crop recommendation
    N = st.number_input("Nitrogen (N)", min_value=0)
    P = st.number_input("Phosphorus (P)", min_value=0)
    K = st.number_input("Potassium (K)", min_value=0)
    temperature = st.number_input("Temperature (°C)")
    humidity = st.number_input("Humidity (%)")
    ph = st.number_input("pH value")
    rainfall = st.number_input("Rainfall (mm)")

    if st.button("Recommend Crop"):
        # Check if all inputs are left at their default values
        if all(val == 0 for val in [N, P, K]):
            st.warning("⚠️ Please fill in the input fields.")
        else:
            input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
            prediction = model.predict(input_data)
            st.success(f"The recommended crop is: {prediction[0]}")

            crop_images = {
                "rice": "Images/rice.jpg",
                "maize": "Images/maize.jpg",
                "chickpea": "Images/chickpea.jpg",
                "kidneybeans": "Images/kidneybeans.jpg",
                "pigeonpeas": "Images/pigeonpeas.jpg",
                "mothbeans": "Images/mothbeans.jpg",
                "mungbean": "Images/mungbean.jpg",
                "blackgram": "Images/blackgram.jpg",
                "lentil": "Images/lentil.jpg",
                "pomegranate": "Images/pomegranate.jpg",
                "banana": "Images/banana.jpg",
                "mango": "Images/mango.jpg",
                "grapes": "Images/grapes.jpg",
                "watermelon": "Images/watermelon.jpg",
                "muskmelon": "Images/muskmelon.jpg",
                "apple": "Images/apple.jpg",
                "orange": "Images/orange.jpg",
                "papaya": "Images/papaya.jpg",
                "coconut": "Images/coconut.jpg",
                "cotton": "Images/cotton.jpg",
                "jute": "Images/jute.jpg",
                "coffee": "Images/coffee.jpg"
            }

            recommended_crop = prediction[0].lower()
            if recommended_crop in crop_images:
                st.image(
                    crop_images[recommended_crop],
                    caption=f"Image of {recommended_crop.capitalize()}",
                    use_container_width=True
                )

# Show recommendation page if logged in
def recommendation():
    if 'logged_in' in st.session_state and st.session_state.logged_in:
        model = load_model()
        recommendation_form(model)
    else:
        st.error("Please login first!")
