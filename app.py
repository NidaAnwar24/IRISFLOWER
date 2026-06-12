import streamlit as st
import joblib
import numpy as np
import os

# Page config
st.set_page_config(page_title="Iris Flower Predictor", page_icon="🌸")

# Get project directory (fixes image path issues)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# -------------------------------
# Background + Animations
# -------------------------------
st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg,#6a0dad,#b19cd9,#d8b4fe);
}

/* Snow animation */

@keyframes snow {
  0% {transform: translateY(-10vh);}
  100% {transform: translateY(110vh);}
}

.snowflake {
  position: fixed;
  top: -10px;
  font-size: 20px;
  animation: snow linear infinite;
}

.s1{left:10%; animation-duration:8s;}
.s2{left:30%; animation-duration:10s;}
.s3{left:50%; animation-duration:7s;}
.s4{left:70%; animation-duration:9s;}
.s5{left:90%; animation-duration:11s;}

/* Flower floating animation */

@keyframes flowers {
  0% {transform: translateY(-10vh) rotate(0deg);}
  100% {transform: translateY(110vh) rotate(360deg);}
}

.flower {
  position: fixed;
  top: -10px;
  font-size: 28px;
  animation: flowers linear infinite;
}

.f1{left:15%; animation-duration:12s;}
.f2{left:40%; animation-duration:14s;}
.f3{left:65%; animation-duration:11s;}
.f4{left:80%; animation-duration:13s;}

</style>

<!-- Snow -->
<div class="snowflake s1">❄️</div>
<div class="snowflake s2">❄️</div>
<div class="snowflake s3">❄️</div>
<div class="snowflake s4">❄️</div>
<div class="snowflake s5">❄️</div>

<!-- Flowers -->
<div class="flower f1">🌸</div>
<div class="flower f2">🌺</div>
<div class="flower f3">🌼</div>
<div class="flower f4">🌷</div>

""", unsafe_allow_html=True)

# -------------------------------
# Load Model
# -------------------------------
model = joblib.load(os.path.join(BASE_DIR, "iris_model.pkl"))

# -------------------------------
# App Title
# -------------------------------
st.title("🌸 Iris Flower Prediction App")

# -------------------------------
# Input Sliders
# -------------------------------
col1, col2 = st.columns(2)

with col1:
    sepal_length = st.slider("Sepal Length", 4.0, 8.0, 5.4)
    sepal_width = st.slider("Sepal Width", 2.0, 4.5, 3.4)

with col2:
    petal_length = st.slider("Petal Length", 1.0, 7.0, 1.3)
    petal_width = st.slider("Petal Width", 0.1, 2.5, 0.2)

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict Flower"):

    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    prediction = model.predict(features)

    flower_names = ["Setosa","Versicolor","Virginica"]
    result = flower_names[prediction[0]]

    st.success(f"Predicted Flower: {result}")

    # Confetti animation
    st.balloons()

    # -------------------------------
    # Show Flower Image
    # -------------------------------
    if result == "Setosa":
        image_path = os.path.join(BASE_DIR, "images", "setosa.jpg")
        st.image(image_path, caption="Iris Setosa", width=350)

    elif result == "Versicolor":
        image_path = os.path.join(BASE_DIR, "images", "versicolor.jpg")
        st.image(image_path, caption="Iris Versicolor", width=350)

    else:
        image_path = os.path.join(BASE_DIR, "images", "virginica.jpg")
