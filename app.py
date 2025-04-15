import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import firebase_admin
from firebase_admin import credentials, firestore
import psycopg2

# Load Models
@st.cache_resource
def load_models():
    return {
        "Skin Disease": load_model("models/skin_disease_model.h5"),
        "Eye Disease": load_model("models/eye_disease_model.h5"),
        "Alzheimer’s Disease": load_model("models/alzheimer_model.h5"),
    }
models = load_models()

# Image Preprocessing
def preprocess_image(image):
    img = Image.open(image).convert('RGB').resize((224, 224))
    return np.expand_dims(np.array(img) / 255.0, axis=0)

# Database Connection (PostgreSQL)
conn = psycopg2.connect(database="disease_db", user="admin", password="pass123", host="localhost", port="5432")
cursor = conn.cursor()

# Firebase Connection
cred = credentials.Certificate("path/to/firebase_serviceAccountKey.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

# Streamlit UI
st.sidebar.title("Disease Prediction App")
disease_option = st.sidebar.selectbox("Select Disease", ["Skin Disease", "Eye Disease", "Alzheimer’s Disease"])

st.title(f"{disease_option} Prediction")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png"])
if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    processed_img = preprocess_image(uploaded_file)

    with st.spinner('Processing...'):
        prediction = models[disease_option].predict(processed_img)

    confidence_score = round(prediction[0][0] * 100, 2)
    st.success(f"Prediction: {confidence_score}% confidence")

    # Store in Firebase
    db.collection("predictions").add({"disease": disease_option, "prediction": str(confidence_score)})

    # Store in PostgreSQL
    cursor.execute("INSERT INTO results (disease, prediction) VALUES (%s, %s)", (disease_option, str(confidence_score)))
    conn.commit()

    st.sidebar.subheader("Previous Predictions")
    cursor.execute("SELECT * FROM results")
    past_results = cursor.fetchall()
    for record in past_results:
        st.write(f"**Disease:** {record[0]}, **Prediction:** {record[1]}")