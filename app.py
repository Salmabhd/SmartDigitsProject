import streamlit as st
import joblib
import numpy as np
from PIL import Image, ImageOps

# Charger le modèle entraîné avec joblib
model = joblib.load('mnist_mlp_model.joblib')

# Fonction de prétraitement de l'image
def preprocess_image(image):
    image = ImageOps.grayscale(image)
    image = image.resize((28, 28))
    image = np.array(image)
    image = image.reshape(1, 28 * 28).astype('float32')
    image /= 255.0
    return image

# Barre latérale
st.sidebar.title("Welcome to My Dashboard")
st.sidebar.image('CoreDuo.png', use_column_width=True, caption="")
st.sidebar.write("**AI & Machine Learning Specialist**")

st.sidebar.header("About the Model")
st.sidebar.write("""
    The MNIST Digit Recognition model is a sophisticated neural network designed to classify handwritten digits from 0 to 9. It is built on the MNIST dataset, which comprises thousands of digit images.

    Model Details:
    - Type: Feedforward Neural Network (MLPClassifier)
    - Hidden Layers: 2
    - Activation: ReLU-like (scikit-learn handles activation internally)
    - Training Epochs: 15
    - Batch Size: 200
""")

st.sidebar.header("About us")
st.sidebar.write("We are the CoreDuo, a team to develop your machine learning models...")

st.sidebar.header("Contact Information")
st.sidebar.write("[LinkedIn](https://www.linkedin.com/in/)")
st.sidebar.write("[GitHub](https://github.com/SalmaBhd/)")
st.sidebar.write("[Email](mailto:babahdisalma@gmail.com)")

# Section principale
st.title("MNIST Digit Recognition")
st.write("Upload a digit image to classify it.")

# Upload de l'image
uploaded_file = st.file_uploader("Choose an image...", type="png")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Prétraitement
    processed_image = preprocess_image(image)

    # Prédiction
    predicted_digit = model.predict(processed_image)[0]

    # Affichage du résultat
    st.write(f"**Predicted Digit:** {predicted_digit}")

# Bouton de prédiction (optionnel)
if st.button('Predict'):
    if uploaded_file is not None:
        st.write("Prediction Complete")
    else:
        st.write("Please upload an image first.")
