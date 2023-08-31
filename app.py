import streamlit as st
import numpy as np
from model import train_and_save_model

model, iris_data = train_and_save_model()

# Uygulama başlığı
st.title("Iris Çiçeği Türü Tahmini")

# Kullanıcıdan girdileri alma
sepal_length = st.slider("Sepal Uzunluğu (cm)", 4.0, 8.0, 5.5)
sepal_width = st.slider("Sepal Genişliği (cm)", 2.0, 4.5, 3.5)
petal_length = st.slider("Petal Uzunluğu (cm)", 1.0, 7.0, 4.5)
petal_width = st.slider("Petal Genişliği (cm)", 0.1, 2.5, 1.5)

# Kullanıcının girdiği değerleri bir numpy dizisine dönüştürme
features = np.array([sepal_length, sepal_width, petal_length, petal_width]).reshape(1, -1)

# Tahmin butonu
if st.button("Tahmin Et"):
    prediction = model.predict(features)
    st.write(f"Tahmin edilen tür: {iris_data.target_names[prediction[0]]}")
