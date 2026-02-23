import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import streamlit as st it

def run_weather_prediction():
    st.title("Weather Prediction App") 
    
    X = np.array([[30, 70], [25, 80], [27, 60], [31, 65], [23, 85], [28, 75]])
    y = np.array([0, 1, 0, 0, 1, 1]) 

    
    k_value = 3
    knn = KNeighborsClassifier(n_neighbors=k_value)
    knn.fit(X, y)

    new_point = np.array([[26, 78]])
    prediction = knn.predict(new_point)[0]

    fig, ax = plt.subplots(figsize=(8, 6)) t

    ax.scatter(X[y == 0, 0], X[y == 0, 1], label="Sunny (0)", s=100, edgecolors='k')
    ax.scatter(X[y == 1, 0], X[y == 1, 1], label="Rainy (1)", s=100, edgecolors='k')

    ax.scatter(
        new_point[0, 0],
        new_point[0, 1],
        marker="*",
        s=300,
        color="red",
        label=f"New Point (Pred: {'Rainy' if prediction == 1 else 'Sunny'})",
    )

    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Humidity (%)")
    ax.set_title(f"KNN Weather Classification (k={k_value})")
    ax.legend()
    ax.grid(alpha=0.3)
    
    st.write(f"### Predicted Weather: {'Rainy 🌧️' if prediction == 1 else 'Sunny ☀️'}")
    st.pyplot(fig) 

if __name__ == "__main__":
    run_weather_prediction()
