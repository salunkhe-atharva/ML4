import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import streamlit as st  # <--- 1. Import streamlit

def run_weather_prediction():
    st.title("Weather Prediction App") # <--- 2. Add a title for the web page
    
    # ---------------- Data ----------------
    X = np.array([[30, 70], [25, 80], [27, 60], [31, 65], [23, 85], [28, 75]])
    y = np.array([0, 1, 0, 0, 1, 1])  # 0 = Sunny, 1 = Rainy

    # ---------------- KNN Model ----------------
    k_value = 3
    knn = KNeighborsClassifier(n_neighbors=k_value)
    knn.fit(X, y)

    new_point = np.array([[26, 78]])
    prediction = knn.predict(new_point)[0]

    # ---------------- Visualization ----------------
    fig, ax = plt.subplots(figsize=(8, 6)) # <--- 3. Create a figure object

    # Plot existing data (using 'ax' instead of 'plt' is best practice in Streamlit)
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
    
    # ---------------- Streamlit Output ----------------
    st.write(f"### Predicted Weather: {'Rainy 🌧️' if prediction == 1 else 'Sunny ☀️'}")
    st.pyplot(fig) # <--- 4. Pass the figure to Streamlit

if __name__ == "__main__":
    run_weather_prediction()
