import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

def run_weather_prediction():
    # ---------------- Data ----------------
    # Features: [Temperature, Humidity]
    X = np.array([[30, 70], [25, 80], [27, 60], [31, 65], [23, 85], [28, 75]])
    y = np.array([0, 1, 0, 0, 1, 1])  # 0 = Sunny, 1 = Rainy

    # ---------------- KNN Model ----------------
    k_value = 3
    knn = KNeighborsClassifier(n_neighbors=k_value)
    knn.fit(X, y)

    # New point to predict
    new_point = np.array([[26, 78]])
    prediction = knn.predict(new_point)[0]

    # ---------------- Visualization ----------------
    plt.figure(figsize=(8, 6))

    # Plot existing data
    plt.scatter(X[y == 0, 0], X[y == 0, 1], label="Sunny (0)", s=100, edgecolors='k')
    plt.scatter(X[y == 1, 0], X[y == 1, 1], label="Rainy (1)", s=100, edgecolors='k')

    # Plot the new point
    plt.scatter(
        new_point[0, 0],
        new_point[0, 1],
        marker="*",
        s=300,
        color="red",
        label=f"New Point (Pred: {'Rainy' if prediction == 1 else 'Sunny'})",
    )

    plt.xlabel("Temperature (°C)")
    plt.ylabel("Humidity (%)")
    plt.title(f"KNN Weather Classification (k={k_value})")
    plt.legend()
    plt.grid(alpha=0.3)
    
    print(f"Predicted Weather: {'Rainy 🌧️' if prediction == 1 else 'Sunny ☀️'}")
    plt.show()

if __name__ == "__main__":
    run_weather_prediction()
