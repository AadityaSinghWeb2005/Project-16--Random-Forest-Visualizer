import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Page Title
st.title("Random Forest Hyperparameter Tuning Tool")

# Sidebar for Hyperparameters
st.sidebar.header("Tuning Parameters")

n_estimators = st.sidebar.slider("Number of Trees (n_estimators)", 1, 100, 10)
max_samples = st.sidebar.slider("Max Samples (Bootstrap)", 0.1, 1.0, 0.5)
max_features = st.sidebar.slider("Max Features", 1, 2, 1)
bootstrap = st.sidebar.selectbox("Bootstrap", [True, False])

# Step 1: Generate Synthetic Data (Moons Dataset)
X, y = make_moons(n_samples=500, noise=0.3, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2: Initialize and Train Random Forest
rf = RandomForestClassifier(
    n_estimators=n_estimators,
    max_samples=max_samples if bootstrap else None,
    max_features=max_features,
    bootstrap=bootstrap,
    random_state=42
)
rf.fit(X_train, y_train)

# Step 3: Predict and Calculate Accuracy
y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Display Accuracy
st.write(f"### Model Accuracy: {accuracy:.2%}")

# Step 4: Plotting Decision Boundary
def plot_decision_boundary(X, y, model):
    h = .02  # mesh step size
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    fig, ax = plt.subplots()
    ax.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
    ax.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap='RdYlBu')
    ax.set_title(f"Trees: {n_estimators}, Samples: {max_samples}")
    return fig

# Show Plot
st.pyplot(plot_decision_boundary(X, y, rf))

st.info("Sliders ko move karke dekhiye kaise 'Number of Trees' badhane se boundary smooth hoti hai aur overfitting kam hoti hai!")