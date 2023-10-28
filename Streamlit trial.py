import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris

# Load the Iris dataset
iris = load_iris()
data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
data["target"] = iris.target
data["target_names"] = iris.target_names[data["target"]]

# Streamlit app
st.title("Iris Dataset Explorer")

# Sidebar with class selection
selected_class = st.sidebar.selectbox("Select Iris Class", data["target_names"].unique())

# Display the selected class data
st.write(f"Showing data for Iris class: {selected_class}")
selected_data = data[data["target_names"] == selected_class]
st.write(selected_data)

# Display a histogram of the selected class
st.subheader("Histogram of Sepal Length")
st.hist(selected_data["sepal length (cm)"], bins=20)

st.subheader("Histogram of Sepal Width")
st.hist(selected_data["sepal width (cm)"], bins=20)

st.subheader("Histogram of Petal Length")
st.hist(selected_data["petal length (cm)"], bins=20)

st.subheader("Histogram of Petal Width")
st.hist(selected_data["petal width (cm)"], bins=20)
