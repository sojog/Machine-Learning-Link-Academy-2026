# pip install streamlit
import streamlit as st
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris

with open("iris_model.pkl", "rb") as file_reader:
    model:KNeighborsClassifier = pickle.load(file_reader)

iris_cat = load_iris().target_names

st.title("Folosirea unui model intr-un website streamlit")


sepal_length = st.number_input("Introduceti lungimea sepalei")
sepal_width = st.number_input("Introduceti lățimea sepalei")
petal_length = st.number_input("Introduceti lungimea petalei")
petal_width = st.number_input("Introduceti lățimea petalei")

check_button = st.button("Verifica categoria")

if check_button:
    clasa_prezisa = model.predict([[sepal_length, sepal_width,	petal_length,	petal_width]])
    st.write("Clasa prezisa este:", clasa_prezisa, iris_cat[clasa_prezisa])

## Rularea site-ului este: streamlit run nume_fisier