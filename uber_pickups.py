"""
# My first app
Here's our first attempt at using data to create a table:
"""

import streamlit as st
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline


st.image("logotip.png", width=400)

st.title('Model "Fill mask"')

inp = st.text_input('Введите текст на языке согласно представленному ниже образцу',
                    'I study economics at [MASK].')

unmasker = pipeline('fill-mask', model='albert-base-v2')

if inp:
    text = unmasker(inp)
    print(inp)
    print(text)
    st.write("Возможныe варианты ответа:")
    for el in text:
        print(el)
        st.write(el["sequence"])


# streamlit run C:/Users/Админ/PycharmProjects/fill-mask/uber_pickups.py
