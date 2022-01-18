"""
# Our first app
"""

import streamlit as st
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline


st.image("logotip.png", width=400)

st.title('Model "Fill mask"')

inp = st.text_input('Введите текст на английском языке согласно представленному ниже образцу',
                    'I study economics at [MASK].')

unmasker = pipeline('fill-mask', model='albert-base-v2')

if inp:
    text = unmasker(inp)
    st.write("Возможныe варианты ответа:")
    for el in text:
        st.write(el["sequence"])
