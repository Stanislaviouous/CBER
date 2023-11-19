import io
import streamlit as st
import numpy as np
import pandas as pd
import time
from PIL import Image
import pickle 
import random

st.markdown("""<style>
    .stProgress .st-bk {
        background-color: red;
    }
    </style>""", 
    unsafe_allow_html=True)
# @st.cache(allow_output_mutation=True)
# def load_model():
#     return EfficientNetB0(weights='imagenet')

def load_image(gender):
    gen = ''
    if (gender == 1): gen = 'm'
    image = Image.open(f'people/{gen}{int(random.random() * 24) + 1}.jpg')
    st.image(image)
    return image
    # filew = f"https://xsgames.co/randomusers/assets/avatars/{gender}/{int(random.random() * 79)}.jpg"
    # print(filew)
    # st.image("https://cdn.britannica.com/q:60/49/161649-050-3F458ECF/Bernese-mountain-dog-grass.jpg")
    # st.image('https://xsgames.co/randomusers/assets/avatars/male/75.jpg')

def load_df():
    uploaded_file = st.file_uploader(label='Выберите file',)
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write(df)
        return df
    else:
        return None

def predicto(model, df):
    # y_pred = model.predict(xgb.DMatrix(df.values))
    print(int(random.random() + 0.5))
    return int(random.random() + 0.5)

def load_progress():
    progress_text = 'работают машины'
    progress = st.progress(0, text = progress_text)
    for i in range(100):
        progress.progress(i)
        time.sleep(0.05)
    time.sleep(1)
    progress.empty()

def load_model():
    pickle_in = open('model.pkl', 'rb') 
    # return pickle.load(pickle_in) 
    return ''
model = load_model()

# def print_predictions(preds):
#     classes = decode_predictions(preds, top=3)[0]
#     for cl in classes:
#         st.write(cl[1], cl[2])
# model = load_model()


st.title("""Кто вы? 
         По вашим покупкам""")
df = load_df()

result = st.button('Узнать')
if (result and df is not None):
    progress = load_progress()
    predict = predicto(model, df)
    image = load_image(predict)
time.sleep(0.5)



# result = st.button('Распознать изображение')
# if result:
#     x = preprocess_image(img)
#     preds = model.predict(x)
#     st.write('**Результаты распознавания:**')
#     # print_predictions(preds)