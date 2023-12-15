import streamlit as st
import numpy as np
from PIL import Image
import cv2

st.markdown("""<style>
    .stProgress .st-bk {
        background-color: red;
    }
    </style>""", 
    unsafe_allow_html=True)

def load():
    uploaded = st.file_uploader(label='Выберите фото',)
    if uploaded is not None:
        img = Image.open(uploaded)
        cv2Image = np.array(img)
        # st.image(cv2Image)
        return cv2Image
    else:
        return None
    
colors = {
    "red" : 0, 
    "green": 1, 
    "blue": 2
}

st.title("""Коректируй фото""")
img = load()

add_selectbox = st.sidebar.selectbox(
    "Какой цветовой компонент?",
    ("red", "green", "blue")
)

with st.sidebar:
    comp = st.slider('Выбери n', 0, 255, 1)

if (comp and add_selectbox is not None and img is not None):
    n = img.shape[1]
    m = img.shape[0]
    omage = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    omage[:, :, colors[add_selectbox]] +=  comp
    newImage = cv2.cvtColor(omage, cv2.COLOR_HSV2BGR)
    st.image(newImage)

# def load_image(gender):
#     gen = ''
#     if (gender == 1): gen = 'm'
#     image = Image.open(f'people/{gen}{int(random.random() * 24) + 1}.jpg')
#     st.image(image)
#     return image
    # filew = f"https://xsgames.co/randomusers/assets/avatars/{gender}/{int(random.random() * 79)}.jpg"
    # print(filew)
    # st.image("https://cdn.britannica.com/q:60/49/161649-050-3F458ECF/Bernese-mountain-dog-grass.jpg")
    # st.image('https://xsgames.co/randomusers/assets/avatars/male/75.jpg')

# def predicto(model, df):
#     # y_pred = model.predict(xgb.DMatrix(df.values))
#     print(int(random.random() + 0.5))
#     return int(random.random() + 0.5)

# def load_progress():
#     progress_text = 'работают машины'
#     progress = st.progress(0, text = progress_text)
#     fr i in range(100):
#         progress.progress(i)
#         time.sleep(0.05)
#     time.sleep(1)
#     progress.empty()

# def load_model():
#     pickle_in = open('model.pkl', 'rb') 
#     return pickle.load(pickle_in) 
#     return ''

# st.write("I'm ", comp, 'years old')
# st.write(add_selectbox)

# model = load_model()

# def print_predictions(preds):
#     classes = decode_predictions(preds, top=3)[0]
#     for cl in classes:
#         st.write(cl[1], cl[2])
# model = load_model()

# result = st.button('Узнать')
# if (result and df is not None):
#     progress = load_progress()
#     predict = predicto(model, df)
#     image = load_image(predict)
# time.sleep(0.5)


# result = st.button('Распознать изображение')
# if result:
#     x = preprocess_image(img)
#     preds = model.predict(x)
#     st.write('**Результаты распознавания:**')
#     # print_predictions(preds)