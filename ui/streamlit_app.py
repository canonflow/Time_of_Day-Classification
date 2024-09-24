import streamlit as st
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import joblib

st.set_page_config(layout="wide")
css = """
<style>
    h2{
        text-align: center;
    }
</style>
"""
st.markdown(css, unsafe_allow_html=True)

st.header("Prediksi Waktu Hari Berdasarkan Gambar Langit")
upload = st.file_uploader("Masukkan gambar untuk diprediksi!", type=['png', 'jpg'])
c1, c2 = st.columns(2)

if upload is not None:
    im = Image.open(upload)  # Buka Image dari upload
    im = np.asarray(im)  # Ubah Image ke numpy (soalnya PIL gk pake array-nya numpy)
    im = cv2.resize(im, (244,244), interpolation=cv2.INTER_AREA)
    im = im / 255   

    with c1:
        c3,c4,c5 = st.columns(3)
        with c4:
            st.header("IMAGE")
            st.image(im)    

    img_pred = Image.open(upload)  # Buka Image dari upload
    img_pred = np.asarray(img_pred)  # Ubah Image ke numpy (soalnya PIL gk pake array-nya numpy)
    img_pred = cv2.resize(img_pred, (128, 128), interpolation=cv2.INTER_AREA)
    img_pred = img_pred / 255
    img_pred = img_pred.flatten()
    SVM_classifier = joblib.load("svm_model.pkl")
    KNN_classifier = joblib.load("knn_model.pkl")
    SVM_predicted_class = SVM_classifier.predict([img_pred])
    KNN_predicted_class = KNN_classifier.predict([img_pred])
    c2.header("  Class:")
    my_dataframe = pd.DataFrame({"SVM":[SVM_predicted_class[0]], 
                                 "KNN":[KNN_predicted_class[0]]}, 
                                 index = ["Predicted Output : "])
    
    with c2:
        st.write(f"""<div style="display: flex; justify-content: center;">{my_dataframe.to_html(index=True)}</div>""", unsafe_allow_html=True)

    
