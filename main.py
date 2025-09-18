import streamlit as st 
import cv2 
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px 
import plotly.graph_objects as go
import pandas as pd
import sys
from PIL import Image
import io
st.set_page_config(layout='wide')

def binary_image(picture): 
    file_bytes = np.asarray(bytearray(picture.getvalue()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes,1)
    gray =  cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _,bin_image = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
    st.image(bin_image,use_container_width=True)
    return bin_image


def negative_image(picture):
    file_bytes = np.asarray(bytearray(picture.getvalue()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes,1)
    negative_img = 255 - image
    negative_img = cv2.cvtColor(negative_img, cv2.COLOR_BGR2RGB)
    st.image(negative_img,use_container_width=True)
    return negative_img


def plot_histogram_each_band(picture):
    file_bytes = np.asarray(bytearray(picture.getvalue()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    channels = cv2.split(img)
    colors = ('red', 'green', 'blue')

    figs = []
    for channel, col in zip(channels, colors):
        hist = cv2.calcHist([channel], [0], None, [256], [0, 256]).flatten()
        x = np.arange(256)
        y = hist
        fig = px.bar(x=x, y=y, title=f"Histograma {col.capitalize()}",labels={"y":"Frequencia","x":"Nivel de Intensidade"})
        fig.update_traces(marker_color=col,width=0.9)

        figs.append(fig)

    return figs

def plot_combined_histogram(picture):
    file_bytes = np.asarray(bytearray(picture.getvalue()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    channels = cv2.split(img)
    colors = ('red', 'green', 'blue')

    df = pd.DataFrame()
    for channel, col in zip(channels, colors):
        hist = cv2.calcHist([channel], [0], None, [256], [0, 256]).flatten()
        df[col] = hist

    df["intensity"] = np.arange(256)

    fig = px.line(
        df, 
        x="intensity", 
        y=df.columns[:-1], 
        title="Histograma Combinado",
        labels={"value": "Frequência", "intensity": "Intensidade"},
        color_discrete_map={"red": "red", "green": "green", "blue": "blue"}
    )

    fig.update_layout(
        xaxis=dict(title="Níveis de intensidade (0-255)"),
        yaxis=dict(title="Número de pixels"),
        legend_title="Canais"
    )

    return fig


    


st.title("OPENCV: Histograma, Imagem Negativa e Imagem Binarizada")

upload_method = st.selectbox(label="Escolha a forma de envio",options=['Tirar foto','Enviar arquivo'])
if upload_method == 'Tirar foto':
    picture = st.camera_input(label="Camera")
else:
    picture = st.file_uploader(label="Envia a sua",type=["jpg", "png"])
if picture:
    st.subheader("Imagens Processadas")
    img_col1, img_col2 = st.columns(2)
    with img_col1:
       bin_img = binary_image(picture)
       png_bin_img = Image.fromarray(bin_img)

       img_bytes = io.BytesIO()  # cria buffer vazio
       png_bin_img.save(img_bytes, format="PNG")
       img_bytes.seek(0)
       negative_down_btn = st.download_button(label="Baixar",data=img_bytes,file_name="binary_image.jpg",mime="image/png",key='bin_btn')
    with img_col2:
        neg_imag = negative_image(picture)
        png_neg_img = Image.fromarray(neg_imag)
        img_bytes = io.BytesIO()
        png_neg_img.save(img_bytes,format="PNG")
        img_bytes.seek(0)
        st.download_button(label="Baixar",data=img_bytes,file_name="negative_image.jpg",mime="image/png",key="neg_btn")

    st.subheader("Histogramas RGB")
    hist_col1, hist_col2, hist_col3 = st.columns(3)
    hist_red, hist_green, hist_blue = plot_histogram_each_band(picture)
    with hist_col1:
        st.plotly_chart(hist_red, use_container_width=True)
    with hist_col2:
        st.plotly_chart(hist_green, use_container_width=True)
    with hist_col3:
        st.plotly_chart(hist_blue, use_container_width=True)
    st.plotly_chart(plot_combined_histogram(picture=picture))