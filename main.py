import streamlit as st 
import cv2 
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px 
import pandas as pd
import sys

st.set_page_config(layout='wide')

def binary_image(picture): 
    file_bytes = np.asarray(bytearray(picture.getvalue()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes,1)
    gray =  cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _,bin_image = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
    st.image(bin_image,use_container_width=True)


def negative_image(picture):
    file_bytes = np.asarray(bytearray(picture.getvalue()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes,1)
    negative_img = 255 - image
    negative_img = cv2.cvtColor(negative_img, cv2.COLOR_BGR2RGB)
    st.image(negative_img,use_container_width=True)


def plot_histogram_each_band(picture):
    # Lê os bytes da imagem
    file_bytes = np.asarray(bytearray(picture.getvalue()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    # Converte BGR → RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Separa os canais RGB
    channels = cv2.split(img)
    colors = ('red', 'green', 'blue')

    figs = []
    for channel, col in zip(channels, colors):
        # Histograma do canal
        hist = cv2.calcHist([channel], [0], None, [256], [0, 256]).flatten()
        x = np.arange(256)
        y = hist

        # Gráfico de barras
        fig = px.bar(x=x, y=y, title=f"Histograma {col.capitalize()}")
        fig.update_traces(marker_color=col)

        figs.append(fig)

    return figs




st.title("TOPENCV: Histograma, Imagem Negativa e Imagem Binarizada")

# Escolhe entre câmera local ou upload de arquivo
enable_camera = st.checkbox("Enable camera (Local only)")

if enable_camera and ("localhost" in st.get_option("server.address") or sys.platform.startswith("win")):
    picture = st.camera_input("Take a picture")
else:
    picture = st.file_uploader("Upload an image", type=["jpg","png","jpeg"])
if picture:
    st.subheader("Imagens Processadas")
    img_col1, img_col2 = st.columns(2)
    with img_col1:
        binary_image(picture)
    with img_col2:
        negative_image(picture)

    st.subheader("Histogramas RGB")
    hist_col1, hist_col2, hist_col3 = st.columns(3)
    hist_red, hist_green, hist_blue = plot_histogram_each_band(picture)
    with hist_col1:
        st.plotly_chart(hist_red, use_container_width=True)
    with hist_col2:
        st.plotly_chart(hist_green, use_container_width=True)
    with hist_col3:
        st.plotly_chart(hist_blue, use_container_width=True)
