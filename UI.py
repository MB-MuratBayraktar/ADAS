import streamlit as st
import pandas as pd
import tempfile
import cv2
import sys, os


def main():
    names = []
    with open("names.txt") as f:
        for each_line in f:
            names.append(each_line)

    print(names[0])

    st.title('Multiple Object Tracking system')
    st.sidebar.title('Settings')
    st.markdown(
        """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child{width: 400px;}    
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child{width: 400px; margin-left: -400px}    
    </style>
    """,
        unsafe_allow_html=True)

    confidence = st.sidebar.slider('Confidence', min_value=0.0, max_value=1.0, value=0.5)
    st.sidebar.markdown('---')
    save_img = st.sidebar.checkbox('Save Video')
    save_img = st.sidebar.checkbox('Enable GPU')
    custom_classes = st.sidebar.checkbox('Use custom classes')
    st.sidebar.markdown('---')
    assigned_class_id = []

    if custom_classes:
        assigned_class = st.sidebar.multiselect('Select the custom classes', list(names), default=names[0])
        for each in assigned_class:
            assigned_class_id.append(names.index(each))

    video_file_buffer = st.sidebar.file_uploader('Upload your video here', type=["mp4", "mov", "avi", "asf", "m4v"])
    demo_video = './test media/dashcam_3.mp4'
    tfflie = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)

    if not video_file_buffer:
        vid = cv2.VideoCapture(demo_video)
        tfflie.name = demo_video
        dem_vid = open(tfflie.name, 'rb')
        demo_byte = dem_vid.read()
        st.sidebar.text('Input Video')
        st.sidebar.video(demo_byte)
    else:
        tfflie.write(video_file_buffer.read())
        dem_vid = open(tfflie.name, 'rb')
        demo_byte = dem_vid.read()
        st.sidebar.text("Input Video")
        st.sidebar.video(demo_byte)

    print(tfflie.name)
    stframe = st.empty()
    st.sidebar.markdown('---')

    kpi1, kpi2, kpi3 = st.columns(3)

    with kpi1:
        st.markdown("*Frame rate*")
        kpi1_text = st.markdown("0")

    with kpi2:
        st.markdown("*Tracked Objects*")
        kpi1_text = st.markdown("0")

    with kpi3:
        st.markdown("*Width*")
        kpi1_text = st.markdown("0")


if __name__ == '__main__':
    try:
        main()
    except SystemExit:
        pass
