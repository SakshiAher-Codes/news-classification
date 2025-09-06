import streamlit as st
import tempfile
import os

st.title("Deep Fake Detection Application")

st.sidebar.title("Navigation")
page = st.sidebar.radio("Select a page:", ("Home", "Upload Video", "Detection Results", "Feedback", "About"))

if page == "Home":
    st.header("Welcome to the Deep Fake Detection Application!")
    st.write(
        "This application uses advanced machine learning techniques to detect deep fake videos. "
        "Upload your video files and analyze them to see if they are real or fake."
    )

elif page == "Upload Video":
    st.header("Upload Video for Detection")
    uploaded_file = st.file_uploader("Choose a video file...", type=["mp4", "mov", "avi"])

    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        video_path = tfile.name
        st.video(video_path)

        if st.button("Analyze Video"):
            st.success("Processing your video...")
            result = detect_deep_fake(video_path)

            st.session_state["detection_result"] = result
            st.experimental_rerun()
