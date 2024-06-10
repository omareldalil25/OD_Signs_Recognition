# app.py

# External packages
import streamlit as st
# Python In-built packages
from pathlib import Path
import PIL
# Local Modules
import settings
import helper

# Setting page layout
st.set_page_config(
    page_title="Sign language recognition using YOLOv8",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
custom_css = """
<style>
.st-emotion-cache-1gnzxwn {
    position: absolute;
    background: #000;
    color: #fff;
    inset: 0px;
    color-scheme: dark;
    overflow: hidden;
}

.st-emotion-cache-6qob1r {
    position: relative;
    height: 100%;
    width: 100%;
    overflow: overlay;
    background-color: #FF0080;
}

.st-emotion-cache-xpqsr {
    position: fixed;
    top: 0px;
    left: 0px;
    right: 0px;
    height: 2.875rem;
    background: #FFF7FC;
    outline: none;
    z-index: 999990;
    display: block;
    color: #000;
}

.st-emotion-cache-4y6tnn {
    display: inline-flex;
    -webkit-box-align: center;
    align-items: center;
    -webkit-box-pack: center;
    justify-content: center;
    font-weight: 400;
    border-radius: 0.5rem;
    margin: 0px 0.125rem;
    color: inherit;
    width: auto;
    user-select: none;
    background-color: transparent;
    border: none;
    font-size: 14px;
    line-height: 1;
    min-width: 2rem;
    min-height: 2rem;
    padding: 0px;
    background-color: #000;
    color: #fff;
}
.st-emotion-cache-10y5sf6 {
    font-family: "Source Code Pro", monospace;
    font-size: 14px;
    padding-bottom: 9.33333px;
    color: #fff;
    top: -22px;
    position: absolute;
    white-space: nowrap;
    background-color: transparent;
    line-height: 1.6;
    font-weight: normal;
}
.st-emotion-cache-1vzeuhh {
    -webkit-box-align: center;
    align-items: center;
    background-color: #fff;
    border-radius: 100%;
    border-style: none;
    box-shadow: none;
    display: flex;
    height: 0.75rem;
    -webkit-box-pack: center;
    justify-content: center;
    width: 0.75rem;
}
.st-b8 {
    background-color: #000;
}
.st-bq {
    background-color: #fff;
}
.st-ar {
    background: linear-gradient(to right, 
    rgb(21, 21, 21) 0%, 
    rgb(21, 21, 21) 20%, 
    rgb(255, 247, 252) 20%, 
    rgb(255, 247, 252) 100%);
}
.st-ce {
    background: linear-gradient(to right, 
    rgb(21, 21, 21) 0%, 
    rgb(21, 21, 21) 52%, 
    rgb(255, 247, 252)) 52%, 
    rgb(255, 247, 252)) 100%);
}
.st-dj {
    background: linear-gradient(to right, 
    rgb(21, 21, 21) 0%, 
    rgb(21, 21, 21) 77.3333%, 
    rgb(255, 247, 252) 77.3333%, 
    rgb(255, 247, 252) 100%);
}
.st-emotion-cache-1lra3nu {
    display: inline-flex;
    -webkit-box-align: center;
    align-items: center;
    -webkit-box-pack: center;
    justify-content: center;
    font-weight: 400;
    padding: 0.25rem 0.75rem;
    border-radius: 0.5rem;
    min-height: 38.4px;
    margin: 0px;
    line-height: 1.6;
    color: inherit;
    width: auto;
    user-select: none;
    background-color: #000;
    border: 1px solid rgba(255, 255, 255, 0.2);
}
.st-emotion-cache-1yc9733 {
    display: flex;
    -webkit-box-align: center;
    align-items: center;
    padding: 1rem;
    background-color: #000;
    border-radius: 0.5rem;
    border-color:#fff;
    color: rgb(255, 255, 255);
}
.st-c9 {
    background-color: #000;
}
}
</style>
"""

# Add custom CSS to Streamlit app
st.markdown(custom_css, unsafe_allow_html=True)

# Main page heading
st.title("Sign language recognition using YOLOv8")

confidence = float(st.sidebar.slider(
    "Select Model Confidence", 25, 100, 40)) / 100

model_path = Path(settings.DETECTION_MODEL)
# Load Pre-trained DL Model
try:
    model = helper.load_model(model_path)
except Exception as ex:
    st.error(f"Unable to load model. Check the specified path: {model_path}")
    st.error(ex)

st.sidebar.header("Image/Video Config")
source_radio = st.sidebar.radio(
    "Select Source", settings.SOURCES_LIST)

source_img = None
# If image is selected
if source_radio == settings.IMAGE:
    source_img = st.sidebar.file_uploader(
        "Choose an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))

    col1, col2 = st.columns(2)

    with col1:
        try:
            if source_img is None:
                default_image_path = str(settings.DEFAULT_IMAGE)
                default_image = PIL.Image.open(default_image_path)
                st.image(default_image_path, caption="Default Image",
                         use_column_width=True)
            else:
                uploaded_image = PIL.Image.open(source_img)
                st.image(source_img, caption="Uploaded Image",
                         use_column_width=True)
        except Exception as ex:
            st.error("Error occurred while opening the image.")
            st.error(ex)

    with col2:
        if source_img is None:
            default_detected_image_path = str(settings.DEFAULT_DETECT_IMAGE)
            default_detected_image = PIL.Image.open(
                default_detected_image_path)
            st.image(default_detected_image_path, caption='Detected Image',
                     use_column_width=True)
        else:
            if st.sidebar.button('Detect Objects'):
                res = model.predict(uploaded_image,
                                    conf=confidence
                                    )
                boxes = res[0].boxes
                res_plotted = res[0].plot()[:, :, ::-1]
                st.image(res_plotted, caption='Detected Image',
                         use_column_width=True)
                try:
                    with st.expander("Detection Results"):
                        for box in boxes:
                            st.write(box.data)
                except Exception as ex:
                    # st.write(ex)
                    st.write("No image is uploaded yet!")

elif source_radio == settings.VIDEO:
    helper.play_stored_video(confidence, model)

elif source_radio == settings.WEBCAM:
    helper.play_webcam(confidence, model)

else:
    st.error("Please select a valid source type!")
