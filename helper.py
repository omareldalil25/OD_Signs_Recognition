from ultralytics import YOLO
import streamlit as st
import cv2
import time
import settings
from pathlib import Path
import PIL
from gtts import gTTS
import tempfile

def load_model(model_path):
    """
    Loads a YOLO object detection model from the specified model_path.

    Parameters:
        model_path (str): The path to the YOLO model file.

    Returns:
        A YOLO object detection model.
    """
    model = YOLO(model_path)
    return model

def display_tracker_options():
    display_tracker = st.radio("Display Tracker", ('Yes', 'No'))
    is_display_tracker = True if display_tracker == 'Yes' else False
    if is_display_tracker:
        tracker_type = st.radio("Tracker", ("bytetrack.yaml", "botsort.yaml"))
        return is_display_tracker, tracker_type
    return is_display_tracker, None

def text_to_speech(text):
    """
    Converts the given text to speech and plays it.

    Parameters:
        text (str): The text to convert to speech.

    Returns:
        None
    """
    tts = gTTS(text)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio_file:
        tts.save(temp_audio_file.name)
        st.audio(temp_audio_file.name, format='audio/mp3')

def _display_detected_frames(conf, model, st_frame, st_text, image, is_display_tracking=None, tracker=None, enable_tts=False):
    """
    Display the detected objects on a video frame using the YOLOv8 model.

    Args:
    - conf (float): Confidence threshold for object detection.
    - model (YoloV8): A YOLOv8 object detection model.
    - st_frame (Streamlit object): A Streamlit object to display the detected video.
    - st_text (Streamlit object): A Streamlit object to display the detected sentence.
    - image (numpy array): A numpy array representing the video frame.
    - is_display_tracking (bool): A flag indicating whether to display object tracking (default=None).
    - enable_tts (bool): A flag indicating whether to enable text-to-speech for the detected sentence.

    Returns:
    None
    """

    # Resize the image to a standard size
    image = cv2.resize(image, (720, int(720*(9/16))))

    # Display object tracking, if specified
    if is_display_tracking:
        res = model.track(image, conf=conf, persist=True, tracker=tracker)
    else:
        # Predict the objects in the image using the YOLOv8 model
        res = model.predict(image, conf=conf)

    # Plot the detected objects on the video frame
    res_plotted = res[0].plot()
    st_frame.image(res_plotted,
                   caption='Detected Video',
                   channels="BGR",
                   use_column_width=True
                   )

    # Extract detected labels and update the detected sentence
    detected_labels = [model.names[int(box.cls)] for box in res[0].boxes]

    # Initialize session state variables if not already set
    if 'detected_sentence' not in st.session_state:
        st.session_state.detected_sentence = ""
    if 'previous_label' not in st.session_state:
        st.session_state.previous_label = ""
    if 'label_start_time' not in st.session_state:
        st.session_state.label_start_time = None
    if 'previous_sentence' not in st.session_state:
        st.session_state.previous_sentence = ""

    # Update the detected sentence based on the new label
    if detected_labels:
        current_label = detected_labels[0]
        if current_label == st.session_state.previous_label:
            # If the label is the same as the previous one, check the time duration
            if st.session_state.label_start_time is not None:
                elapsed_time = time.time() - st.session_state.label_start_time
                if elapsed_time >= 3:  # Check if 3 seconds have passed
                    # Always append the current label
                    st.session_state.detected_sentence += current_label
                    st.session_state.previous_label = current_label
                    st.session_state.label_start_time = None  # Reset the start time
        else:
            # If the label is different, start a new timer
            st.session_state.previous_label = current_label
            st.session_state.label_start_time = time.time()
    else:
        st.session_state.previous_label = ""
        st.session_state.label_start_time = None

    # Display the detected sentence
    st_text.markdown(f"**Detected Sentence : {st.session_state.detected_sentence}**")

    # Convert the detected sentence to speech if enabled and if it has changed
    if enable_tts and st.session_state.detected_sentence != st.session_state.previous_sentence:
        text_to_speech(st.session_state.detected_sentence)
        st.session_state.previous_sentence = st.session_state.detected_sentence

def play_webcam(conf, model):
    """
    Plays a webcam stream. Detects Objects in real-time using the YOLOv8 object detection model.

    Parameters:
        conf: Confidence of YOLOv8 model.
        model: An instance of the `YOLOv8` class containing the YOLOv8 model.

    Returns:
        None

    Raises:
        None
    """
    source_webcam = settings.WEBCAM_PATH
    is_display_tracker, tracker = display_tracker_options()
    enable_tts = st.sidebar.checkbox("Enable Text-to-Speech")
    if st.sidebar.button('Detect webcam signs'):
        try:
            vid_cap = cv2.VideoCapture(source_webcam)
            st_frame = st.empty()
            st_text = st.empty()
            while (vid_cap.isOpened()):
                success, image = vid_cap.read()
                image = cv2.flip(image, 1)
                if success:
                    _display_detected_frames(conf,
                                             model,
                                             st_frame,
                                             st_text,
                                             image,
                                             is_display_tracker,
                                             tracker,
                                             enable_tts
                                             )
                else:
                    vid_cap.release()
                    break
        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))

def play_stored_video(conf, model):
    """
    Plays a stored video file. Tracks and detects objects in real-time using the YOLOv8 object detection model.

    Parameters:
        conf: Confidence of YOLOv8 model.
        model: An instance of the `YOLOv8` class containing the YOLOv8 model.

    Returns:
        None

    Raises:
        None
    """
    source_vid = st.sidebar.selectbox(
        "Choose a video...", settings.VIDEOS_DICT.keys())

    is_display_tracker, tracker = display_tracker_options()
    enable_tts = st.sidebar.checkbox("Enable Text-to-Speech")

    with open(settings.VIDEOS_DICT.get(source_vid), 'rb') as video_file:
        video_bytes = video_file.read()
    if video_bytes:
        st.video(video_bytes)

    if st.sidebar.button('Detect Video Signs'):
        try:
            vid_cap = cv2.VideoCapture(
                str(settings.VIDEOS_DICT.get(source_vid)))
            st_frame = st.empty()
            st_text = st.empty()
            while (vid_cap.isOpened()):
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(conf,
                                             model,
                                             st_frame,
                                             st_text,
                                             image,
                                             is_display_tracker,
                                             tracker,
                                             enable_tts
                                             )
                else:
                    vid_cap.release()
                    break
        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))
        cv2.waitKey(0)

# Test code or main function
if __name__ == "__main__":
    st.set_page_config(page_title="Sign Language Recognition using YOLOv8")
    st.title("Sign Language Recognition using YOLOv8")

    confidence = float(st.sidebar.slider(
        "Select Model Confidence", 25, 100, 40)) / 100

    model_path = Path(settings.DETECTION_MODEL)
    try:
        model = load_model(model_path)
    except Exception as ex:
        st.error(f"Unable to load model. Check the specified path: {model_path}")
        st.error(ex)

    st.sidebar.header("Image/Video Config")
    source_radio = st.sidebar.radio(
        "Select Source", settings.SOURCES_LIST)

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
                        st.write("No image is uploaded yet!")

    elif source_radio == settings.VIDEO:
        play_stored_video(confidence, model)

    elif source_radio == settings.WEBCAM:
        play_webcam(confidence, model)

    else:
        st.error("Please select a valid source type!")


