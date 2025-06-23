import streamlit as st
import cv2
import torch
import numpy as np
import time
from PIL import Image
from gpiozero import Device, LED, Buzzer
from threading import Thread
import atexit
from picamera2 import Picamera2

class VideoStream:
    def __init__(self, src=0, resolution=(640, 480)):
        self.picam2 = None
        self.resolution = resolution
        self.stopped = False
        self.frame = None
        self._initialize_camera()

    def _initialize_camera(self):
        try:
            self.picam2 = Picamera2()
            camera_config = self.picam2.create_preview_configuration(
                main={"format": 'RGB888', "size": self.resolution}
            )
            self.picam2.configure(camera_config)
            self.picam2.set_controls({
                "AwbEnable": True,
                "AwbMode": 3,
                "ColourGains": (1.4, 1.5)
            })
            self.picam2.start()
            time.sleep(3)
            print("PiCamera2 initialized successfully")
        except Exception as e:
            print(f"Camera initialization error: {e}")
            if self.picam2:
                self.picam2.close()
            raise

    def start(self):
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        while not self.stopped:
            try:
                if self.picam2:
                    frame = self.picam2.capture_array()
                    # Convert RGB to BGR for OpenCV processing
                    #self.frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    self.frame = frame
                time.sleep(0.01)
            except Exception as e:
                print(f"Frame capture error: {e}")
                self.stopped = True
                break

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        if self.picam2:
            try:
                self.picam2.stop()
                self.picam2.close()
                print("PiCamera2 stopped successfully")
            except:
                pass
            self.picam2 = None

st.set_page_config(
    page_title="Smart Imaging Dashboard",
    #page_icon="🎥",
    layout="wide"
)

def initialize_session_state():
    """Initialize session state variables"""
    if 'gpio_initialized' not in st.session_state:
        st.session_state.gpio_initialized = False
        st.session_state.buzzer = None
        st.session_state.led = None
        st.session_state.gpio_error = None

    if 'camera_initialized' not in st.session_state:
        st.session_state.camera_initialized = False
        st.session_state.video_stream = None

    if 'people_counter' not in st.session_state:
        st.session_state.people_counter = {'count': 0}

def setup_gpio():
    """Setup GPIO pins only if not already initialized"""
    if st.session_state.gpio_initialized:
        return st.session_state.buzzer, st.session_state.led

    try:
        buzzer = Buzzer(18)
        led = LED(23)

        st.session_state.buzzer = buzzer
        st.session_state.led = led
        st.session_state.gpio_initialized = True
        st.session_state.gpio_error = None

        atexit.register(cleanup_gpio)

        return buzzer, led

    except Exception as e:
        st.session_state.gpio_error = str(e)
        st.session_state.gpio_initialized = False
        return None, None

def cleanup_gpio():
    """Cleanup GPIO resources"""
    try:
        if st.session_state.get('buzzer') is not None:
            st.session_state.buzzer.close()
        if st.session_state.get('led') is not None:
            st.session_state.led.close()
    except:
        pass
    finally:
        st.session_state.gpio_initialized = False
        st.session_state.buzzer = None
        st.session_state.led = None

def trigger_alarm(buzzer_pin, led_pin, duration=1):
    try:
        buzzer_pin.on()
        led_pin.on()
        time.sleep(duration)
    finally:
        buzzer_pin.off()
        led_pin.off()

@st.cache_resource
def load_model():
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    model.classes = [0]  # Filter for people only (class 0)
    return model

def process_frame(frame, model, restricted_area, people_counter):
    if frame is None:
        return None, False

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(rgb_frame)

    detections = results.pandas().xyxy[0]

    people_detections = detections[detections['class'] == 0]

    people_count = len(people_detections)
    people_counter['count'] = people_count

    intrusion_detected = False

    cv2.rectangle(frame, (restricted_area[0], restricted_area[1]),
                 (restricted_area[2], restricted_area[3]), (0, 0, 255), 2)

    cv2.putText(frame, "RESTRICTED AREA", (restricted_area[0], restricted_area[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    for _, detection in people_detections.iterrows():
        x1, y1, x2, y2 = int(detection['xmin']), int(detection['ymin']), int(detection['xmax']), int(detection['ymax'])

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        conf = detection['confidence']
        label = f"Person: {conf:.2f}"
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        person_center_x = (x1 + x2) // 2
        person_center_y = (y1 + y2) // 2

        if (restricted_area[0] < person_center_x < restricted_area[2] and
            restricted_area[1] < person_center_y < restricted_area[3]):
            intrusion_detected = True

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(frame, "INTRUSION!", (x1, y1 - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.putText(frame, f"People Count: {people_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(frame, timestamp, (10, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return frame, intrusion_detected

def alarm_thread_function(buzzer_pin, led_pin):
    trigger_alarm(buzzer_pin, led_pin, 2)

def initialize_camera(resolution):
    """Initialize camera only if not already done or if resolution changed"""
    if (not st.session_state.camera_initialized or
        st.session_state.get('current_resolution') != resolution):

        if st.session_state.video_stream is not None:
            try:
                st.session_state.video_stream.stop()
            except:
                pass

        try:
            video_stream = VideoStream(src=0, resolution=resolution).start()
            time.sleep(3.0)

            st.session_state.video_stream = video_stream
            st.session_state.camera_initialized = True
            st.session_state.current_resolution = resolution
            return video_stream, True

        except Exception as e:
            st.session_state.camera_initialized = False
            return None, False

    return st.session_state.video_stream, True

def main():
    initialize_session_state()

    st.title("🎥 Smart Intrusion Detection System")
    st.markdown("**Real-time person detection with restricted area monitoring**")

    st.sidebar.title("⚙️ Settings")

    st.sidebar.subheader("Camera Configuration")
    st.sidebar.info("Using Raspberry Pi Camera with PiCamera2")

    resolution_options = {
        "640x480": (640, 480),
        "800x600": (800, 600),
        "1024x768": (1024, 768)
    }
    selected_resolution = st.sidebar.selectbox(
        "Select Resolution",
        options=list(resolution_options.keys()),
        index=0
    )
    resolution = resolution_options[selected_resolution]

    st.sidebar.subheader("🚫 Restricted Area Settings")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        x1 = st.number_input("X1 (Top-Left)", value=200, min_value=0, max_value=resolution[0])
        y1 = st.number_input("Y1 (Top-Left)", value=150, min_value=0, max_value=resolution[1])
    with col2:
        x2 = st.number_input("X2 (Bottom-Right)", value=450, min_value=0, max_value=resolution[0])
        y2 = st.number_input("Y2 (Bottom-Right)", value=350, min_value=0, max_value=resolution[1])

    restricted_area = [x1, y1, x2, y2]

    st.sidebar.subheader("🚨 Alarm Configuration")
    alarm_enabled = st.sidebar.checkbox("Enable Alarm (LED + Buzzer)", value=True)

    confidence_threshold = st.sidebar.slider(
        "Detection Confidence Threshold",
        min_value=0.1,
        max_value=1.0,
        value=0.5,
        step=0.1
    )

    with st.spinner("🔄 Loading YOLOv5 model..."):
        model = load_model()
        model.conf = confidence_threshold

    st.sidebar.subheader("🔌 GPIO Status")
    if st.session_state.gpio_initialized:
        st.sidebar.success("✅ GPIO initialized successfully!")
    elif st.session_state.gpio_error:
        st.sidebar.error(f"❌ GPIO Error: {st.session_state.gpio_error}")
        if st.sidebar.button("🔄 Retry GPIO Setup"):
            st.session_state.gpio_initialized = False
            st.session_state.gpio_error = None
            st.rerun()
    else:
        st.sidebar.info("🔄 Initializing GPIO...")

    # Set up GPIO (only once)
    buzzer_pin, led_pin = setup_gpio()

    if alarm_enabled and st.session_state.gpio_initialized:
        st.sidebar.success("✅ Alarm system enabled")
    elif alarm_enabled:
        st.sidebar.warning("⚠️ Alarm system disabled (GPIO error)")
    else:
        st.sidebar.info("ℹ️ Alarm system disabled by user")

    col1, col2 = st.columns([3, 1])

    with col1:
        st.subheader("📺 Live Camera Feed")
        video_placeholder = st.empty()

    with col2:
        st.subheader("📊 System Status")
        people_count_metric = st.metric("👥 People Detected", st.session_state.people_counter['count'])
        status_metric = st.metric("🛡️ Security Status", "Safe")

        st.subheader("ℹ️ System Info")
        st.info(f"**Resolution:** {resolution[0]}x{resolution[1]}")
        st.info(f"**Restricted Area:** ({x1},{y1}) to ({x2},{y2})")
        st.info(f"**Confidence:** {confidence_threshold}")

        if st.button("🔄 Reset Camera"):
            st.session_state.camera_initialized = False
            st.rerun()

        if st.button("🛑 Stop System"):
            if st.session_state.video_stream:
                st.session_state.video_stream.stop()
            cleanup_gpio()
            st.session_state.camera_initialized = False
            st.success("System stopped successfully!")

    video_stream, camera_success = initialize_camera(resolution)

    if not camera_success:
        st.error("❌ Failed to initialize camera")
        st.stop()

    st.success("✅ Camera initialized successfully!")

    if video_stream:
        frame_count = 0
        start_time = time.time()

        # Create a container for the main loop
        main_container = st.container()

        with main_container:
            while True:
                frame = video_stream.read()

                if frame is None:
                    st.error("❌ Error: Failed to capture frame from camera.")
                    time.sleep(1)
                    continue

                frame_count += 1

                processed_frame, intrusion_detected = process_frame(frame, model, restricted_area, st.session_state.people_counter)

                people_count_metric.metric("👥 People Detected", st.session_state.people_counter['count'])

                if intrusion_detected:
                    status_metric.metric("🛡️ Security Status", "🚨 INTRUSION DETECTED", delta="ALERT!")

                    if alarm_enabled and st.session_state.gpio_initialized and buzzer_pin and led_pin:
                        # Start alarm in a separate thread to avoid blocking the main thread
                        alarm_thread = Thread(target=alarm_thread_function, args=(buzzer_pin, led_pin))
                        alarm_thread.daemon = True
                        alarm_thread.start()

                        # Log intrusion
                        st.sidebar.error(f"🚨 INTRUSION at {time.strftime('%H:%M:%S')}")
                else:
                    status_metric.metric("🛡️ Security Status", "✅ Safe", delta=None)

                # Convert to RGB for Streamlit display
                processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)

                # Display the frame
                video_placeholder.image(processed_frame_rgb, channels="RGB", use_container_width=True)

                # Calculate and display FPS every 30 frames
                if frame_count % 30 == 0:
                    elapsed_time = time.time() - start_time
                    fps = frame_count / elapsed_time
                    st.sidebar.metric("📈 FPS", f"{fps:.1f}")

                time.sleep(0.03)

if __name__ == "__main__":
    main()
