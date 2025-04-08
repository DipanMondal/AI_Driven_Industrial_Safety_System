import streamlit as st
import os
import cv2
import pandas as pd
from datetime import datetime
from pathlib import Path
import json
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoTransformerBase
from video_processor import VideoProcessor
import sys
import asyncio
from streamlit_drawable_canvas import st_canvas
import numpy as np
from PIL import Image
import tempfile
import shutil
import subprocess



if sys.platform.startswith('win'):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

st.set_page_config(page_title="Industry Safety Dashboard", layout="wide")
# st.experimental_set_query_params(watch=False)


# Initialize session state variables
if 'zone_points' not in st.session_state:
    st.session_state.zone_points = []
if 'drawing_mode' not in st.session_state:
    st.session_state.drawing_mode = False
if 'log_file' not in st.session_state:
    st.session_state.log_file = None

# Create folders if not exist
Path("logs").mkdir(exist_ok=True)
Path("zones").mkdir(exist_ok=True)

st.title("🏭 Industry Safety Monitoring System")

tabs = st.tabs(["🎥 Live Feed", "📐 Define Risky Zone", "🚦 Start Session", "🧾 View Logs","📼 Analyze Uploaded Video"])

# --- Tab 1: Live Feed ---
with tabs[0]:
    st.subheader("Live Monitoring Feed")
    
    log_file = st.session_state.get("log_file", None)
    
    print(log_file)
    
    webrtc_streamer(
        key="monitoring",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=lambda: VideoProcessor(log_file=log_file),
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

# --- Tab 2: Define Risky Zone ---

with tabs[1]:
    st.subheader("Step 1: Define Risky Zone")

    uploaded_image = st.file_uploader("Upload a frame to draw on", type=["jpg", "jpeg", "png"])

    if uploaded_image:
        file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if img is None:
            st.error("❌ Failed to decode image.")
        else:
            img = cv2.resize(img, (1024, 640))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)

            canvas_result = st_canvas(
                fill_color="rgba(255, 0, 0, 0.3)",
                stroke_width=3,
                stroke_color="#FF0000",
                background_image=img_pil,
                update_streamlit=True,
                height=640,
                width=1024,
                drawing_mode="polygon",
                key="canvas",
            )

            #print(canvas_result)
            # print(canvas_result.json_data)
            
            if canvas_result.json_data and canvas_result.json_data.get("objects"):
                objects = canvas_result.json_data["objects"]
                zone_points = []

                for obj in objects:
                    # print(obj)
                    if obj and "path" in obj:
                        for p in obj["path"]:
                            # print("\t",p)
                            if len(p)==3:
                                x, y = p[1], p[2]
                                zone_points.append((int(x), int(y)))
                # print(zone_points)

                if zone_points:
                    st.session_state.zone_points = zone_points

                    # Save when user clicks
                    if st.button("✅ Save Zone and Start Monitoring"):
                        zone_file = "zones/risky_zone.json"
                        os.makedirs("zones", exist_ok=True)
                        zone_points_to_save = [list(p) for p in zone_points]
                        with open(zone_file, "w") as f:
                            json.dump(zone_points_to_save, f)
                        st.success("Risky zone saved! Go to the next tab to start live monitoring.")



# --- Tab 3: Start Session ---
with tabs[2]:
    st.subheader("Session Control")
    
    session_name = st.text_input("Enter session name", f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    st.session_state.log_file = f"logs/{session_name}.csv"

    if st.button("🚀 Start Monitoring"):
        st.success(f"Monitoring session '{session_name}' started.")
        st.session_state.log_file = f"logs/{session_name}.csv"
        # You can trigger your existing OpenCV detection script here using subprocess
        # e.g., subprocess.Popen(["python", "safety_detection.py", "--log", st.session_state.log_file])
        st.warning("📷 Real-time detection to be integrated with webcam feed here.")

# --- Tab 4: View Logs ---
with tabs[3]:
    st.subheader("Session Violation Logs")

    log_files = sorted([f for f in os.listdir("logs") if f.endswith(".csv")], reverse=True)
    selected_log = st.selectbox("Select a session log to view:", log_files)

    if selected_log:
        df = pd.read_csv(os.path.join("logs", selected_log))
        st.dataframe(df, use_container_width=True)
        st.download_button("Download Log as CSV", df.to_csv(index=False), file_name=selected_log)
        

# --- Tab 5: Analyze Uploaded Video ---

with tabs[4]:
    st.subheader("Upload and Analyze Video")

    uploaded_video = st.file_uploader("📹 Upload a video", type=["mp4", "avi", "mov"])
    if uploaded_video:
        temp_video_path = os.path.join("temp", uploaded_video.name)
        os.makedirs("temp", exist_ok=True)

        with open(temp_video_path, "wb") as f:
            f.write(uploaded_video.read())

        cap = cv2.VideoCapture(temp_video_path)
        success, frame = cap.read()
        cap.release()

        if success:
            frame = cv2.resize(frame, (1024, 640))
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)

            st.write("✏️ Draw the risky zone on the first frame:")
            canvas_result = st_canvas(
                fill_color="rgba(255, 0, 0, 0.3)",
                stroke_width=3,
                stroke_color="#FF0000",
                background_image=img_pil,
                update_streamlit=True,
                height=640,
                width=1024,
                drawing_mode="polygon",
                key="offline_canvas"
            )

            zone_points = []
            if canvas_result.json_data and canvas_result.json_data.get("objects"):
                objects = canvas_result.json_data["objects"]
                for obj in objects:
                    if obj and "path" in obj:
                        for p in obj["path"]:
                            if len(p) == 3:
                                x, y = p[1], p[2]
                                zone_points.append((int(x), int(y)))

            if zone_points:
                if st.button("🚀 Start Analysis"):
                    output_video_path = temp_video_path.replace(".", "_annotated.")
                    output_log_path = temp_video_path.replace(".", "_log.") + "csv"

                    with open("zones/temp_offline_zone.json", "w") as f:
                        json.dump([list(p) for p in zone_points], f)

                    # Run offline analyzer
                    from video_analyzer import analyze_video
                    analyze_video(temp_video_path, output_video_path, output_log_path, "zones/temp_offline_zone.json")

                    st.success("✅ Analysis complete!")
                    st.video(output_video_path)

                    with open(output_video_path, "rb") as vid_file:
                        st.download_button("⬇️ Download Annotated Video", vid_file, file_name=os.path.basename(output_video_path))

                    df = pd.read_csv(output_log_path)
                    st.dataframe(df)
                    st.download_button("⬇️ Download Log CSV", df.to_csv(index=False), file_name=os.path.basename(output_log_path))
