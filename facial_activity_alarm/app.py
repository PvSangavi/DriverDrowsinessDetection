from flask import Flask, render_template, Response
import cv2
import numpy as np
import time
from playsound import playsound
import threading

app = Flask(__name__)

# Global variables for detection
yawn_count = 0
eye_close_start_time = None
alarm_triggered = False


def play_alarm():
    global alarm_triggered
    if not alarm_triggered:
        alarm_triggered = True
        playsound("static/severe-warning-alarm-98704.mp3")
        alarm_triggered = False


def generate_frames():
    global yawn_count, eye_close_start_time

    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

    while True:
        success, frame = cap.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Eye detection logic
            if len(eyes) == 0:
                if eye_close_start_time is None:
                    eye_close_start_time = time.time()
                elif time.time() - eye_close_start_time > 3:
                    threading.Thread(target=play_alarm).start()
            else:
                eye_close_start_time = None

        # Encode the frame to send it as a video feed
        ret, buffer = cv2.imencode(".jpg", frame)
        frame = buffer.tobytes()
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

    cap.release()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    app.run(debug=True)
