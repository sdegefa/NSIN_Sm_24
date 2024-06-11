from flask import Flask, render_template, Response
import requests
import json
import cv2
import flask_obj_detection

app = Flask(__name__)

def gen(cam):

    while True:
        success, frame = cam.read()

        if not success:
            break
        else:
            frame = flask_obj_detection.inference(frame)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

        yield (b'--frame\r\n' 
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    

@app.route('/')
def index():
    return render_template('index_2.html')

@app.route('/video')
def video():
    cam=cv2.VideoCapture(0)
    return Response(gen(cam), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)