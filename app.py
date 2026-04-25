import cv2
import os
from flask import Flask, request, render_template
from datetime import date, datetime
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import joblib

# Flask app
app = Flask(__name__)

nimgs = 10

# Date formats
datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")

# Face detector
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Create folders if not exist
if not os.path.isdir('Attendance'):
    os.makedirs('Attendance')

if not os.path.isdir('static/faces'):
    os.makedirs('static/faces')

# Create attendance file
attendance_file = f'Attendance/Attendance-{datetoday}.csv'
if attendance_file not in os.listdir('Attendance'):
    with open(attendance_file, 'w') as f:
        f.write('Name,Roll,Time')

# Count users
def totalreg():
    return len(os.listdir('static/faces'))

# Extract faces
def extract_faces(img):
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.2, 5)
        return faces
    except:
        return []

# Identify face
def identify_face(facearray):
    model = joblib.load('static/face_recognition_model.pkl')
    return model.predict(facearray)

# Train model
def train_model():
    faces = []
    labels = []

    userlist = os.listdir('static/faces')

    for user in userlist:
        for imgname in os.listdir(f'static/faces/{user}'):
            img = cv2.imread(f'static/faces/{user}/{imgname}')
            resized_face = cv2.resize(img, (50, 50))
            faces.append(resized_face.ravel())
            labels.append(user)

    faces = np.array(faces)

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(faces, labels)

    joblib.dump(knn, 'static/face_recognition_model.pkl')

# Add attendance
def add_attendance(name):
    username = name.split('_')[0]
    userid = name.split('_')[1]

    current_time = datetime.now().strftime("%H:%M:%S")

    df = pd.read_csv(attendance_file)

    if int(userid) not in list(df['Roll']):
        with open(attendance_file, 'a') as f:
            f.write(f'\n{username},{userid},{current_time}')

# Routes
@app.route('/')
def home():
    return "Face Recognition Attendance System Running"

@app.route('/start')
def start():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces = extract_faces(frame)

        for (x, y, w, h) in faces:
            face = cv2.resize(frame[y:y+h, x:x+w], (50, 50))
            identified_person = identify_face(face.reshape(1, -1))[0]

            add_attendance(identified_person)

            cv2.putText(frame, identified_person, (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow('Attendance', frame)

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

    return "Attendance Taken"

if __name__ == "__main__":
    app.run(debug=True)
