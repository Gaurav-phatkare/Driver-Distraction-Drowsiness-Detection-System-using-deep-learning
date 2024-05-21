import cv2
import os
from keras.models import load_model
import numpy as np
from pygame import mixer
import time

mixer.init()
sound = mixer.Sound("alert/alarm.wav")

face_cascade = cv2.CascadeClassifier('files/haarcascade_frontalface_alt.xml')
leye_cascade = cv2.CascadeClassifier('files/haarcascade_lefteye_2splits.xml')
reye_cascade = cv2.CascadeClassifier('files/haarcascade_righteye_2splits.xml')

model = load_model('models/cnn_v01.h5')
path = os.getcwd()
cap = cv2.VideoCapture(1)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
alarm_active = False
alarm_start_time = 0
alarm_duration = 5  # seconds
time_threshold = 1  # seconds
closing_time = 0

def play_alarm():
    try:
        sound.play(loops=-1)  # loops set to -1 for continuous playback
    except Exception as e:
        print(f"Error playing sound: {e}")

def stop_alarm():
    try:
        sound.stop()
    except Exception as e:
        print(f"Error stopping sound: {e}")

while True:
    ret, frame = cap.read()
    height, width = frame.shape[:2]

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    faces = face_cascade.detectMultiScale(gray, minNeighbors=5, scaleFactor=1.1, minSize=(25, 25))

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 100, 100), 1)

    rpred = [99]
    lpred = [99]

    for (x, y, w, h) in reye_cascade.detectMultiScale(gray):
        r_eye = frame[y:y + h, x:x + w]
        r_eye = cv2.cvtColor(r_eye, cv2.COLOR_BGR2RGB)
        r_eye = cv2.resize(r_eye, (64, 64))
        r_eye = r_eye.reshape((-1, 64, 64, 3))
        rpred = np.argmax(model.predict(r_eye), axis=-1)

    for (x, y, w, h) in leye_cascade.detectMultiScale(gray):
        l_eye = frame[y:y + h, x:x + w]
        l_eye = cv2.cvtColor(l_eye, cv2.COLOR_BGR2RGB)
        l_eye = cv2.resize(l_eye, (64, 64))
        l_eye = l_eye.reshape((-1, 64, 64, 3))
        lpred = np.argmax(model.predict(l_eye), axis=-1)

    if rpred[0] == 0 and lpred[0] == 0:
        closing_time = 0
        if alarm_active and (time.time() - alarm_start_time) > alarm_duration:
            stop_alarm()
            alarm_active = False
    else:
        closing_time += 1
        if closing_time >= time_threshold:
            if not alarm_active:
                alarm_start_time = time.time()
                alarm_active = True
                play_alarm()
                
    cv2.putText(frame, 'Closing Time:' + str(closing_time), (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
