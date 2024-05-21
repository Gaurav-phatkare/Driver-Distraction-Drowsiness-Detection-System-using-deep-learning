import cv2
import os
from tensorflow import keras
from keras.models import load_model
import numpy as np
from pygame import mixer
import time

mixer.init()
sound = mixer.Sound("alert\\alarm.wav")

face = cv2.CascadeClassifier('files\haarcascade_frontalface_alt.xml')
leye = cv2.CascadeClassifier('files\haarcascade_lefteye_2splits.xml')
reye = cv2.CascadeClassifier('files\haarcascade_righteye_2splits.xml')

lbl = ['Close', 'Open']

model = load_model('models\cnn_v01.h5')
path = os.getcwd()
cap = cv2.VideoCapture(1)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
count = 0
score = 0
thicc = 2
rpred = [99]
lpred = [99]

# Allow resizing the window
cv2.namedWindow('frame', cv2.WINDOW_NORMAL)

# Initialize variables for alarm control
alarm_triggered_time = 0

while True:
    ret, frame = cap.read()
    if frame is None:
        break

    # Resize the frame
    frame = cv2.resize(frame, (320, 240))

    height, width = frame.shape[:2]

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    faces = face.detectMultiScale(gray, minNeighbors=5, scaleFactor=1.1, minSize=(25, 25))
    left_eye = leye.detectMultiScale(gray)
    right_eye = reye.detectMultiScale(gray)

    # cv2.rectangle(frame, (0, height - 50), (100, height), (0, 0, 0), thickness=cv2.FILLED)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 100, 100), 1)

    for (x, y, w, h) in right_eye:
        r_eye = frame[y:y + h, x:x + w]
        count += 1
        r_eye = cv2.cvtColor(r_eye, cv2.COLOR_BGR2RGB)
        r_eye = cv2.resize(r_eye, (64, 64))
        r_eye = r_eye.reshape((-1, 64, 64, 3))

        rpred = np.argmax(model.predict(r_eye), axis=-1)
        if rpred[0] == 1:
            lbl = 'Open'
        if rpred[0] == 0:
            lbl = 'Closed'
        break

    for (x, y, w, h) in left_eye:
        l_eye = frame[y:y + h, x:x + w]
        count += 1
        l_eye = cv2.cvtColor(l_eye, cv2.COLOR_BGR2RGB)
        l_eye = cv2.resize(l_eye, (64, 64))
        l_eye = l_eye.reshape((-1, 64, 64, 3))

        lpred = np.argmax(model.predict(l_eye), axis=-1)
        if lpred[0] == 1:
            lbl = 'Open'
        if lpred[0] == 0:
            lbl = 'Closed'
        break

    if rpred[0] == 0 and lpred[0] == 0:
        score += 1
        cv2.putText(frame, "Closed", (10, height - 20), font, 0.7, (255, 255, 255), 1, cv2.LINE_AA)

        # Reset the alarm_triggered_time when eyes are closed
        alarm_triggered_time = 0
    else:
        score -= 1
        cv2.putText(frame, "Open", (10, height - 20), font, 0.7, (255, 255, 255), 1, cv2.LINE_AA)

        # Check if the alarm was triggered and stop it after 2 seconds
        if alarm_triggered_time > 0 and time.time() - alarm_triggered_time > 2:
            mixer.stop()
            alarm_triggered_time = 0

    if score < 0:
        score = 0
    cv2.putText(frame, 'Score:' + str(score), (60, height - 20), font, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
    if score > 7 and alarm_triggered_time == 0:
        cv2.imwrite(os.path.join(path, 'image.jpg'), frame)
        try:
            sound.play()
            alarm_triggered_time = time.time()  # Record the time when the alarm is triggered
            # Draw red square border around the frame
            cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), thicc)
        except:
            pass

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
