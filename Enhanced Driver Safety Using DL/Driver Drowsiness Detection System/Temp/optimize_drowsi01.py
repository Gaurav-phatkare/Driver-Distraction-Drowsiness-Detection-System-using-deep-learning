# import cv2
# import os
# from keras.models import load_model
# import numpy as np
# from pygame import mixer
# import time

# mixer.init()
# sound = mixer.Sound("alert\\alarm.wav")

# face_cascade = cv2.CascadeClassifier('files/haarcascade_frontalface_alt.xml')
# leye_cascade = cv2.CascadeClassifier('files/haarcascade_lefteye_2splits.xml')
# reye_cascade = cv2.CascadeClassifier('files/haarcascade_righteye_2splits.xml')

# lbl = ['Close', 'Open']

# model = load_model('models\cnn_v01.h5')
# path = os.getcwd()
# cap = cv2.VideoCapture(1)
# font = cv2.FONT_HERSHEY_COMPLEX_SMALL
# count = 0
# score = 0
# thicc = 2
# rpred = [99]
# lpred = [99]
# alarm_active = False
# alarm_start_time = 0
# alarm_duration = 1  # seconds
# score_threshold = 4  # Adjust this value for sensitivity

# def play_alarm():
#     try:
#         sound.play(loops=-1)  # loops set to -1 for continuous playback
#     except Exception as e:
#         print(f"Error playing sound: {e}")

# def stop_alarm():
#     try:
#         sound.stop()
#     except Exception as e:
#         print(f"Error stopping sound: {e}")

# while True:
#     ret, frame = cap.read()
#     height, width = frame.shape[:2]

#     # Reduce image size for faster processing
#     frame = cv2.resize(frame, (360, 280))

#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#     faces = face_cascade.detectMultiScale(gray, minNeighbors=5, scaleFactor=1.1, minSize=(25, 25))

#     for (x, y, w, h) in faces:
#         cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 100, 100), 1)

#     for (x, y, w, h) in reye_cascade.detectMultiScale(gray):
#         r_eye = frame[y:y + h, x:x + w]
#         count += 1
#         r_eye = cv2.cvtColor(r_eye, cv2.COLOR_BGR2RGB)
#         r_eye = cv2.resize(r_eye, (64, 64))
#         r_eye = r_eye.reshape((-1, 64, 64, 3))
#         rpred = np.argmax(model.predict(r_eye), axis=-1)

#     for (x, y, w, h) in leye_cascade.detectMultiScale(gray):
#         l_eye = frame[y:y + h, x:x + w]
#         count += 1
#         l_eye = cv2.cvtColor(l_eye, cv2.COLOR_BGR2RGB)
#         l_eye = cv2.resize(l_eye, (64, 64))
#         l_eye = l_eye.reshape((-1, 64, 64, 3))
#         lpred = np.argmax(model.predict(l_eye), axis=-1)

#     if rpred[0] == 0 and lpred[0] == 0:
#         score += 1
#         cv2.putText(frame, "Eyes Closed", (10, height - 40), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
#         if not alarm_active and score > score_threshold:
#             alarm_start_time = time.time()
#             alarm_active = True
#             play_alarm()
#     else:
#         if alarm_active and (time.time() - alarm_start_time) > alarm_duration:
#             alarm_active = False
#             stop_alarm()
#         score = 0  # Reset score when eyes are open

#     cv2.putText(frame, 'Score: ' + str(score), (100, height - 40), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
#     if score > score_threshold:
#         cv2.imwrite(os.path.join(path, 'image.jpg'), frame)
#         if not alarm_active:
#             alarm_active = True
#             play_alarm()

#         if thicc < 10:
#             thicc += 2
#         else:
#             thicc = max(thicc - 2, 2)

#         cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), thicc)

#     cv2.imshow('frame', frame)
#     if cv2.waitKey(5) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()



# import cv2
# import os
# from keras.models import load_model
# import numpy as np
# from pygame import mixer
# import time

# mixer.init()
# sound = mixer.Sound("alert\\alarm.wav")

# face_cascade = cv2.CascadeClassifier('files/haarcascade_frontalface_alt.xml')
# leye_cascade = cv2.CascadeClassifier('files/haarcascade_lefteye_2splits.xml')
# reye_cascade = cv2.CascadeClassifier('files/haarcascade_righteye_2splits.xml')

# lbl = ['Close', 'Open']

# model = load_model('models\cnn_v01.h5')
# path = os.getcwd()
# cap = cv2.VideoCapture(1)
# font = cv2.FONT_HERSHEY_COMPLEX_SMALL
# count = 0
# score = 0
# thicc = 2
# rpred = [99]
# lpred = [99]
# alarm_active = False
# alarm_start_time = 0
# alarm_duration = 1  # seconds
# score_threshold = 4  # Adjust this value for sensitivity

# def play_alarm():
#     try:
#         sound.play(loops=-1)  # loops set to -1 for continuous playback
#     except Exception as e:
#         print(f"Error playing sound: {e}")

# def stop_alarm():
#     try:
#         sound.stop()
#     except Exception as e:
#         print(f"Error stopping sound: {e}")

# while True:
#     ret, frame = cap.read()
#     height, width = frame.shape[:2]

#     # Reduce image size for faster processing
#     frame = cv2.resize(frame, (360, 280))

#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#     faces = face_cascade.detectMultiScale(gray, minNeighbors=5, scaleFactor=1.1, minSize=(25, 25))

#     for (x, y, w, h) in faces:
#         cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 100, 100), 1)

#     for (x, y, w, h) in reye_cascade.detectMultiScale(gray):
#         r_eye = frame[y:y + h, x:x + w]
#         count += 1
#         r_eye = cv2.cvtColor(r_eye, cv2.COLOR_BGR2RGB)
#         r_eye = cv2.resize(r_eye, (64, 64))
#         r_eye = r_eye.reshape((-1, 64, 64, 3))
#         rpred = np.argmax(model.predict(r_eye), axis=-1)

#     for (x, y, w, h) in leye_cascade.detectMultiScale(gray):
#         l_eye = frame[y:y + h, x:x + w]
#         count += 1
#         l_eye = cv2.cvtColor(l_eye, cv2.COLOR_BGR2RGB)
#         l_eye = cv2.resize(l_eye, (64, 64))
#         l_eye = l_eye.reshape((-1, 64, 64, 3))
#         lpred = np.argmax(model.predict(l_eye), axis=-1)

#     if rpred[0] == 0 and lpred[0] == 0:
#         score += 1
#         cv2.putText(frame, "Eyes Closed", (10, 30), font, 0.7, (0, 0, 255), 1, cv2.LINE_AA)
#         if not alarm_active and score > score_threshold:
#             alarm_start_time = time.time()
#             alarm_active = True
#             play_alarm()
#     else:
#         if alarm_active and (time.time() - alarm_start_time) > alarm_duration:
#             alarm_active = False
#             stop_alarm()
#         score = 0  # Reset score when eyes are open

#     cv2.putText(frame, 'Score: ' + str(score), (10, 60), font, 0.7, (0, 255, 0), 1, cv2.LINE_AA)
#     if score > score_threshold:
#         cv2.imwrite(os.path.join(path, 'image.jpg'), frame)
#         if not alarm_active:
#             alarm_active = True
#             play_alarm()

#         if thicc < 10:
#             thicc += 2
#         else:
#             thicc = max(thicc - 2, 2)

#         cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), thicc)

#     cv2.imshow('frame', frame)
#     if cv2.waitKey(5) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()








import cv2
import os
from keras.models import load_model
import numpy as np
from pygame import mixer
import time

IMG_SIZE = 64

mixer.init()
sound = mixer.Sound("alert\warning.mp3")

face_cascade = cv2.CascadeClassifier('files/haarcascade_frontalface_alt.xml')
leye_cascade = cv2.CascadeClassifier('files/haarcascade_lefteye_2splits.xml')
reye_cascade = cv2.CascadeClassifier('files/haarcascade_righteye_2splits.xml')

lbl = ['Close', 'Open']

# model = load_model('models\cnn_v04.h5')

model = load_model('models\mobileNet\mobilenet64dist.h5')
model.load_weights('models\mobileNet\mobilenet_weights_aug_setval_sgd.hdf5')
path = os.getcwd()
cap = cv2.VideoCapture(1)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
count = 0
score = 0
thicc = 2
rpred = [99]
lpred = [99]
alarm_active = False
alarm_start_time = 0
alarm_duration = 1  # seconds
score_threshold = 3 # Adjust this value for sensitivity
skip_frames = 2  # Skip every 2 frames

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
    for _ in range(skip_frames):
        ret, frame = cap.read()

    if not ret:
        break

    height, width = frame.shape[:2]



    # Reduce image size for faster processing
    frame = cv2.resize(frame, (360, 280))

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    faces = face_cascade.detectMultiScale(gray, minNeighbors=5, scaleFactor=1.1, minSize=(25, 25))

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 100, 100), 1)

    for (x, y, w, h) in reye_cascade.detectMultiScale(gray):
        r_eye = frame[y:y + h, x:x + w]
        count += 1
        r_eye = cv2.cvtColor(r_eye, cv2.COLOR_BGR2RGB)
        r_eye = cv2.resize(r_eye, (IMG_SIZE, IMG_SIZE))

        # r_eye = r_eye / 255.0
        
        r_test = np.array(r_eye).reshape(-1, IMG_SIZE, IMG_SIZE,3)
        

        # r_eye = r_eye.reshape((-1, IMG_SIZE, IMG_SIZE, 3))
        rpred = np.argmax(model.predict(r_test), axis=-1)

    for (x, y, w, h) in leye_cascade.detectMultiScale(gray):
        l_eye = frame[y:y + h, x:x + w]
        count += 1
        l_eye = cv2.cvtColor(l_eye, cv2.COLOR_BGR2RGB)
        l_eye = cv2.resize(l_eye, (IMG_SIZE, IMG_SIZE))

        # l_eye = l_eye / 255.0
        l_test = np.array(l_eye).reshape(-1, IMG_SIZE, IMG_SIZE,3)

        # l_eye = l_eye.reshape((-1, IMG_SIZE, IMG_SIZE, 3))
        lpred = np.argmax(model.predict(l_test), axis=-1)

    if rpred[0] == 0 and lpred[0] == 0:
        score += 1
        cv2.putText(frame, "Eyes Closed", (10, 30), font, 0.7, (0, 0, 255), 1, cv2.LINE_AA)
        if not alarm_active and score > score_threshold:
            alarm_start_time = time.time()
            alarm_active = True
            play_alarm()
    else:
        if alarm_active and (time.time() - alarm_start_time) > alarm_duration:
            alarm_active = False
            stop_alarm()
        score = 0  # Reset score when eyes are open

    cv2.putText(frame, 'Score: ' + str(score), (10, 60), font, 0.7, (0, 255, 0), 1, cv2.LINE_AA)
    if score > score_threshold:
        cv2.imwrite(os.path.join(path, 'image.jpg'), frame)
        if not alarm_active:
            alarm_active = True
            play_alarm()

        if thicc < 10:
            thicc += 2
        else:
            thicc = max(thicc - 2, 2)

        cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), thicc)

    cv2.imshow('frame', frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()





