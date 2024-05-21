import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import cv2
import numpy as np

from pygame import mixer

import playsound
import queue as queue
from threading import Thread

resnet_model = load_model('models\\resnetModel01.h5')

resnet_model.load_weights('models\\resnet_weights_aug_extralayers_sgd_setval.hdf5')



def play_sound(file_path):
  mixer.init()
  mixer.music.load(file_path)
  mixer.music.play()
  
sound_file_path = "Alarm\\beep.mp3"

tags = { "C0": "safe driving",
"C1": "texting - right",
"C2": "talking on the phone - right",
"C3": "texting - left",
"C4": "talking on the phone - left",
"C5": "operating the radio",
"C6": "drinking",
"C7": "reaching behind",
"C8": "hair and makeup",
"C9": "talking to passenger" }



cap = cv2.VideoCapture(1)
prev = 0
count = 0
predicted_class = 0
output = ""
font = cv2.FONT_HERSHEY_SIMPLEX
x = 10 
y = 20 
q = queue.Queue(maxsize = 10)
fcount = 0
while(True):
    ret, frame = cap.read()
    conti = True
    if conti == True:
        # print(str(count)  + " " + str(prev))
        gray = cv2.cvtColor(frame, cv2.IMREAD_COLOR)
        if count == 2:
            print(str(count)  + " " + str(prev))
            
            img = gray[50:, 120:-50]
            img = cv2.resize(img, (128, 128))

            test = np.array(img).reshape(-1, 128, 128, 3)
            prediction = resnet_model.predict(test)
            predicted_class = 'C'+str(np.where(prediction == np.amax(prediction))[1][0])
            output = tags[predicted_class]
      
            output, model_out = "" + output , prediction
            count = 0
            prev = 0
            predicted_class = np.argmax(model_out)
            if q.full() == True:
                if fcount >= 5:
                    print("---------beep---------")
                    thread = Thread(target=play_sound, args=(sound_file_path,))
                    thread.start()
                fcount -= q.get()
            if predicted_class == 0:
                q.put(0)
            else:
                q.put(1)
                fcount += 1
            cv2.putText(gray,output,(10,100), font, 1,(0,255,0),2,cv2.LINE_AA)
            cv2.imshow('frame',gray)
        count += 1 
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else :
        break
cap.release()
cv2.destroyAllWindows()
