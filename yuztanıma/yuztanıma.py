import cv2
import numpy as np

tanıyıcı = cv2.face.LBPHFaceRecognizer_create()
tanıyıcı.read('deneme/deneme.yml')
yolsınıflandırıcı = "haarcascade_frontalface_default.xml"
yuzsınıflandırıcı = cv2.CascadeClassifier(yolsınıflandırıcı);

font = cv2.FONT_HERSHEY_SIMPLEX
vid_cam = cv2.VideoCapture(0)

while True:
  
    ret, kamera =vid_cam.read()

    gri = cv2.cvtColor(kamera,cv2.COLOR_BGR2GRAY)

    yuzler = yuzsınıflandırıcı.detectMultiScale(gri, 1.2,5)
        
    for(x,y,w,h) in yuzler:

        cv2.rectangle(kamera, (x-20,y-20), (x+w+20,y+h+20), (25,20,200), 4)
               
        Id,conf = tanıyıcı.predict(gri[y:y+h,x:x+w])

        if(Id == 1):
            Id = "berkan"
       
        elif(Id == 2):
            Id = "berat"

        elif(Id == 3):
            Id = "ugurhoca"

      
        cv2.rectangle(kamera, (x-22,y-90), (x+w+22, y-22), (200,255,30), -1)
        cv2.putText(kamera, str(Id), (x,y-40), font, 2, (0,0,255), 3)
            
    cv2.imshow('kamera',kamera) 
       
    if cv2.waitKey(10) & 0xFF == ord('s'):
      break

vid_cam.release()

cv2.destroyAllWindows()