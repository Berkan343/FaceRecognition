import cv2

vid_cam = cv2.VideoCapture(0)

yuz_dedektor = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

yuz_ismi = 3

sayı = 1

while(True):
    _,resim_cerceve = vid_cam.read()
    gri = cv2.cvtColor(resim_cerceve, cv2.COLOR_BGR2GRAY)
    yuzler = yuz_dedektor.detectMultiScale(gri, 1.3, 5)

    for (x,y,w,h) in yuzler:
                
        cv2.rectangle(resim_cerceve, (x,y), (x+w,y+h), (25,20,200), 2)
               
        sayı += 1
               
        cv2.imwrite("veri/User." + str(yuz_ismi) + '.' + str(sayı) + ".jpg", gri[y:y+h,x:x+w])
             
        cv2.imshow('cerceve', resim_cerceve)
            
    if cv2.waitKey(20) & 0xFF == ord('s'):
        break

    elif sayı>40:
        break

vid_cam.release()

cv2.destroyAllWindows()
