import cv2

alg = "haarcascade_frontalface_default.xml"

haar_cascade = cv2.CascadeClassifier(alg)  #Loading Algorithm

cam = cv2.VideoCapture(0) #Initializing Camera ID

while True:
    _,img = cam.read()    #reading frame from camera

    grayImg = cv2.cvtColor(img,COLOR_BGR2GRAY)

    face = haar_cascade.detectMultiScale(grayImg,1.3,5)   #getting coordinates

    for (x,y,w,h) in face:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),3)
    cv2.imshow("Face Detection",img)
    key = cv2.waitKey(10)
    if key == 27:
        break
cam.release()
cv2.destroyAllWindows()