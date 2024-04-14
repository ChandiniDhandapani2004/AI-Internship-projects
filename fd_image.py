import cv2

alg = "haarcascade_frontalface_default.xml"

haar_cascade = cv2.CascadeClassifier(alg)  #Loading Algorithm

img_path = "msd.jpg"

img = cv2.imread(img_path) # here we use imread to read image as input

while True:
    grayImg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    face = haar_cascade.detectMultiScale(grayImg,1.3,5)   #getting coordinates

    for (x,y,w,h) in face:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),5)
    cv2.imshow("Face Detection",img)

    key = cv2.waitKey(10)
    print(key)
    if key == 27:
        break

cv2.destroyAllWindows()
