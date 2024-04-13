import cv2
import imutils

cam = cv2.VideoCapture(0)

firstFrame = None
area = 500

while True:
    _, img = cam.read()  # read from camera
    text = "Normal"
    img = imutils.resize(img, width=1000)  # resizing
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gaussianImg = cv2.GaussianBlur(img, (21, 21), 0)
    
    if firstFrame is None:
        firstFrame = gaussianImg   
        continue
    
    imgDiff = cv2.absdiff(firstFrame, gaussianImg)
    thresholdImg = cv2.threshold(imgDiff, 25, 255, cv2.THRESH_BINARY)[1]
    thresholdImg = cv2.dilate(thresholdImg, None, iterations=2)

    # Convert thresholdImg to grayscale (CV_8UC1)
    thresholdImg_gray = cv2.cvtColor(thresholdImg, cv2.COLOR_BGR2GRAY)

    cnts = cv2.findContours(thresholdImg_gray.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    for c in cnts:
        if cv2.contourArea(c) < area:
            continue
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        text = "Moving object Detected"
    
    print(text)
    
    cv2.putText(img, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.imshow("camerafeed", img)

    key = cv2.waitKey(10)
    print(key)
    if key == ord("c"):
        break

cam.release()
cv2.destroyAllWindows()
