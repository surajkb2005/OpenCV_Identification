import cv2 as cv

vidcap = cv.VideoCapture(0)
har_cat = cv.CascadeClassifier('assets/xml/cat_data.xml')

while True:
    isTrue,frame = vidcap.read()
    flippedframe = cv.flip(frame,1)
    if not isTrue:
        break

    cv.imshow("Original video",flippedframe)
    cat_rect = har_cat.detectMultiScale(flippedframe,scaleFactor=1.1,minNeighbors=8)

    for (x,y,w,h) in cat_rect:
        cv.rectangle(flippedframe,(x,y),(x+w,y+h),(255,0,0),thickness=2)
    
    text = f"NUMBER OF CAT FOUND: {len(cat_rect)}"
    cv.putText(flippedframe,text,(30,30),fontFace=1,fontScale=1,color=(255,0,0))

    cv.imshow("Cat found : ",flippedframe)

    if cv.waitKey(5) & 0xff == ord('d'):
        break

vidcap.release()
cv.destroyAllWindows()