import cv2      #opencv library

algorithm = "haarcascade_frontalface_default.xml"         #initialising algorithm filename which is in folder

alg = cv2.CascadeClassifier(algorithm)     #loading algorithm

camera = cv2.VideoCapture(0)   #initialising camera


while True:     #infinite loop to run cam continuously

    _ , img = camera.read()    #reading frame from camera

    grayImg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)    #convert to grayscale pic

    faces = alg.detectMultiScale(grayImg,1.3,5)   #get coordinates of faces

    for (x,y,w,h) in faces:

        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)   #draw rectangle around faces

    cv2.imshow("face",img)   #display

    key = cv2.waitKey(10)    #wait for 10 frames

    if key == 27:    #when esc key is pressed, close camera
        break

camera.release()    #release camera


cv2.destroyAllWindows()   #close window

        
        

    
