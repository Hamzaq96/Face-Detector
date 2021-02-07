import cv2

#Train the Algorithm.
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#img = cv2.imread('RP.jpg')
#To capture video from webcam
webcam = cv2.VideoCapture(0)

#Iterate through all the frames in the video.
while True:
    #Read the current frame
    successful_frame_read, frame = webcam.read()

    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

    for (x,y,w,h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)

    cv2.imshow('Face Detector', frame)

    key = cv2.waitKey(1)

    #Press Q to exit from the loop.
    if key==81 or key==113:
        break

#Release the video capture 
webcam.release()

# #Need to change the image to greyscale.
# grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# #Detect Faces
# face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

# #Draw a rectangle around the faces.
# #[x, y, w, h] = face_coordinates[0]
# for (x,y,w,h) in face_coordinates:
#     cv2.rectangle(img, (x, y), (x+w, y+h), (0,255,0), 2)

# #print(face_coordinates)

# cv2.imshow('Face Detector', img)

# #It waits for any keypress so you can view the image.
# cv2.waitKey()

print("Code Completed.")