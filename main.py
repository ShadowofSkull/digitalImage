import cv2

#Read the video.
vid = cv2.VideoCapture("street.mp4")
out = cv2.VideoWriter('processed_video.avi', #Set the file name of the new video.
                      cv2.VideoWriter_fourcc(*'MJPG'), #Set the codec.
                      30.0, #Set the frame rate.
                      (1280,720) #Set the resolution (width, height).
                      )
total_no_frames = vid.get(cv2.CAP_PROP_FRAME_COUNT) #Get the total number of frames.
for frame_count in range(0, int(total_no_frames)): #To loop through all the frames.
    success, frame = vid.read() #Read a single frame from the video.
    # Do something here.

    out.write(frame) #Save processed frame into the new video.

# Blurring faces
face_cascade = cv2.CascadeClassifier("face_detector.xml") #Load pre-trained Haar cascade model.
faces = face_cascade.detectMultiScale(frame, 1.3, 5) #Perform face detection.
for (x, y, w, h) in faces: #To loop through all the detected faces.
    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2) #Draw a bounding box.






