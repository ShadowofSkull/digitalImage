import cv2

cap = cv2.VideoCapture("./processed_video.mp4")  # Read the processed video.
print(type(cap))
while cap.isOpened():
    ret, frame = cap.read()  # Read a single frame from the video.

    cv2.imshow("Frame", frame)  # Display the frame.

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()  # Release the video capture object.
cap.destroyAllWindows()  # Close all the windows.
