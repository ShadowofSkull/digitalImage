import cv2
import matplotlib.pyplot as plt
import numpy as np

# Obtain the path of the video from the user.
# path = input("Enter the path of the video: ")
# vid = cv2.VideoCapture(path)
# Read the video
vid = cv2.VideoCapture("./vids/traffic.mp4")
out = cv2.VideoWriter(
    "./processed_video.avi",  # Set the file name of the new video.
    cv2.VideoWriter_fourcc(*"MJPG"),  # Set the codec.
    30.0,  # Set the frame rate.
    (1280, 720),  # Set the resolution (width, height).
)
brightness_threshold = 120
total_brightness = 0
total_no_frames = vid.get(cv2.CAP_PROP_FRAME_COUNT)  # Get the total number of frames.
for frame_count in range(0, int(total_no_frames)):  # To loop through all the frames.
    success, frame = vid.read()  # Read a single frame from the video.
    # Do something here.
    if not success:
        break  # break if the video is not present or error.

    hsv = cv2.cvtColor(
        frame, cv2.COLOR_BGR2HSV
    )  # Convert the frame from BGR to HSV color space.
    pixel_values = hsv[
        :, :, 2
    ]  # Extract the Value (brightness) channel from the HSV image.
    mean_v = np.mean(pixel_values)  # Calculate the average brightness value.

    if mean_v < brightness_threshold:  # Check if the brightness is below the threshold.
        frame = cv2.add(frame, 50)

    # Detect faces
    face_cascade = cv2.CascadeClassifier(
        "./face_detector.xml"
    )  # Load pre-trained Haar cascade model.
    faces = face_cascade.detectMultiScale(frame, 1.3, 5)  # Perform face detection.
    for x, y, w, h in faces:  # To loop through all the detected faces.

        blurArea = frame[y : y + h, x : x + w]
        # Blurring the faces
        blurArea = cv2.GaussianBlur(blurArea, (23, 23), 30)
        # Applying blur to actual frame
        frame[y : y + blurArea.shape[0], x : x + blurArea.shape[1]] = blurArea
        cv2.imshow("Bounding", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):  # Wait for 1 millisecond.
            break

    out.write(frame)  # Save processed frame into the new video.

vid.release()  # Release the video capture object.
out.release()  # Release the video writer object.
cv2.destroyAllWindows()  # Close all the windows.
