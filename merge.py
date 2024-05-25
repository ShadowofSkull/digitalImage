import cv2
import numpy as np

# Obtain the path of the video from the user.
file_name = input("Enter the name of the video (no ext e.g. mp4): ")
# Read the video
vid = cv2.VideoCapture(f"./vids/{file_name}.mp4")

fps = 30.0
success, frame = vid.read()
shape = frame.shape
resolution = (shape[1], shape[0])
out = cv2.VideoWriter(
    f"./out/{file_name}_processed_video.avi",  # Set the file name of the new video.
    cv2.VideoWriter_fourcc(*"MJPG"),  # Set the codec.
    fps,  # Set the frame rate.
    resolution,  # Set the resolution (width, height).
)

logo = cv2.imread("./imgs/logo.png")
logo_size = 64
logo_spacing = 50

# import overlay
talking_vid = cv2.VideoCapture('./vids/talking.mp4')
# Overlay parameters (bottom left)
overlay_width = int(vid.get(3) * 0.40)  # 40% of main video width
overlay_height = int(vid.get(4) * 0.40)  # 40% of main video height
x_offset = 0
y_offset = int(vid.get(4)) - overlay_height

# Resize watermark in case of different resolution
watermark1 = cv2.imread("./imgs/watermark1.png")
watermark1 = cv2.resize(watermark1, resolution)
watermark2 = cv2.imread("./imgs/watermark2.png")
watermark2 = cv2.resize(watermark2, resolution)
# Set the watermark to be displayed initially
watermark_num = 2
brightness_threshold = 135
total_no_frames = vid.get(cv2.CAP_PROP_FRAME_COUNT)  # Get the total number of frames.
fade_val = 255

for frame_count in range(0, int(total_no_frames)):  # To loop through all the frames.
    success, frame = vid.read()  # Read a single frame from the video.
    talking_ret, talking_frame = talking_vid.read() # Read a single frame from the video.
    # Do something here.
    if not success:
        break  # break if the video is not present or error.

    # Creating fade in effect
    if frame_count >= 0 and frame_count <= 50:
        frame = cv2.subtract(frame, fade_val)
        fade_val -= 5
    # Adding logo to every frame
    
    frame[logo_spacing: logo_spacing + logo_size, logo_spacing: logo_spacing + logo_size] = logo

    # Adding alternating watermark every 5s
    frames_per_5s = int(fps * 5)
    if frame_count % frames_per_5s == 0:
        match watermark_num:
            case 1:
                watermark = watermark2
                watermark_num = 2
            case 2:
                watermark = watermark1
                watermark_num = 1

    frame = cv2.add(frame, watermark)

    # Increase brightness if nighttime
    hsv = cv2.cvtColor(
        frame, cv2.COLOR_BGR2HSV
    )  # Convert the frame from BGR to HSV color space.
    pixel_values = hsv[
        :, :, 2
    ]  # Extract the Value (brightness) channel from the HSV image.
    mean_v = np.mean(pixel_values)  # Calculate the average brightness value.

    if mean_v < brightness_threshold:  # Check if the brightness is below the threshold.
        frame = cv2.add(frame, 70)

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
        
        
    if talking_ret:
        talking_frame = cv2.resize(talking_frame,(int(vid.get(3) * 0.40), int(vid.get(4) * 0.40)))
        
        hsv = cv2.cvtColor(talking_frame, cv2.COLOR_BGR2HSV)
        lGreen = np.array([36,25,25])
        uGreen = np.array([70,255,255])
        mask = cv2.inRange(hsv, lGreen, uGreen)
        mask_inv = cv2.bitwise_not(mask) 
        
        # Extract the foreground and background
        foreground = cv2.bitwise_and(talking_frame, talking_frame, mask = mask_inv)
        background = cv2.bitwise_and(frame[y_offset:y_offset+overlay_height, x_offset:x_offset+overlay_width], frame[y_offset:y_offset+overlay_height, x_offset:x_offset+overlay_width], mask=mask)       
        
        # combine the foreground and background
        overlay = cv2.add(foreground, background)
        frame[y_offset:y_offset+overlay_height, x_offset:x_offset+overlay_width] = overlay
        
        
        cv2.imshow("Demo", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):  # Wait for 1 millisecond.
            break
    
    
        



    out.write(frame)  # Save processed frame into the new video.

# Ending screen
ending = cv2.VideoCapture("./vids/endscreen.mp4")

total_no_frames = ending.get(
    cv2.CAP_PROP_FRAME_COUNT
)  # Get the total number of frames.

fade_val = 0
for frame_count in range(0, int(total_no_frames)):  # To loop through all the frames.
    success, frame = ending.read()
    if not success:
        break
    frame = cv2.resize(frame, resolution)
    if frame_count >= total_no_frames - 51 and frame_count <= total_no_frames:
        frame = cv2.subtract(frame, fade_val)
        fade_val += 5

    cv2.imshow("Demo", frame)
    out.write(frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

vid.release()  # Release the video capture object.
ending.release()
out.release()  # Release the video writer object.
cv2.destroyAllWindows()  # Close all the windows.
