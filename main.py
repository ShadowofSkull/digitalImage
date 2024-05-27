import cv2
import numpy as np


# Create fade effect
def fade(frame, fade_val, brightness):
    # Remove brightness val from frame
    frame = cv2.subtract(frame, fade_val)
    rate = 5
    # Increase value to remove which make the next frame darker
    if brightness == "darken":
        fade_val += rate
    # Decrease value to remove which make the next frame brighter
    elif brightness == "brighten":
        fade_val -= rate
    return fade_val, frame


# Initialising variables:
# Obtain the path of the video to process from the user.
file_name = input("Enter the name of the video (no ext e.g. mp4): ")
# Read the video
vid = cv2.VideoCapture(f"./vids/{file_name}.mp4")

fps = 30.0
# Obtain the resolution of the video.
success, frame = vid.read()
shape = frame.shape
resolution = (shape[1], shape[0])
out = cv2.VideoWriter(
    f"./out/{file_name}_processed_video.avi",  # Set the file name of the new video.
    cv2.VideoWriter_fourcc(*"MJPG"),  # Set the codec.
    fps,  # Set the frame rate.
    resolution,  # Set the resolution (width, height).
)

# Logo position variables
logo_size = 64
logo_padding = 50

# import overlay
talking_vid = cv2.VideoCapture("./vids/talking.mp4")
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
# Set cut off value to determine time of day
brightness_threshold = 135
total_no_frames = vid.get(cv2.CAP_PROP_FRAME_COUNT)  # Get the total number of frames.
# Starting brightness value to negate for fade in/out effect
fade_val = 255
# Amount of frames before detecting faces
detectInterval = 8

# Processing of video frames takes place here:
for frame_count in range(0, int(total_no_frames)):  # To loop through all the frames.
    success, frame = vid.read()  # Read a single frame from the video.
    talking_ret, talking_frame = (
        talking_vid.read()
    )  # Read a single frame from the video.

    # Do something here.
    if not success:
        break  # break if the video is not present or error.

    # Adding logo to every frame
    # Create a black background for logo and set its size
    logo = np.zeros((64, 64, 3), np.uint8)
    # Sets the coordinates the line will be drawn between
    pts = np.array(
        [[32, 10], [16, 54], [54, 32], [10, 32], [48, 54], [32, 10]], np.int32
    )
    print(resolution[1])
    # Draw lines according to the coordinates to create a star
    cv2.polylines(logo, [pts], True, (0, 255, 255))
    # [y1:y2, x1:x2]
    print(frame.shape)
    frame[
        logo_padding : logo_padding + logo_size,
        resolution[0] - logo_padding - logo_size : resolution[0] - logo_padding,
    ] = logo
    cv2.imshow("frame", frame)
    cv2.waitKey(1)
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

    # Detect faces at a interval
    if frame_count % detectInterval == 0:
        face_cascade = cv2.CascadeClassifier(
            "./face_detector.xml"
        )  # Load pre-trained Haar cascade model.
        faces = face_cascade.detectMultiScale(frame, 1.3, 5)  # Perform face detection.

    for x, y, w, h in faces:  # To loop through all the detected faces.
        blurArea = frame[y : y + h, x : x + w]
        # Blurring the faces
        blurArea = cv2.GaussianBlur(blurArea, (23, 23), 30)
        # Applying blur to actual frame
        frame[y : y + h, x : x + w] = blurArea

    # Talking video overlay
    if talking_ret:
        talking_frame = cv2.resize(
            talking_frame, (int(vid.get(3) * 0.40), int(vid.get(4) * 0.40))
        )

        hsv = cv2.cvtColor(talking_frame, cv2.COLOR_BGR2HSV)
        lGreen = np.array([36, 25, 25])
        uGreen = np.array([70, 255, 255])
        mask = cv2.inRange(hsv, lGreen, uGreen)
        mask_inv = cv2.bitwise_not(mask)
        # Region of Interest (ROI) in the frame
        roi = frame[
            y_offset : y_offset + overlay_height, x_offset : x_offset + overlay_width
        ]
        # Extract the foreground and background
        foreground = cv2.bitwise_and(talking_frame, talking_frame, mask=mask_inv)
        background = cv2.bitwise_and(roi, roi, mask=mask)

        # combine the foreground and background
        overlay = cv2.add(foreground, background)
        frame[
            y_offset : y_offset + overlay_height, x_offset : x_offset + overlay_width
        ] = overlay

    # Creating fade in/out effect
    if frame_count >= 0 and frame_count <= 50:
        fade_val, frame = fade(frame, fade_val, "brighten")
    elif frame_count >= total_no_frames - 51 and frame_count < total_no_frames:
        fade_val, frame = fade(frame, fade_val, "darken")

    out.write(frame)  # Save processed frame into the new video.

# Add Ending screen
ending = cv2.VideoCapture("./vids/endscreen.mp4")

total_no_frames = ending.get(
    cv2.CAP_PROP_FRAME_COUNT
)  # Get the total number of frames.

fade_val = 0
for frame_count in range(0, int(total_no_frames)):  # To loop through all the frames.
    success, frame = ending.read()
    if not success:
        break
    # Resize frame to match the resolution of the main video
    frame = cv2.resize(frame, resolution)
    # Creating fade out effect
    if frame_count >= total_no_frames - 51 and frame_count < total_no_frames:
        fade_val, frame = fade(frame, fade_val, "darken")
    # Save processed frame into the new video.
    out.write(frame)


vid.release()  # Release the video capture object.
ending.release()  # Release the video capture object.
out.release()  # Release the video writer object.
