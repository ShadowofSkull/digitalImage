import cv2
import numpy as np
import time

# ------------------ Load and Setup ------------------

# Paths to videos and images (update these with your file paths)
main_video_path = "../vids/singapore.mp4"
talking_head_path = "../vids/talking.mp4"
logo_path = "../imgs/logo.png"
watermark1_path = "../imgs/watermark1.png"
watermark2_path = "../imgs/watermark2.png"
face_detector_path = "../face_detector.xml"

# Load videos
main_vid = cv2.VideoCapture(main_video_path)
talking_head_vid = cv2.VideoCapture(talking_head_path)

# Load images
logo = cv2.imread(logo_path)
watermark1 = cv2.imread(watermark1_path)
watermark2 = cv2.imread(watermark2_path)

# Overlay parameters (bottom left)
overlay_width = int(main_vid.get(3) * 0.25)  # 25% of main video width
overlay_height = int(main_vid.get(4) * 0.25)  # 25% of main video height
x_offset = 0
y_offset = int(main_vid.get(4)) - overlay_height

# Output video setup
output_path = "./processed_video.avi"
out = cv2.VideoWriter(
    output_path, 
    cv2.VideoWriter_fourcc(*"MJPG"), 
    30.0,  # Frames per second
    (1280, 720),  # Output video resolution
)

# Video processing parameters
brightness_threshold = 135  # Threshold for adjusting brightness in low-light scenes
total_no_frames = int(main_vid.get(cv2.CAP_PROP_FRAME_COUNT))
fade_val = 255
start_time = time.time()  # For timing the watermark transitions

# Face detection setup
face_cascade = cv2.CascadeClassifier(face_detector_path)

# ------------------ Main Video Processing ------------------

for frame_count in range(total_no_frames):
    # Read frames
    success, main_frame = main_vid.read()
    talking_head_ret, talking_head_frame = talking_head_vid.read()
    if not success:
        break  # End of main video

    # --- Fading effect (beginning of video) ---
    if frame_count <= 50:
        main_frame = cv2.subtract(main_frame, fade_val)
        fade_val -= 5

    # --- Add logo to each frame ---
    main_frame[606:670, 1166:1230] = logo

    # --- Add alternating watermarks every 5 seconds ---
    end_time = time.time()
    if int((end_time - start_time) % 5) == 0:
        main_frame = cv2.add(main_frame, watermark1)
    else:
        main_frame = cv2.add(main_frame, watermark2)

    # --- Brightness adjustment for nighttime scenes ---
    hsv = cv2.cvtColor(main_frame, cv2.COLOR_BGR2HSV)
    mean_v = np.mean(hsv[:, :, 2])
    if mean_v < brightness_threshold:
        main_frame = cv2.add(main_frame, 70)

    # --- Face detection and blurring ---
    faces = face_cascade.detectMultiScale(main_frame, 1.3, 5)
    for (x, y, w, h) in faces:
        roi = main_frame[y:y+h, x:x+w]
        roi = cv2.GaussianBlur(roi, (23, 23), 30)
        main_frame[y:y+h, x:x+w] = roi

    # --- Talking Head Overlay (Green Screen Removal) ---
    if talking_head_ret:
        # Resize talking head frame to match overlay dimensions
        talking_head_frame = cv2.resize(talking_head_frame, (overlay_width, overlay_height))

        # Green screen removal (Chroma keying)
        hsv = cv2.cvtColor(talking_head_frame, cv2.COLOR_BGR2HSV)
        lower_green = np.array([36, 25, 25])
        upper_green = np.array([70, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)
        mask_inv = cv2.bitwise_not(mask)

        # Extract foreground (person) and background 
        fg = cv2.bitwise_and(talking_head_frame, talking_head_frame, mask=mask_inv)
        bg = cv2.bitwise_and(main_frame[y_offset:y_offset+overlay_height, x_offset:x_offset+overlay_width], main_frame[y_offset:y_offset+overlay_height, x_offset:x_offset+overlay_width], mask=mask)

        # Combine foreground and background
        final_overlay = cv2.add(fg, bg)
        main_frame[y_offset:y_offset+overlay_height, x_offset:x_offset+overlay_width] = final_overlay

    out.write(main_frame)  

    if cv2.waitKey(1) & 0xFF == ord("q"):  
        break
    
    
 
ending = cv2.VideoCapture("../vids/endscreen.mp4")
total_no_frames = ending.get(
    cv2.CAP_PROP_FRAME_COUNT
)  # Get the total number of frames.

fade_val = 0
for frame_count in range(0, int(total_no_frames)):  # To loop through all the frames.
    success, frame = ending.read()
    if not success:
        break

    if frame_count >= total_no_frames - 51 and frame_count <= total_no_frames:
        frame = cv2.subtract(frame, fade_val)
        fade_val += 5
        print(fade_val)
    cv2.imshow("Ending", frame)
    out.write(frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

main_vid.release()  # Release the video capture object.
ending.release()
out.release()  # Release the video writer object.
cv2.destroyAllWindows()  # Close all the windows.
   
    
    
    
    
    
    
    
    
    
    
    
    