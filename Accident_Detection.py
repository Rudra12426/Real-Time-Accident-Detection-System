#----------This logic monitors multiple cctv camera simultaneously at one -----------and this is final version of it ------------
import cv2
import os
from datetime import datetime
import winsound
import time

# -------- SETTINGS --------
SPIKE_THRESHOLD = 4500
STILLNESS_THRESHOLD = 800
CONFIRMATION_FRAMES = 8

FREQ = 2500
DUR = 300
REPEAT = 10

DISPLAY_DURATION = 30

os.makedirs("outputs/accident_frames", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# -------- CAMERA LIST --------
# use video paths if you have traffic or accident detection videos 
# Replace 0, 1 with your webcams or RTSP URLs
camera_sources = [0, 1]  # Example: [0, 1, "rtsp://username:pass@192.168.1.100/stream1"]
#
#camera_sources = [
   # 0,  # first webcam for (USB CAMERA)
   #1 ,# Second webcam FOR (USOB CAMERA) and so on.......
    #"rtsp://username:password@192.168.1.100:554/Streaming/Channels/101", IP CCTV1
    #"rtsp://username:password@192.168.1.101:554/Streaming/Channels/101"] IP CCTV2



# -------- INITIALIZE CAMERA DATA --------
cams = {}
for idx, src in enumerate(camera_sources):
    cap = cv2.VideoCapture(src)
    ret, frame = cap.read()
    if not ret:
        print(f"❌ Cannot access camera {src}")
        continue
    prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cams[idx] = {
        "cap": cap,
        "prev_gray": prev_gray,
        "state": 0,
        "still_counter": 0,
        "accident_confirmed": False,
        "accident_frame_display": False,
        "display_counter": 0,
        "frame_number": 0
    }

if not cams:
    print("❌ No cameras available")
    exit()

# -------- MAIN LOOP --------
while True:
    for cam_id, data in cams.items():
        cap = data["cap"]
        ret, frame = cap.read()
        if not ret:
            continue

        data["frame_number"] += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(data["prev_gray"], gray)
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        clean = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        total_motion_area = sum(cv2.contourArea(c) for c in contours)

        # Draw bounding boxes
        for c in contours:
            if cv2.contourArea(c) > 300:
                x, y, w, h = cv2.boundingRect(c)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # -------- STATE LOGIC --------
        if data["state"] == 0:
            if total_motion_area > SPIKE_THRESHOLD:
                data["state"] = 1
        elif data["state"] == 1:
            if total_motion_area < STILLNESS_THRESHOLD:
                data["still_counter"] += 1
                if data["still_counter"] >= CONFIRMATION_FRAMES:
                    data["state"] = 2
            else:
                data["still_counter"] = 0

        # -------- ACCIDENT CONFIRMED --------
        if data["state"] == 2 and not data["accident_confirmed"]:
            data["accident_confirmed"] = True
            data["accident_frame_display"] = True
            data["display_counter"] = 0

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            frame_path = f"outputs/accident_frames/cam{cam_id}_{timestamp}.jpg"
            cv2.imwrite(frame_path, frame)
            with open("logs/accident_log.txt", "a") as f:
                f.write(f"Camera {cam_id} Frame: {data['frame_number']}, Time: {timestamp}\n")

            print(f"🚨 Accident detected on Camera {cam_id}!")

            for _ in range(REPEAT):
                winsound.Beep(FREQ, DUR)
                time.sleep(0.05)

        # -------- DISPLAY --------
        cv2.putText(frame, f"Motion Area: {int(total_motion_area)}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        if data["accident_frame_display"]:
            cv2.putText(frame, "ACCIDENT CONFIRMED", (40, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            data["display_counter"] += 1
            if data["display_counter"] >= DISPLAY_DURATION:
                data["accident_frame_display"] = False

        cv2.imshow(f"Camera {cam_id}", frame)
        cv2.imshow(f"Motion Mask {cam_id}", clean)

        data["prev_gray"] = gray

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

for cam_id, data in cams.items():
    data["cap"].release()
cv2.destroyAllWindows()
