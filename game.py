import cv2
import mediapipe as mp
import numpy as np
import time

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

device = 0

start = [50, 240]
goal = [590, 240]
cnt = 0
flag=0

back = cv2.imread('./imgs/back1.jpg')

def game(img, player):
    global cnt
    global flag
    if cv2.waitKey(5) & 0xFF == ord('r'):
            cnt = 0
            flag = 0
    if flag == 1:
        cv2.putText(img, "GameOver!!",  (150, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 5)
    elif flag == 2:
        cv2.putText(img, "Clear!!",  (250, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 5)
    elif all(back[player[1]][player[0]]!=[255,255,255]):
        flag = 1
        cv2.putText(img, "GameOver!!",  (150, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 5)

    elif player[0]>=goal[0] and goal[1]-8 <= player[1] <=goal[1]+8:
        flag = 2
        cv2.putText(img, "Clear!!",  (250, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 5)

    

        
        

   

def drawFingertip(img, landmarks):
    img_width, img_height = img.shape[1], img.shape[0]
    landmark_point = []
    global cnt

    for index, landmark in enumerate(landmarks.landmark):
        if landmark.visibility < 0 or landmark.presence < 0:
            continue

        # Convert the obtained landmark values x and y to the coordinates on the img
        landmark_x = min(int(landmark.x * img_width), img_width - 1)
        landmark_y = min(int(landmark.y * img_height), img_height - 1)
        landmark_z = landmark.z

        landmark_point.append([landmark_x, landmark_y, landmark_z])
    # Draw a circle on index finger and display the coordinate value
    cv2.circle(img, (landmark_point[8][0], landmark_point[8][1]), 7, (0, 0, 255), -1)
    player = [landmark_point[8][0], landmark_point[8][1]]
    if player[0]<=start[0] and start[1]-5 <= player[1] <=start[1]+5 or cnt!=0:
        cnt += 1
        if 6 >= cnt >= 0:
            cv2.putText(img, "Start!!",  (250, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 5)
        game(img, player)
    else:
        cv2.putText(img, "Put yourfinger on the start position",  (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 5)

def combine(img):
    white = np.ones((img.shape), dtype=np.uint8) * 255 #make a matrix whose size is the same as img 
    y = img.shape[0]-back.shape[0]
    x = img.shape[1]-back.shape[1]
    white[y:img.shape[0],x:img.shape[1]] = back

    img[white!=[255, 255, 255]] = white[white!=[255, 255, 255]]

    cv2.circle(img, (start[0],start[1]), 7, (0, 0, 255), 3) #start
    cv2.putText(img, "start", (start[0]-25, start[1]-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 5)
    cv2.circle(img, (goal[0],goal[1]), 7, (0, 0, 255), 3) #goal
    cv2.putText(img, "goal", (goal[0]-25, goal[1]-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 5)

    return img

def getFrameNumber(start:float, fps:int):
    now = time.perf_counter() - start
    frame_now = int(now * 1000 / fps)

    return frame_now

def main():
    # For webcam input:
    global device
    cap = cv2.VideoCapture(device)
    fps = cap.get(cv2.CAP_PROP_FPS)
    wt  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    ht  = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    print("Size:", ht, "x", wt, "/Fps: ", fps)

    start = time.perf_counter()
    frame_prv = -1

    cv2.namedWindow('MediaPipe Hands', cv2.WINDOW_NORMAL)
    with mp_hands.Hands(
        max_num_hands = 1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7) as hands:
        while cap.isOpened():
            frame_now=getFrameNumber(start, fps)
            if frame_now == frame_prv:
                continue
            frame_prv = frame_now

            ret, frame = cap.read()
            if not ret:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue

            # Flip the img horizontally for a later selfie-view display, and convert
            # the BGR img to RGB.
            frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
    
            # To improve performance, optionally mark the img as not writeable to
            # pass by reference.
            frame.flags.writeable = False
            results = hands.process(frame)

            # Draw the index finger annotation on the img.
            frame.flags.writeable = True
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            combine(frame) #画像合成
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    drawFingertip(frame, hand_landmarks)
            cv2.imshow('MediaPipe Hands', frame)
            if cv2.waitKey(5) & 0xFF == 27:
                break
    cap.release()

if __name__ == '__main__':
    main()
