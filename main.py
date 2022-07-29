import cv2
import mediapipe as mp
import time

# webcam selected
cap = cv2.VideoCapture(0)

# hand tracking
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0


# code to run webcam
while True:
    success, img = cap.read()
    #converts input to an RGB image
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # sends an RBG image to the hands
    results = hands.process(imgRGB)
    # to test that the hand is being detected print(results.multi_hand_landmarks)

    #get information from each hand
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                # print(id, lm)
                #find coordinates for specific landmarks
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                print(id, cx, cy)
                #detect a specific landmark
                if id == 0:
                    cv2.circle(img, (cx, cy), 25, (255,0,255), cv2.FILLED)


            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
    #Finds the frames per second
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    #displays frames per second
    cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3,
        (255,0,255), 3)


    cv2.imshow("Image", img)
    cv2.waitKey(1)
