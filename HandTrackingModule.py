import time
import cv2 
import mediapipe as mp


class handDetector():
    def __init__(self, mode=False, maxHands=2,complexity=1, detectionCon=0.5, trackCon=0.5): 
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.tackCon = trackCon
        # initializations
        self.comlexity = complexity
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.comlexity, self.detectionCon, self.tackCon)
        self.mpDraw = mp.solutions.drawing_utils
        
    def findHands(self, Image, draw=True):
        imgRGB = cv2.cvtColor(Image, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLandmarks in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(Image, handLandmarks, self.mpHands.HAND_CONNECTIONS)
        return Image

    def findPosition(self,Image, handNo=0, draw=True):
        lmList=[]
        if self.results.multi_hand_landmarks:
            myHand= self.results.multi_hand_landmarks[handNo]
            for idNumber, landmarkInformation in enumerate(myHand.landmark):
                h, w, c = Image.shape
                cx, cy = int(landmarkInformation.x * w), int(landmarkInformation.y * h)
                print(idNumber, cx, cy)
                lmList.append([idNumber,cx,cy])

            if idNumber == 4:
                if draw:
                    if idNumber == 0:
                        cv2.circle(Image, (cx, cy), 25, (5, 5, 5), cv2.FILLED)
                    else:
                        cv2.circle(Image, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
        return lmList

    def fingersUp(self):
        fingers = []


        #Thumb 
        if self.lmList[self.tipIds[0]][1] > self.lmList[self.tipIds[0] -1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        #Fingers
        for id in range(1, 5):
            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] -2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers



def main():
    previousTime = 0
    currentTime = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector()

    while True:
        success, Image = cap.read()
        Image = detector.findHands(Image)
        lmList = detector.findPosition(Image)
        if len(lmList)!=0:
            print(lmList[4])

        currentTime = time.time()
        fps = 1 / (currentTime - previousTime)
        previousTime = currentTime

        cv2.putText(Image, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 255), 3)
        cv2.imshow("Image", Image)
        cv2.waitKey(1)


if __name__ == '__main__':
    main()
