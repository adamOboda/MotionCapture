import cv2
import mediapipe as mp

class poseDetector():

    def __init__(self, mode=False, modelCompl = 1, smoothLand = True, enablSegm = False, smooth=True, detectionCon = 0.5, trackCon=0.5):
        # static_image_mode = False,
        # model_complexity = 1,
        # smooth_landmarks = True,
        # enable_segmentation = False,
        # smooth_segmentation = True,
        # min_detection_confidence = 0.5,
        # min_tracking_confidence = 0.5):
        self.mode = mode
        self.modelCompl = modelCompl
        self.smoothLand = smoothLand
        self.smooth = smooth
        self.enablSegm = enablSegm
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.modelCompl, self.smoothLand, self.enablSegm, self.smooth, self.detectionCon, self.trackCon)

    def findPose(self, img, draw = True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        return img

def main():
    cap = cv2.VideoCapture('poseVideos/2.mp4')
    detector = poseDetector()
    while True:
        succes, img = cap.read()
        img = detector.findPose(img)
        cv2.imshow("Image", img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()