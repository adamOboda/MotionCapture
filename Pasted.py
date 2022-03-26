import cv2
import numpy as np
from cv2 import destroyAllWindows
import mediapipe as mp

mpdrawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

def rescale_frame(image, percent=75):
    width = int(image.shape[1] * percent/ 50)
    height = int(image.shape[0] * percent/ 20)
    dim = (width, height)
    return cv2.resize(image, dim, interpolation =cv2.INTER_AREA)


angle_min = []
angle_min_hip = []
angle_min_ankle = []


# Curl counter variables
counter = 0
min_ang = 0
max_ang = 0
min_ang_hip = 0
max_ang_hip = 0
min_ang_ankle = 0
max_ang_ankle = 0
stage = None


capture = cv2.VideoCapture('PoseVideos/4.mp4')
size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 'x264' doesn't work
out = cv2.VideoWriter('PoseVideos/2_output.mp4',fourcc, 60.0, size)  # 'False' for 1-ch instead of 3-ch for color
# fgbg= cv2.createBackgroundSubtractorMOG2()

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while capture.isOpened():  #while True:
        ret, img = capture.read()

        if img is not None:
            frame_ = rescale_frame(img, percent=70)






        # ERROR Z KONWERSJA PLIKU WYSTEPUJE PONIZEJ !!!!

        image = cv2.cvtColor(frame_, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # END OF ERROR

        try:
            landmarks = results.pose_landmarks.landmark

            shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
            ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                     landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
            heel = [landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].x,
                     landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].y]
            foot_index = [landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].x,
                     landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y]

            angle = calculate_angle(shoulder, elbow, wrist)
            angle = round(angle, 1)

            angle_knee = calculate_angle(hip, knee, ankle)
            angle_knee = round(angle_knee, 1)

            angle_hip = calculate_angle(shoulder, hip, knee)
            angle_hip = round(angle_hip, 1)

            angle_ankle = calculate_angle(knee, heel, foot_index)
            angle_ankle = round(angle_ankle, 1)

            hip_angle = 180 - angle_hip
            knee_angle = 180 - angle_knee



            angle_min.append(angle_knee)
            angle_min_hip.append(angle_hip)

            # # Real time measurement - elbow
            # cv2.putText(image, str(angle),
            #             tuple(np.multiply(elbow, [640, 600]).astype(int)),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA
            #                                 )
            # Real time measurement - knee
            cv2.putText(image, str(angle_knee),
                        tuple(np.multiply(knee, [300, 750]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA
                        )
            # Real time measurement - hip
            cv2.putText(image, str(angle_hip),
                        tuple(np.multiply(hip, [300, 700]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA
                        )
            # Real Time measurement - Ankle
            cv2.putText(image, str(angle_ankle),
                        tuple(np.multiply(heel, [300, 770]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA
                        )
            min_ang = min(angle_min)
            min_ang_hip = min(angle_min_hip)
            if angle_knee > 169:
                stage = "up"
                # counter = 0
                # min_ang_hip = 0
                # min_ang = 0
            if angle_knee <= angle_knee < 90 and stage == 'up':
                stage = "down"
                counter += 1
                print(counter)
                min_ang = min(angle_min)
                max_ang = max(angle_min)

                min_ang_hip = min(angle_min_hip)
                max_ang_hip = max(angle_min_hip)

                # min_ang_ankle = min(angle_min_ankle)
                # max_ang_ankle = max(angle_ankle)

                print(min(angle_min), " _ ", max(angle_min))
                print(min(angle_min_hip), " _ ", max(angle_min_hip))
                print(min(angle_min_ankle), " _ ", max(angle_min_ankle))
                angle_min = []
                angle_min_hip = []
                # angle_min_ankle = []



        except:
            pass

        # COUNTERS
        cv2.rectangle(image, (5, 5), (400, 110), (0, 0, 0,), -1)
        # Reps Counter
        cv2.putText(image, "Rep : " + str(counter),
                    (30, 30),
        # Knee Joint Angle
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(image, "Min.knee-joint angle: " + str(min_ang),
                    (30, 60),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 1, cv2.LINE_AA)
        # Hip Joint Angle
        cv2.putText(image, "Min.hip-joint angle: " + str(min_ang_hip),
                    (30, 90),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 1, cv2.LINE_AA)
        # Ankle Joint Angle
        # cv2.putText(image, "Min.ankle-joint angle: " + str(min_ang_hip),
        #             (30, 160),
        #             cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        # # Stage data
        # cv2.putText(image, 'STAGE', (65,12),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)


        mpdrawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mpdrawing.DrawingSpec(color=(0, 0, 0), thickness=2, circle_radius=2),
                                  mpdrawing.DrawingSpec(color=(203, 17, 17), thickness=2, circle_radius=2)
                                  )
        # cv2.imshow("Squat Analysis", img)
        # out.write(img)
        # cv2.imshow('img', img)
        # if ret==True:
        #     fgmask = fgbg.apply(img)
        #     out.write(img)
        #     cv2.imshow('Squat',image)

        if ret:

            cv2.waitKey(1) & 0xFF == ord('q')
            cv2.imshow("Squat Analysis", img)
            out.write(img)

        else:

            # printerror = print('no video')
            capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        if counter > 8 and stage == "up":
            capture.release()
            cv2.destroyAllWindows()

        #if(cv2.waitKey(27)!=-1):  # observed it will close the imshow window immediately
        #    break                 # so change to below
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    capture.release()
    out.release()
    cv2.destroyAllWindows()

