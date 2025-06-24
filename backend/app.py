import cv2
import mediapipe as mp
import numpy as np
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import threading

result_lock = threading.Lock()
latest_detection_result = None

def draw_landmarks_on_image(bgr_image, detection_result):
    if not detection_result or not detection_result.pose_Landmarks:
        return bgr_image
    
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(bgr_image)

    for i in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[i]

        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
        ])

        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            pose_landmarks_proto,
            solutions.pose.POSE_CONNECTIONS,
            solutions.drawing_styles.get_default_pose_landmarks_style()
        )
    
    return annotated_image

def save_result_callback(result: vision.PoseLandmarkerResult, output_image: mp):
    global latest_detection_result
    with result_lock:
        latest_detection_result = result

def main():
    base_options = python.BaseOptions(model_asset_path='model/pose_landmarker_full.task')
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.LIVE_STREAM,
        output_segmentation_masks=True,
        result_callback=save_result_callback
    )

    with vision.PoseLandmarker.create_from_options(options) as detector:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: could not open webcam")
            return
        
    frame_timestamp_ms = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame")
            continue

        frame = cv2.flip(frame, 1) # flip the frame horizontally

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = np.Image(image_format=mp.ImageFormat.SRB, data=rgb_frame)

        frame_timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))

        detector.detect_async(mp_image, frame_timestamp_ms) # async

        current_result = None
        with result_lock:
            if latest_detection_result is not None:
                current_result = latest_detection_result
            
        annotated_image = draw_landmarks_on_image(frame, current_result)

        cv2.imshow("Real-Time Pose Landmarks", annotated_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if __name__ == "__main__":
        main()