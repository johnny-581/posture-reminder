import cv2
import mediapipe as mp
import numpy as np
import time

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


annotated_image = None

def draw_and_save_result(result: vision.PoseLandmarkerResult, output_image: mp, timestamp_ms: int):
    global annotated_image
    annotated_image_np = output_image.numpy_view().copy()

    if result.pose_landmarks:
        for pose_landmarks in result.pose_landmarks:
            pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            pose_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
            ])

            solutions.drawing_utils.draw_landmarks(
                annotated_image_np,
                pose_landmarks_proto,
                solutions.pose.POSE_CONNECTIONS,
                solutions.drawing_styles.get_default_pose_landmarks_style()
            )
        
    annotated_image = annotated_image_np

def main():
    base_options = python.BaseOptions(model_asset_path='model/pose_landmarker_full.task')
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.LIVE_STREAM,
        output_segmentation_masks=True,
        result_callback=draw_and_save_result
    )

    with vision.PoseLandmarker.create_from_options(options) as detector:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("error: could not open webcam")
            return
        
        global annotated_image
        success, frame = cap.read()
        if success:
            annotated_image = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("ignoring empty cemra frame")
                continue

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format = mp.ImageFormat.SRGB, data = rgb_frame)
            frame_timestemp_ms = int(time.time() * 1000)
            detector.detect_async(mp_image, frame_timestemp_ms)

            if annotated_image is not None:
                bgr_display_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
                cv2.imshow("Real-Time Pose Landmarks", bgr_display_image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()