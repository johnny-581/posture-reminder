import cv2
import mediapipe as mp
import numpy as np
import time

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

annotated_image = None

FACE_LANDMARKS_INDICES = list(range(11))
SHOULDER_LANDMARKS_INDICES = [11, 12]

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

            # print(f"{solutions.pose.PoseLandmark.LEFT_HIP}")

            z_difference_text = draw_z_values(pose_landmarks, annotated_image_np)
    
    cv2.putText(
        img=annotated_image_np,
        text=z_difference_text,
        org=(10, 30),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.8,
        # fontScale=font_scale,
        color=(0, 255, 0),
        thickness=2,
        lineType=cv2.LINE_AA
    )

    annotated_image = annotated_image_np


def draw_z_values(pose_landmarks, annotated_image_np):
    image_height, image_width, _ = annotated_image_np.shape

    face_z_values = [
        pose_landmarks[i].z for i in FACE_LANDMARKS_INDICES
        if pose_landmarks[i].visibility > 0.5
    ]

    shoulder_z_values = [
        pose_landmarks[i].z for i in SHOULDER_LANDMARKS_INDICES
        if pose_landmarks[i].visibility > 0.5
    ]

    if face_z_values and shoulder_z_values:
        average_face_z = np.mean(face_z_values)
        average_shoulder_z = np.mean(shoulder_z_values)
        z_difference = average_face_z - average_shoulder_z
        z_difference_text = f"Head-Shoulder Z-Diff: {z_difference:.2f}"

        font_scale = abs(z_difference) * 2

    for landmark in pose_landmarks:
        if landmark.visibility > 0.5:
            # Convert normalized coordinates to pixel coordinates
            pixel_x = int(landmark.x * image_width)
            pixel_y = int(landmark.y * image_height)
            
            z_text = f"z: {landmark.z:.1f}"
            
            # Put the text on the image near the landmark
            cv2.putText(
                img=annotated_image_np,
                text=z_text,
                org=(pixel_x + 10, pixel_y), # Offset text to not overlap the landmark
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.4,
                color=(255, 255, 255),
                thickness=1
            )

    return z_difference_text


def main():
    base_options = python.BaseOptions(model_asset_path='model/pose_landmarker_full.task')
    # base_options = python.BaseOptions(model_asset_path='model/pose_landmarker_heavy.task')
    # base_options = python.BaseOptions(model_asset_path='model/pose_landmarker_lite.task')
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.LIVE_STREAM,
        output_segmentation_masks=False,
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