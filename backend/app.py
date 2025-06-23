from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import cv2

def draw_landmarks_on_image(rgb_image, detection_result):
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)

    for i in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[i]

        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
        ])

        bgr_image_to_draw_on = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)

        solutions.drawing_utils.draw_landmarks(
            bgr_image_to_draw_on,
            pose_landmarks_proto,
            solutions.pose.POSE_CONNECTIONS,
            solutions.drawing_styles.get_default_pose_landmarks_style()
        )
    
    return bgr_image_to_draw_on

base_options = python.BaseOptions(model_asset_path='model/pose_landmarker_full.task')
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=True
)
detector = vision.PoseLandmarker.create_from_options(options)
image = mp.Image.create_from_file("image.jpg")

detection_result = detector.detect(image)

annotated_image_bgr = draw_landmarks_on_image(image.numpy_view(), detection_result)

cv2.imshow("Annotated Image", annotated_image_bgr)
cv2.waitKey(0)
cv2.destroyAllWindows()