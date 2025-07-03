import cv2
import mediapipe as mp
import numpy as np
import time

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

app_mode = "STARTING" # STARTING, COLLECTING, INFERENCING
model = None
training_data = []

annotated_image = None
latest_features = None

MODEL_FILE = 'slouch_model.txt'
DATA_FILE = 'slouch_training_data.csv'
FEATURE_COLUMNS = [
    'head_shoulder_z_diff',
    'head_shoulder_y_diff',
    'posture_angle',
    'head_tilt_z_diff'
]

LABEL_SLOUCHING = 0
LABEL_UP_STRAIGHT = 1
LABEL_MAP = {
    LABEL_SLOUCHING: "Slouching",
    LABEL_UP_STRAIGHT: "Up Straight"
}

FACE_LANDMARKS_INDICES = list(range(11))
SHOULDER_LANDMARKS_INDICES = [11, 12]
REQUIRED_LANDMARKS_FOR_FEATURES = [0, 7, 8, 9, 10, 11, 12]

KEY_UP = 2490368
KEY_DOWN = 2621440
KEY_T = ord('t')
KEY_W = ord('w')
KEY_Q = ord('q')



def extract_features(pose_landmarks_list):
    pass


def draw_and_process_result(result: vision.PoseLandmarkerResult, output_image: mp, timestamp_ms: int):
    global annotated_image, latest_features, app_mode, model

    annotated_image_np = output_image.numpy_view().copy()
    prediction_text = ""
    text_color = (0, 255, 0)

    latest_features = extract_features(result.pose_landmarks)

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

        if app_mode == "INFERENCING" and model is not None and latest_features is not None:
            

    annotated_image = annotated_image_np


def main():
    global app_mode, model, training_data

    base_options = python.BaseOptions(model_asset_path='model/pose_landmarker_full.task')
    # base_options = python.BaseOptions(model_asset_path='model/pose_landmarker_heavy.task')
    # base_options = python.BaseOptions(model_asset_path='model/pose_landmarker_lite.task')
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.LIVE_STREAM,
        output_segmentation_masks=False,
        result_callback=draw_and_process_result
    )

    with vision.PoseLandmarker.create_from_options(options) as detector:
        # model loading
        if os.path.exists(MODEL_FILE):
            try:
                model = lgb.Booster(model_file=MODEL_FILE)
                lgbm = lgb.LGBMClassifier()
                lgbm._Booster = model
                lgbm._n_classes = 2
                lgbm.fitted_ = True
                model = lgbm
                app_mode = "INFERENCING"
                print("loaded existing model, starting INFERENCING mode")
            except Exception as e:
                print("could not load model, starting fresh")
                app_mode= "COLLECTING"
        else:
            print("no model file found, starting fresh")
            app_mode= "COLLECTING"

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("error: could not open webcam")
            return
        
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
                display_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
                key = cv2.waitKey(5)

                if key == KEY_Q:
                    break

                # data collection mode
                if app_mode == "COLLECTING":
                    if key == KEY_UP:
                        if latest_features is not None:
                            training_data.append(np.append(latest_features, LABEL_UP_STRAIGHT))
                            print(f"captured 'up straight' sample. Total: {len(training_data)}")
                    elif key == KEY_DOWN:
                        if latest_features is not None:
                            training_data.append(np.append(latest_features, LABEL_SLOUCHING))
                            print(f"captured 'slouching' sample. Total: {len(training_data)}")
                    elif key == KEY_T:
                        up_count = sum(1 for item in training_data if item[-1] == LABEL_UP_STRAIGHT)
                        slouch_count = sum(1 for item in training_data if item[-1] == LABEL_SLOUCHING)
                        if up_count > 5 and slouch_count > 5:
                            print("training model...")
                            df = pd.DataFrame(training_data, columns=FEATURE_COLUMNS + ['label'])
                            X = df[FEATURE_COLUMNS]
                            y = df['label'].astype(int)

                            model = lgb.LGBMClassifier(objective='binary')
                            model.fit(X, y)

                            app_mode = "INFERENCE"
                            print("training complete. Switching to INFERENCING mode")
                        else:
                            print("not enough data to train on")

                elif app_mode == "INFERENCING":
                    if key == KEY_W:
                        # save data
                        if model is not None:
                            model.booster_.save_model(MODEL_FILE)
                            df = pd.DataFrame(training_data, FEATURE_COLUMNS + ['label'])
                            df.to_csv(DATA_FILE, index=False)

                cv2.imshow("Posture Reminder", display_image)

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()