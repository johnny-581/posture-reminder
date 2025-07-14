import cv2
import mediapipe as mp
import numpy as np
import time
import pandas as pd
import lightgbm as lgb
import os
import joblib

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

app_mode = "STARTING" # STARTING, COLLECTING, INFERENCING
model = None
training_data = []

annotated_image = None
latest_features = None

MODEL_FILE = 'slouch_model.txt'
DATA_FILE = 'slouch_training_data.csv'

FEATURE_SETS = {
    "set1": [
        'head_shoulder_z_diff',
        'head_shoulder_y_diff',
        'posture_angle',
        'head_tilt_z_diff',
    ],
    "set2": [
        'head_shoulder_z_diff',
        'head_shoulder_y_diff',
        'posture_angle',
    ],
    "set3": [
        'head_shoulder_z_diff',
        'head_shoulder_y_diff',
    ],
    "set4": [
        'posture_angle',
    ],
    "set5": [
        'head_shoulder_z_diff',
        'head_shoulder_y_diff',
        ''
    ]
}

FEATURE_COLUMNS = [
        'head_shoulder_z_diff',
        'head_shoulder_y_diff',
        'posture_angle',
        'head_tilt_z_diff',
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

KEY_UP = 0
KEY_DOWN = 1
KEY_T = ord('t')
KEY_W = ord('w')
KEY_Q = ord('q')



def extract_features(pose_landmarks_list):
    if not pose_landmarks_list:
        return None
    
    p = pose_landmarks_list[0] # the first detected person

    for i in REQUIRED_LANDMARKS_FOR_FEATURES:
        if p[i].visibility < 0.7:
            return None
    
    left_shoulder = np.array([p[12].x, p[12].y, p[12].z])
    right_shoulder = np.array([p[11].x, p[11].y, p[11].z])
    left_ear = np.array([p[8].x, p[8].y, p[8].z])
    right_ear = np.array([p[7].x, p[7].y, p[7].z])
    nose = np.array([p[0].x, p[0].y, p[0].z])
    mouth_left = np.array([p[10].x, p[10].y, p[10].z])
    mouth_right = np.array([p[9].x, p[9].y, p[9].z])

    shoulder_center = (left_shoulder + right_shoulder) / 2
    head_center = (left_ear + right_ear) / 2
    mouth_center = (mouth_left + mouth_right) / 2
    
    shoulder_width = np.linalg.norm(left_shoulder - right_shoulder)
    if shoulder_width < 1e-6:
        return None
    
    head_shoulder_z_diff = (head_center[2] - shoulder_center[2]) / shoulder_width
    head_shoulder_y_diff = (head_center[1] - shoulder_center[1]) / shoulder_width

    vec_y = head_center[1] - shoulder_center[1]
    vec_z = head_center[2] - shoulder_center[2]
    posture_angle = np.arctan2(vec_y, vec_z)

    head_tilt_z_diff = (nose[2] - mouth_center[2]) / shoulder_width

    return np.array([
        head_shoulder_z_diff,
        head_shoulder_y_diff,
        posture_angle,
        head_tilt_z_diff
    ])


def draw_and_process_result(result: vision.PoseLandmarkerResult, output_image: mp, timestamp_ms: int):
    global annotated_image, latest_features, app_mode, model

    annotated_image_np = output_image.numpy_view().copy()
    prediction_text = ""
    text_color = (0, 255, 0) # green

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

        # prediction
        if app_mode == "INFERENCING" and model is not None and latest_features is not None:
            prediction_df = pd.DataFrame([latest_features], columns=FEATURE_COLUMNS)
            prediction = model.predict(prediction_df)
            predicted_label = prediction[0]
            prediction_text = f"Prediction: {LABEL_MAP[predicted_label]}"
            if predicted_label == LABEL_SLOUCHING:
                text_color = (0, 0, 255) # red
        
    if app_mode == "COLLECTING":
        ui_text = "Mode: COLLECTING | UP_ARROW: Up-Straight | DOWN_ARROW: Slouch | T: Train"
    elif app_mode == "INFERENCING":
        ui_text = "Mode: INFERENCING | W: Save Model | Q: Quit"
    else:
        ui_text = "Initializing..."

    # ui text
    cv2.putText(annotated_image_np, ui_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # collected data count
    up_count = sum(1 for item in training_data if item[-1] == LABEL_UP_STRAIGHT)
    slouch_count = sum(1 for item in training_data if item[-1] == LABEL_SLOUCHING)
    cv2.putText(annotated_image_np, f"Up: {up_count} | Slouch: {slouch_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # prediction text
    if prediction_text:
        cv2.putText(annotated_image_np, prediction_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)

    annotated_image = annotated_image_np


def main():
    global app_mode, model, training_data

    base_options = python.BaseOptions(model_asset_path='model/pose_landmarker_full.task')
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
                model = joblib.load(MODEL_FILE)
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

            if annotated_image is None:
                display_image = frame
            else:
                display_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
            
            key = cv2.waitKey(5)

            if key == KEY_Q:
                break

            # data collection mode
            if app_mode == "COLLECTING":
                if key == KEY_UP:
                    print("up pressed")
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
                        print("------starting feature set evaluation------")
                        ALL_FEATURE_COLUMNS = [
                            'head_shoulder_z_diff',
                            'head_shoulder_y_diff',
                            'posture_angle',
                            'head_tilt_z_diff'
                        ]
                        df = pd.DataFrame(training_data, columns=ALL_FEATURE_COLUMNS + ['label'])
                        X = df[ALL_FEATURE_COLUMNS]
                        y = df['label'].astype(int)

                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

                        results = {}

                        for set_name, feature_columns in FEATURE_SETS.items():
                            print(f"\n---training and evaluating for {set_name}---")
                            X_train_subset = X_train[feature_columns]
                            X_test_subset = X_test[feature_columns]

                            model = lgb.LGBMClassifier(objective='binary')
                            model.fit(X_train_subset, y_train)

                            y_predict = model.predict(X_test_subset)

                            accuracy = accuracy_score(y_test, y_predict)
                            report = classification_report(y_test, y_predict, target_names=LABEL_MAP.values())

                            print(f"Accuracy: {accuracy:.4f}")
                            print("Classification Report:")
                            print(report)

                            results[set_name] = {
                                'accuracy': accuracy,
                                'model': model,
                                'features': feature_columns
                            }
                        
                        best_set_name = max(results, key=lambda k: results[k]['accuracy'])
                        model = results[best_set_name]['model']
                        FEATURE_COLUMNS = results[best_set_name]['features']

                        print(f"---Best feature set is '{best_set_name}' with accuracy {results[best_set_name]['accuracy']:.4f}")

                        print("Switching to INFERENCING mode with the best model")
                        app_mode = "INFERENCING"
                    else:
                        print("not enough data to train on")

            elif app_mode == "INFERENCING":
                if key == KEY_W:
                    # save data
                    if model is not None:
                        joblib.dump(model, MODEL_FILE)
                        print(f"model saved to {MODEL_FILE}!")

                        if training_data:
                            df = pd.DataFrame(training_data, columns=FEATURE_COLUMNS + ['label'])
                            df.to_csv(DATA_FILE, index=False)
                            print(f"training data saved to {DATA_FILE}!")

            cv2.imshow("Posture Reminder", display_image)

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()