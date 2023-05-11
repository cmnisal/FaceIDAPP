import cv2
import mediapipe as mp
import streamlit as st


FIVE_LANDMARKS = [470, 475, 1, 57, 287]
FACE_CONNECTIONS = mp.solutions.face_mesh_connections.FACEMESH_TESSELATION



def draw_bounding_box(img, detections, ident_names, margin=10):
    # Draw the bounding box on the original frame
    for detection, name in zip(detections, ident_names):
        
        color = (255, 0, 0) if name == "Unknown" else (0, 255, 0)

        x_coords = [
            landmark.x * img.shape[1] for landmark in detection.multi_face_landmarks.landmark
        ]
        y_coords = [
            landmark.y * img.shape[0] for landmark in detection.multi_face_landmarks.landmark
        ]

        x_min, x_max = int(min(x_coords) - margin), int(max(x_coords) + margin)
        y_min, y_max = int(min(y_coords) - margin), int(max(y_coords) + margin)

        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 2)
        cv2.rectangle(img, (x_min, y_min - img.shape[0] // 25), (x_max, y_min), color, -1)

    return img


def draw_text(
    img,
    detections,
    ident_names,
    margin=10,
    font_scale=1,
    font_color=(0, 0, 0),
    font=cv2.FONT_HERSHEY_SIMPLEX,
):
    
    font_scale = img.shape[0] / 1000
    for detection, name in zip(detections, ident_names):
        x_coords = [
            landmark.x * img.shape[1] for landmark in detection.multi_face_landmarks.landmark
        ]
        y_coords = [
            landmark.y * img.shape[0] for landmark in detection.multi_face_landmarks.landmark
        ]

        x_min = int(min(x_coords) - margin)
        y_min = int(min(y_coords) - margin)
        
        cv2.putText(
            img,
            name,
            (x_min + img.shape[0] // 400, y_min - img.shape[0] // 100),
            font,
            font_scale,
            font_color,
            2,
        )

    return img


def draw_mesh(img, detections):
    for detection in detections:
        # Draw the connections
        for connection in FACE_CONNECTIONS:
            cv2.line(
                img,
                (
                    int(detection.multi_face_landmarks.landmark[connection[0]].x * img.shape[1]),
                    int(detection.multi_face_landmarks.landmark[connection[0]].y * img.shape[0]),
                ),
                (
                    int(detection.multi_face_landmarks.landmark[connection[1]].x * img.shape[1]),
                    int(detection.multi_face_landmarks.landmark[connection[1]].y * img.shape[0]),
                ),
                (255, 255, 255),
                1,
            )

        # Draw the landmarks
        for points in detection.multi_face_landmarks.landmark:
            cv2.circle(
                img,
                (
                    int(points.x * img.shape[1]),
                    int(points.y * img.shape[0]),
                ),
                1,
                (0, 255, 0),
                -1,
            )
    return img


def draw_landmarks(img, detections):
    # Draw the face landmarks on the original frame
    for points in FIVE_LANDMARKS:
        for detection in detections:
            cv2.circle(
                img,
                (
                    int(
                        detection.multi_face_landmarks.landmark[points].x
                        * img.shape[1]
                    ),
                    int(
                        detection.multi_face_landmarks.landmark[points].y
                        * img.shape[0]
                    ),
                ),
                5,
                (0, 0, 255),
                -1,
            )
    return img
