import cv2
import mediapipe as mp


FIVE_LANDMARKS = [470, 475, 1, 57, 287]


class Annotator:
    def __init__(self):
        self.connections = mp.solutions.face_mesh_connections.FACEMESH_TESSELATION

    def draw_bounding_box(self, img, landmarks, recognition, margin):
        if not landmarks:
            return img
        

        # Draw the bounding box on the original frame
        for face_landmarks, identity in zip(landmarks, recognition):
            
            color = (255, 0, 0) if identity == "Unknown" else (0, 255, 0)

            x_coords = [
                landmark.x * img.shape[1] for landmark in face_landmarks.landmark
            ]
            y_coords = [
                landmark.y * img.shape[0] for landmark in face_landmarks.landmark
            ]

            x_min, x_max = int(min(x_coords) - margin), int(max(x_coords) + margin)
            y_min, y_max = int(min(y_coords) - margin), int(max(y_coords) + margin)

            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 2)
            cv2.rectangle(img, (x_min, y_min - img.shape[0] // 25), (x_max, y_min), color, -1)

        return img

    def draw_text(
        self,
        img,
        landmarks,
        names,
        margin,
        font_scale=1,
        font_color=(0, 0, 0),
        font=cv2.FONT_HERSHEY_SIMPLEX,
    ):
        
        font_scale = img.shape[0] / 1000
        if not landmarks:
            return img
        for face_landmarks, name in zip(landmarks, names):
            x_coords = [
                landmark.x * img.shape[1] for landmark in face_landmarks.landmark
            ]
            y_coords = [
                landmark.y * img.shape[0] for landmark in face_landmarks.landmark
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

    def draw_mesh(self, img, landmarks):
        if not landmarks:
            return img
        for face_landmarks in landmarks:
            # Draw the connections
            for connection in self.connections:
                cv2.line(
                    img,
                    (
                        int(face_landmarks.landmark[connection[0]].x * img.shape[1]),
                        int(face_landmarks.landmark[connection[0]].y * img.shape[0]),
                    ),
                    (
                        int(face_landmarks.landmark[connection[1]].x * img.shape[1]),
                        int(face_landmarks.landmark[connection[1]].y * img.shape[0]),
                    ),
                    (255, 255, 255),
                    1,
                )

            # Draw the landmarks
            for face_landmark_point in face_landmarks.landmark:
                cv2.circle(
                    img,
                    (
                        int(face_landmark_point.x * img.shape[1]),
                        int(face_landmark_point.y * img.shape[0]),
                    ),
                    1,
                    (0, 255, 0),
                    -1,
                )
        return img

    def draw_landmarks(self, img, landmarks):
        if not landmarks:
            return img
        # Draw the face landmarks on the original frame
        for face_landmark_point in FIVE_LANDMARKS:
            for face_landmarks in landmarks:
                cv2.circle(
                    img,
                    (
                        int(
                            face_landmarks.landmark[face_landmark_point].x
                            * img.shape[1]
                        ),
                        int(
                            face_landmarks.landmark[face_landmark_point].y
                            * img.shape[0]
                        ),
                    ),
                    5,
                    (0, 0, 255),
                    -1,
                )
        return img
