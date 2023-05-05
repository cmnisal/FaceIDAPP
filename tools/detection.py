import cv2
import mediapipe as mp


class FaceDetector:
    def __init__(
        self, shape, max_faces=2, detection_confidence=0.5, tracking_confidence=0.5
    ):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            refine_landmarks=True,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
            max_num_faces=max_faces,
        )
        self.shape = shape
        self.face_count = 0

    def detect_faces(self, frame, scale, display):
        # Get the desired resize shape for image input
        self.resized_shape = (int(self.shape[1] * scale), int(self.shape[0] * scale))

        # Resize the frame
        frame = cv2.resize(frame, self.resized_shape)

        # Display the resized input frame
        if display is not None:
            display.image(
                frame, channels="RGB", caption="Input Image", use_column_width=True
            )

        # Process the frame with MediaPipe Face Mesh
        results = self.face_mesh.process(frame)

        # Get number of detected faces
        self.face_count = (
            len(results.multi_face_landmarks) if results.multi_face_landmarks else 0
        )

        # Get the Bounding Boxes from the detected faces
        bboxes = []
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                x_coords = [
                    landmark.x * frame.shape[1] for landmark in face_landmarks.landmark
                ]
                y_coords = [
                    landmark.y * frame.shape[0] for landmark in face_landmarks.landmark
                ]

                x_min, x_max = int(min(x_coords)), int(max(x_coords))
                y_min, y_max = int(min(y_coords)), int(max(y_coords))

                bboxes.append((x_min, y_min, x_max, y_max))

        return results.multi_face_landmarks, bboxes
