import streamlit as st
import cv2
import time
import mediapipe as mp
import numpy as np
from skimage.transform import SimilarityTransform

FIVE_LANDMARKS = [475, 470, 1, 57, 287]


def create_text_image(width, height, text, font_scale=1.0, font_color=(0, 0, 0)):
    blank_image = np.ones((height, width, 3), np.uint8) * 255  # Create a white blank image

    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size, _ = cv2.getTextSize(text, font, font_scale, 2)  # Get text size to center it on the image
    text_x = (width - text_size[0]) // 2
    text_y = (height + text_size[1]) // 2

    cv2.putText(blank_image, text, (text_x, text_y), font, font_scale, font_color, 2)  # Put text on the image
    return blank_image


class FaceDetector:
    def __init__(
        self, shape, max_faces=1, detection_confidence=0.5, tracking_confidence=0.5
    ):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            refine_landmarks=True,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
            max_num_faces=max_faces,
        )
        self.shape = shape
        self.face_count = 0


    def face_detection(self, frame, scale, display):
        # Get the desired resize shape for image input
        self.resized_shape = (int(self.shape[1] * scale), int(self.shape[0] * scale))
        
        # bgr to rgb
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize the frame
        frame = cv2.resize(
            frame, self.resized_shape
        )

        # Display the resized input frame
        display.image(frame, channels="RGB", caption="Input Image", use_column_width=True)
        
        # Process the frame with MediaPipe Face Mesh
        results = self.face_mesh.process(frame)

        # Get number of detected faces
        self.face_count = len(results.multi_face_landmarks) if results.multi_face_landmarks else 0
        
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


class Annotator:
    def __init__(self, shape):
        self.shape = shape
        self.connections = mp.solutions.face_mesh_connections.FACEMESH_TESSELATION

    def draw_bounding_box(self, img, landmarks, margin):
        if not landmarks:
            return img
        # Draw the bounding box on the original frame
        for face_landmarks in landmarks:
            x_coords = [
                landmark.x * img.shape[1] for landmark in face_landmarks.landmark
            ]
            y_coords = [
                landmark.y * img.shape[0] for landmark in face_landmarks.landmark
            ]

            x_min, x_max = int(min(x_coords) - margin), int(max(x_coords) + margin)
            y_min, y_max = int(min(y_coords) - margin), int(max(y_coords) + margin)

            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

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
                        int(
                            face_landmarks.landmark[connection[0]].x * self.shape[1]
                        ),
                        int(
                            face_landmarks.landmark[connection[0]].y * self.shape[0]
                        ),
                    ),
                    (
                        int(
                            face_landmarks.landmark[connection[1]].x * self.shape[1]
                        ),
                        int(
                            face_landmarks.landmark[connection[1]].y * self.shape[0]
                        ),
                    ),
                    (255, 255, 255),
                    1,
                )

            # Draw the landmarks
            for face_landmark_point in face_landmarks.landmark:
                cv2.circle(
                    img,
                    (
                        int(face_landmark_point.x * self.shape[1]),
                        int(face_landmark_point.y * self.shape[0]),
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
                            * self.shape[1]
                        ),
                        int(
                            face_landmarks.landmark[face_landmark_point].y
                            * self.shape[0]
                        ),
                    ),
                    5,
                    (0, 0, 255),
                    -1,
                )
        return img


class FaceNormalizer:
    def __init__(self, shape, target_size=(112, 112)):
        self.target_size = target_size
        self.shape = shape

    def normalizer(self, img, detections, display):
        if not detections[0]:
            display.image(create_text_image(self.target_size[0], self.target_size[0], "?", 2), channels="BGR", caption="Normalized Face 1")
            return []
        
        faces = []
        for detection in detections[0]:
            faces.append(self.normalize(img, detection.landmark))

        display.image(faces[0], channels="BGR", caption="Normalized Face 1")
        return faces
    
    def normalize(self, img, landmarks):
        src = np.array(
            [[landmarks[i].x * self.shape[1], landmarks[i].y * self.shape[0]] for i in FIVE_LANDMARKS],
        )

        dst = np.array(
            [
                [38.2946, 51.6963],
                [73.5318, 51.5014],
                [56.0252, 71.7366],
                [41.5493, 92.3655],
                [70.7299, 92.2041],
            ],
            dtype=np.float32,
        )
        tform = SimilarityTransform()
        tform.estimate(src, dst)
        tmatrix = tform.params[0:2, :]
        return cv2.warpAffine(img, tmatrix, self.target_size, borderValue=0.0)


def main():
    # Set the page layout -------------------------------------------------------
    st.set_option("deprecation.showfileUploaderEncoding", False)
    st.markdown(
        """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 400px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 400px;
        margin-left: -400px;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )
    # ---------------------------------------------------------------------------

    # SIDEBAR -------------------------------------------------------------------
    st.sidebar.markdown("## Preferences")

    st.sidebar.markdown("---")
    st.sidebar.markdown("## Face Detection")
    max_faces = st.sidebar.number_input("Maximum Number of Faces", value=1, min_value=1)
    detection_confidence = st.sidebar.slider(
        "Min Detection Confidence", min_value=0.0, max_value=1.0, value=0.9
    )
    tracking_confidence = st.sidebar.slider(
        "Min Tracking Confidence", min_value=0.0, max_value=1.0, value=0.9
    )
    margin = st.sidebar.slider("Bounding box margin", 0, 100, 25, 1)
    scale = st.sidebar.slider(
        "Scale for preprocessing", min_value=0.0, max_value=1.0, value=0.5
    )
    st.sidebar.markdown("---")
    # ---------------------------------------------------------------------------

    # Main Window ---------------------------------------------------------------
    st.title("Face Recognition Demonstration App")
    st.markdown("---")

    output_window = st.empty()
    st.markdown("---")
    process_window = st.empty()
    st.markdown("---")
    face_window = st.empty()

    vid = cv2.VideoCapture(0)
    shape = (
        int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
    )

    face_detector = FaceDetector(
        shape,
        max_faces=max_faces,
        detection_confidence=detection_confidence,
        tracking_confidence=tracking_confidence,
    )

    face_normalizer = FaceNormalizer(shape=shape)

    annotator = Annotator(shape=shape)

    fps = 0
    i = 0

    # DASHBOARD KPIs
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi5, kpi6, kpi7 = st.columns(3)

    with kpi1:
        st.markdown("**FrameRate**")
        kpi1_text = st.markdown("0")

    with kpi2:
        st.markdown("**Detected Faces**")
        kpi2_text = st.markdown("0")

    with kpi3:
        st.markdown("**Image Dims**")
        kpi3_text = st.markdown("0")

    with kpi4:
        st.markdown("**Face Detection [ms]**")
        kpi4_text = st.markdown("0")

    with kpi5:
        st.markdown("**Annotations [ms]**")
        kpi5_text = st.markdown("0")

    with kpi6:
        st.markdown("**Image Resized Dims**")
        kpi6_text = st.markdown("0")
    
    with kpi7:
        st.markdown("**Face Normalization [ms]**")
        kpi7_text = st.markdown("0")

    st.markdown("<hr/>", unsafe_allow_html=True)
    # ---------------------------------------------------------------------------

    prevTime = 0
    while vid.isOpened():
        i += 1
        ret, frame = vid.read()
        if not ret:
            continue

        # FACE DETECTION ---------------------------------------------------------
        # Stop the time for the following operations
        start_time = time.time()
        detections = face_detector.face_detection(frame, scale, display=process_window)
        end_time = time.time()
        time_face_detection = end_time - start_time

        # FACE NORMALIZATION ------------------------------------------------------
        start_time = time.time()
        faces = face_normalizer.normalizer(frame, detections, display=face_window)
        end_time = time.time()
        time_face_normalization = end_time - start_time

        # ANNOTATIONS ------------------------------------------------------------
        start_time = time.time()
        frame.flags.writeable = True # make them faster
        frame = annotator.draw_mesh(frame, detections[0])
        frame = annotator.draw_landmarks(frame, detections[0])
        frame = annotator.draw_bounding_box(frame, detections[0], margin)
        output_window.image(
            frame, channels="BGR", caption="Output", use_column_width=True
        )
        end_time = time.time()
        time_annotations = end_time - start_time

        # TODO Get the Bounding Box from Landmarks

        # TODO Face Alignment using the specific Landmarks -> Look which are used and then use my face warp method

        # TODO Face Recognition -> How to get the Face Recognition Network running in here?

        # CALCULATE FPS ----------------------------------------------------------
        currTime = time.time()
        fps = 1 / (currTime - prevTime)
        prevTime = currTime

        # UPDATE DASHBOARD -------------------------------------------------------
        kpi1_text.write(
            f"<h1 style='text-align: center; color: red;'>{int(fps)}</h1>",
            unsafe_allow_html=True,
        )
        kpi2_text.write(
            f"<h1 style='text-align: center; color: red;'>{face_detector.face_count}</h1>",
            unsafe_allow_html=True,
        )
        kpi3_text.write(
            f"<h1 style='text-align: center; color: red;'>{face_detector.shape}</h1>",
            unsafe_allow_html=True,
        )
        kpi4_text.write(
            f"<h1 style='text-align: center; color: red;'>{time_face_detection * 1000:.2f}</h1>",
            unsafe_allow_html=True,
        )
        kpi5_text.write(
            f"<h1 style='text-align: center; color: red;'>{time_annotations * 1000:.2f}</h1>",
            unsafe_allow_html=True,
        )
        kpi6_text.write(
            f"<h1 style='text-align: center; color: red;'>{face_detector.resized_shape}</h1>",
            unsafe_allow_html=True,
        )
        kpi7_text.write(
            f"<h1 style='text-align: center; color: red;'>{time_face_normalization * 1000:.2f}</h1>",
            unsafe_allow_html=True,
        )

    vid.release()


if __name__ == "__main__":
    main()
