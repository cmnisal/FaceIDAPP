from .nametypes import Detection, Identity
import numpy as np
import cv2
from sklearn.metrics.pairwise import cosine_distances
from skimage.transform import SimilarityTransform


def detect_faces(frame, model):
    # Process the frame with MediaPipe Face Mesh
    results = model.process(frame)

    # Get the Bounding Boxes from the detected faces
    detections = []
    if results.multi_face_landmarks:
        for face in results.multi_face_landmarks:
            xs = [landmark.x for landmark in face.landmark]
            ys = [landmark.y for landmark in face.landmark]
            bbox = [min(xs), min(ys), max(xs), max(ys)]

            FIVE_LANDMARKS = [470, 475, 1, 57, 287]

            landmarks = [[face.landmark[i].x, face.landmark[i].y] for i in FIVE_LANDMARKS]

            detections.append(Detection(bbox=bbox, landmarks=landmarks))
    return detections


def align(img, landmarks, target_size=(112, 112)):
    # Transform to Landmark-Coordinates from relative landmark positions
    dst = np.asarray(landmarks) * img.shape[:2][::-1]

    # Target Landmarks-Coordinates from ArcFace Paper
    src = np.array(
        [
            [38.2946, 51.6963],
            [73.5318, 51.5014],
            [56.0252, 71.7366],
            [41.5493, 92.3655],
            [70.7299, 92.2041],
        ],
        dtype=np.float32,
    )

    # Estimate the transformation matrix
    tform = SimilarityTransform()
    tform.estimate(dst, src)
    tmatrix = tform.params[0:2, :]

    # Apply the transformation matrix
    img = cv2.warpAffine(img, tmatrix, target_size, borderValue=0.0)

    return img


def align_faces(img, detections):
    updated_detections = []
    for detection in detections:
        updated_detections.append(detection._replace(face=align(img, detection.landmarks)))
    return updated_detections


def inference(detections, model):
    updated_detections = []
    faces = [detection.face for detection in detections if detection.face is not None]

    if len(faces) > 0:
        faces = np.asarray(faces).astype(np.float32) / 255
        model.resize_tensor_input(model.get_input_details()[0]["index"], faces.shape)
        model.allocate_tensors()
        model.set_tensor(model.get_input_details()[0]["index"], faces)
        model.invoke()
        embs = [model.get_tensor(elem["index"]) for elem in model.get_output_details()][0]

        for idx, detection in enumerate(detections):
            updated_detections.append(detection._replace(embedding=embs[idx]))
    return updated_detections


def recognize_faces(detections, gallery, thresh=0.67):
    if len(gallery) == 0 or len(detections) == 0:
        return detections

    gallery_embs = np.asarray([identity.embedding for identity in gallery])
    detection_embs = np.asarray([detection.embedding for detection in detections])

    cos_distances = cosine_distances(detection_embs, gallery_embs)

    updated_detections = []
    for idx, detection in enumerate(detections):
        idx_min = np.argmin(cos_distances[idx])
        if thresh and cos_distances[idx][idx_min] > thresh:
            dist = cos_distances[idx][idx_min]
            pred = None
        else:
            dist = cos_distances[idx][idx_min]
            pred = idx_min
        updated_detections.append(
            detection._replace(
                name=gallery[pred].name.split(".jpg")[0].split(".png")[0].split(".jpeg")[0]
                if pred is not None
                else None,
                embedding_match=gallery[pred].embedding if pred is not None else None,
                face_match=gallery[pred].image if pred is not None else None,
                distance=dist,
            )
        )

    return updated_detections


def process_gallery(files, face_detection_model, face_recognition_model):
    gallery = []
    for file in files:
        file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
        img = cv2.cvtColor(cv2.imdecode(file_bytes, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

        detections = detect_faces(img, face_detection_model)

        # We accept only one face per image!
        if detections == []:
            continue
        elif len(detections) > 1:
            detections = detections[:1]

        detections = align_faces(img, detections)
        detections = inference(detections, face_recognition_model)

        gallery.append(
            Identity(
                name=file.name,
                embedding=detections[0].embedding,
                image=detections[0].face,
            )
        )

    return gallery


def draw_detections(
    frame,
    detections,
    bbox=True,
    landmarks=True,
    name=True,
):
    shape = np.asarray(frame.shape[:2][::-1])

    for detection in detections:
        # Draw Landmarks
        if landmarks:
            for landmark in detection.landmarks:
                cv2.circle(
                    frame,
                    (np.asarray(landmark) * shape).astype(int),
                    2,
                    (0, 0, 255),
                    -1,
                )

        # Draw Bounding Box
        if bbox:
            cv2.rectangle(
                frame,
                (np.asarray(detection.bbox[:2]) * shape).astype(int),
                (np.asarray(detection.bbox[2:]) * shape).astype(int),
                (0, 255, 0),
                2,
            )

        # Draw Name
        if name:
            cv2.rectangle(
                frame,
                (
                    int(detection.bbox[0] * shape[0]),
                    int(detection.bbox[1] * shape[1] - (shape[1] // 25)),
                ),
                (int(detection.bbox[2] * shape[0]), int(detection.bbox[1] * shape[1])),
                (255, 255, 255),
                -1,
            )

            cv2.putText(
                frame,
                detection.name,
                (
                    int(detection.bbox[0] * shape[0] + shape[0] // 400),
                    int(detection.bbox[1] * shape[1] - shape[1] // 100),
                ),
                cv2.LINE_AA,
                0.5,
                (0, 0, 0),
                2,
            )

    return frame
