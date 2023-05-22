import numpy as np
import cv2


class Annotation:
    def __init__(self, draw_bbox=True, draw_landmarks=True, draw_name=True):
        self.bbox = draw_bbox
        self.landmarks = draw_landmarks
        self.name = draw_name

    def __call__(self, frame, detections, identities, matches):
        shape = np.asarray(frame.shape[:2][::-1])

        frame.flags.writeable = True

        for detection in detections:
            # Draw Landmarks
            if self.landmarks:
                for landmark in detection.landmarks:
                    cv2.circle(
                        frame,
                        (landmark).astype(int),
                        2,
                        (255, 255, 255),
                        -1,
                    )

            # Draw Bounding Box
            if self.bbox:
                cv2.rectangle(
                    frame,
                    detection.bbox[0].astype(int),
                    detection.bbox[1].astype(int),
                    (255, 0, 0),
                    2,
                )
        
        # Draw Name
        if self.name:
            for match in matches:
                detection = detections[identities[match.identity_idx].detection_idx]

                cv2.rectangle(
                    frame,
                    detection.bbox[0].astype(int),
                    detection.bbox[1].astype(int),
                    (0, 255, 0),
                    2,
                )

                cv2.rectangle(
                    frame,
                    (
                        (detection.bbox[0][0]).astype(int),
                        (detection.bbox[0][1] - (shape[1] // 25)).astype(int),
                    ),
                    (
                        (detection.bbox[1][0]).astype(int),
                        (detection.bbox[0][1]).astype(int),
                    ),
                    (255, 255, 255),
                    -1,
                )

                cv2.putText(
                    frame,
                    match.name,
                    (
                        (detection.bbox[0][0] + shape[0] // 400).astype(int),
                        (detection.bbox[0][1] - shape[1] // 100).astype(int),
                    ),
                    cv2.LINE_AA,
                    0.5,
                    (0, 0, 0),
                    2,
                )

        return frame
