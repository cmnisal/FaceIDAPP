import numpy as np
import cv2


class Annotation:
    def __init__(self, draw_bbox=True, draw_landmarks=True, draw_name=True, upscale=True):
        self.bbox = draw_bbox
        self.landmarks = draw_landmarks
        self.name = draw_name
        self.upscale = upscale

    def __call__(self, frame, detections, identities, matches, gallery):
        shape = np.asarray(frame.shape[:2][::-1])
        if self.upscale:
            frame = cv2.resize(frame, (1920, 1080))
            upscale_factor = np.asarray([1920 / shape[0], 1080 / shape[1]])
            shape = np.asarray(frame.shape[:2][::-1])
        else:
            upscale_factor = np.asarray([1, 1])

        frame.flags.writeable = True

        for detection in detections:
            # Draw Landmarks
            if self.landmarks:
                for landmark in detection.landmarks:
                    cv2.circle(
                        frame,
                        (landmark * upscale_factor).astype(int),
                        2,
                        (255, 255, 255),
                        -1,
                    )

            # Draw Bounding Box
            if self.bbox:
                cv2.rectangle(
                    frame,
                    (detection.bbox[0] * upscale_factor).astype(int),
                    (detection.bbox[1] * upscale_factor).astype(int),
                    (255, 0, 0),
                    2,
                )

            # Draw Index
            cv2.putText(
                frame,
                str(detection.idx),
                (
                    ((detection.bbox[1][0] + 2) * upscale_factor[0]).astype(int),
                    ((detection.bbox[1][1] + 2) * upscale_factor[1]).astype(int),
                ),
                cv2.LINE_AA,
                0.5,
                (0, 0, 0),
                2,
            )

        # Draw Name
        if self.name:
            for match in matches:
                try:
                    detection = detections[identities[match.identity_idx].detection_idx]
                except:
                    print("Identity IDX: ", match.identity_idx)
                    print("Len(Detections): ", len(detections))
                    print("Len(Identites): ", len(identities))
                    print("Detection IDX: ", identities[match.identity_idx].detection_idx)

                    # print("Detections: ", detections)

                cv2.rectangle(
                    frame,
                    (detection.bbox[0] * upscale_factor).astype(int),
                    (detection.bbox[1] * upscale_factor).astype(int),
                    (0, 255, 0),
                    2,
                )

                cv2.rectangle(
                    frame,
                    (
                        (detection.bbox[0][0] * upscale_factor[0]).astype(int),
                        (detection.bbox[0][1] * upscale_factor[1] - (shape[1] // 25)).astype(int),
                    ),
                    (
                        (detection.bbox[1][0] * upscale_factor[0]).astype(int),
                        (detection.bbox[0][1] * upscale_factor[1]).astype(int),
                    ),
                    (255, 255, 255),
                    -1,
                )

                cv2.putText(
                    frame,
                    gallery[match.gallery_idx].name,
                    (
                        ((detection.bbox[0][0] + shape[0] // 400) * upscale_factor[0]).astype(int),
                        ((detection.bbox[0][1] - shape[1] // 100) * upscale_factor[1]).astype(int),
                    ),
                    cv2.LINE_AA,
                    0.5,
                    (0, 0, 0),
                    2,
                )

        return frame
