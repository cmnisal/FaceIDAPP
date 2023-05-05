import numpy as np
import tflite_runtime.interpreter as tflite
from sklearn.metrics.pairwise import cosine_distances
import cv2


MODEL_PATHS = {
    "MobileNet": "./models/mobileNet.tflite",
    "ResNet": "./models/resNet.tflite",
}


class FaceIdentity:
    def __init__(self, model="MobileNet"):
        self.model = tflite.Interpreter(model_path=MODEL_PATHS[model])

    def extract(self, imgs, scale):
        if len(imgs) > 0:
            resized_imgs = []
            for img in imgs:
                resized_shape = (int(img.shape[1] * scale), int(img.shape[0] * scale))
                tmp = cv2.resize(img, resized_shape, interpolation=cv2.INTER_NEAREST)
                resized_imgs.append(cv2.resize(tmp, (112, 112), interpolation=cv2.INTER_CUBIC))

            resized_imgs = np.asarray(resized_imgs).astype(np.float32) / 255
            self.model.resize_tensor_input(
                self.model.get_input_details()[0]["index"], resized_imgs.shape
            )
            self.model.allocate_tensors()
            self.model.set_tensor(self.model.get_input_details()[0]["index"], resized_imgs)
            self.model.invoke()
            embs = [
                self.model.get_tensor(elem["index"])
                for elem in self.model.get_output_details()
            ]
            return embs[0]
        else:
            return []


def identify(emb_src, embs_gal, thresh=None):
    """
    TODO
    :param emb_src:
    :param embs_gal:
    :param thresh:
    :return:
    """
    if len(embs_gal) == 0:
        return None, None

    dists = cosine_distances(emb_src, embs_gal)[0]
    pred = np.argmin(dists)
    if (
        thresh and dists[pred] > thresh
    ):  # if OpenSet set prediction to None if above threshold
        idx = np.argsort(dists)
        dist = dists[idx[0]]
        pred = None
    else:
        idx = np.argsort(dists)
        dist = dists[pred]
    return pred, dist


def recognize(probe_embeddings, gallery_embeddings, gallery_labels, gallery_images, thresh):
    identities, dists, gal_imgs = [], [], []
    for i in range(len(probe_embeddings)):
        pred, dist = identify(
            np.expand_dims(probe_embeddings[i], axis=0), gallery_embeddings, thresh=thresh
        )
        identities.append(gallery_labels[pred] if pred is not None else "Unknown")
        dists.append(dist)
        gal_imgs.append(gallery_images[pred] if pred is not None else None)
    return identities, dists, gal_imgs
