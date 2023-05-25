import numpy as np
import tensorflow as tf
import torch
from .utils import get_file

URLS = {
    "mobileNet": "https://github.com/Martlgap/FaceIDLight/releases/download/v.0.1/mobileNet.tflite",
    "resNet": "https://github.com/Martlgap/FaceIDLight/releases/download/v.0.1/resNet.tflite",
    "FaceTransformerOctupletLoss": "https://github.com/Martlgap/octuplet-loss/releases/download/modelweights/FaceTransformerOctupletLoss.pt",
    "ArcFaceOctupletLoss": "https://github.com/Martlgap/octuplet-loss/releases/download/modelweights/ArcFaceOctupletLoss.tf.zip",
}

FILE_HASHES = {
    "mobileNet": "6c19b789f661caa8da735566490bfd8895beffb2a1ec97a56b126f0539991aa6",
    "resNet": "f4d8b0194957a3ad766135505fc70a91343660151a8103bbb6c3b8ac34dbb4e2",
    "FaceTransformerOctupletLoss": "f2c7cf1b074ecb17e546dc7043a835ad6944a56045c9675e8e1158817d662662",
    "ArcFaceOctupletLoss": "8603f374fd385081ce5ce80f5997e3363f4247c8bbad0b8de7fb26a80468eeea",
}

class TFModel:
    @staticmethod
    def __preprocess(img):
        if img.ndim != 4:
            img = np.expand_dims(img, axis=0)
        return img

    def _inference(self, img):
        return self.model.predict(self.__preprocess(img))


class ArcFaceOctupletLoss(TFModel):
    def __init__(self, batch_size=32):
        self.model = tf.keras.models.load_model(
            get_file(URLS["ArcFaceOctupletLoss"], FILE_HASHES["ArcFaceOctupletLoss"])
        )
        self.batch_size = batch_size

    def __call__(self, imgs):
        embs = []
        for i in range(0, imgs.shape[0], self.batch_size):
            embs.append(self._inference(imgs[i : i + self.batch_size]))
        return np.concatenate(embs)


class PyTorchModel:
    def __preprocess(self, img) -> np.ndarray:
        if img.ndim != 4:
            img = np.expand_dims(img, axis=0)
        img = torch.from_numpy(np.transpose(img, [0, 3, 1, 2]).astype("float32") * 255).clamp(0.0, 255.0).to(self.device)
        return img

    def _inference(self, img) -> np.ndarray:
        return self.model(self.__preprocess(img)).cpu().detach().numpy()


class FaceTransformerOctupletLoss(PyTorchModel):
    def __init__(self, batch_size=32) -> None:
        self.device = torch.device("cuda")  # or cuda
        self.model = torch.load(get_file(URLS["FaceTransformerOctupletLoss"], FILE_HASHES["FaceTransformerOctupletLoss"]), map_location=self.device)
        self.model.eval()
        self.batch_size = batch_size

    def __call__(self, imgs):
        embs = []
        for i in range(0, imgs.shape[0], self.batch_size):
            embs.append(self._inference(imgs[i : i + self.batch_size]))
        return np.concatenate(embs)


class TFLiteModel:
    def _inference(self, img):
        """Inferences an image through the model with tflite interpreter on CPU
        :param model: a tflite.Interpreter loaded with a model
        :param img: image
        :return: list of outputs of the model
        """
        # Check if img is np.ndarray
        if not isinstance(img, np.ndarray):
            img = np.asarray(img)

        # Check if dim is 4
        if len(img.shape) == 3:
            img = np.expand_dims(img, axis=0)

        input_details = self.model.get_input_details()
        output_details = self.model.get_output_details()
        self.model.resize_tensor_input(input_details[0]["index"], img.shape)
        self.model.allocate_tensors()
        self.model.set_tensor(input_details[0]["index"], img.astype(np.float32))
        self.model.invoke()
        return [self.model.get_tensor(elem["index"]) for elem in output_details][0]


class MobileNetV2(TFLiteModel):
    def __init__(self):
        self.model = tf.lite.Interpreter(model_path=get_file(URLS["mobileNet"], FILE_HASHES["mobileNet"]))

    def __call__(self, imgs):
        return self._inference(imgs)


class ResNet50(TFLiteModel):
    def __init__(self):
        self.model = tf.lite.Interpreter(model_path=get_file(URLS["resNet"], FILE_HASHES["resNet"]))

    def __call__(self, imgs):
        return self._inference(imgs)
