import numpy as np
import tensorflow as tf
import torch
from .utils import get_file
from .vit_face import ViT_face
import onnxruntime as rt


# TODO merge into single dict
# TODO make progress bars in Streamlit visible
URLS = {
    "o_net": "https://github.com/Martlgap/FaceIDLight/releases/download/v.0.1/o_net.tflite",
    "p_net": "https://github.com/Martlgap/FaceIDLight/releases/download/v.0.1/p_net.tflite",
    "r_net": "https://github.com/Martlgap/FaceIDLight/releases/download/v.0.1/r_net.tflite",
    "MobileNetV2": "https://github.com/Martlgap/FaceIDLight/releases/download/v.0.1/mobileNet.tflite",
    "ResNet50": "https://github.com/Martlgap/FaceIDLight/releases/download/v.0.1/resNet.tflite",
    "FaceTransformerOctupletLossONNX": "https://github.com/Martlgap/FaceIDLight/releases/download/v.0.1/FaceTransformerOctupletLoss.onnx",
    "MobileNetV2ONNX": "https://github.com/Martlgap/FaceIDLight/releases/download/v.0.1/MobileNetV2.onnx",
    "FaceTransformerOctupletLossPT": "https://github.com/Martlgap/FaceIDLight/releases/download/v.0.1/FaceTransformerOctupletLoss.pt",
    "ArcFaceOctupletLossTF": "https://github.com/Martlgap/octuplet-loss/releases/download/modelweights/ArcFaceOctupletLoss.tf.zip",
}

FILE_HASHES = {
    "o_net": "768385d570300648b7b881acbd418146522b79b4771029bb2e684bdd8c764b9f",
    "p_net": "530183192e24f7cc86b6706e1eb600482c4ed4306399ac939c472e3957bae15e",
    "r_net": "5ec33b065eb2802bc4c2575d21feff1a56958d854785bc3e2907d3b7ace861a2",
    "MobileNetV2": "6c19b789f661caa8da735566490bfd8895beffb2a1ec97a56b126f0539991aa6",
    "ResNet50": "f4d8b0194957a3ad766135505fc70a91343660151a8103bbb6c3b8ac34dbb4e2",
    "FaceTransformerOctupletLossPT": "b10faa1c170b9fd0f95e3142d9e584ad6f9647d3566207d8bfcc259df8dbdf0f",
    "ArcFaceOctupletLossTF": "8603f374fd385081ce5ce80f5997e3363f4247c8bbad0b8de7fb26a80468eeea",
    "FaceTransformerOctupletLossONNX": "aa995cce8b137ccdc65b394cc57c6b1fdafc7012ce5197e62a4cf8d8e61db4f2",
    "MobileNetV2ONNX": "6f53fb10f0db558403f73cfe744a96b12d763bdf1294a38d14ef14307d61ecf3",
}


class TFModel:
    def _inference(self, img):
        return self.model.predict(img)


class ArcFaceOctupletLoss(TFModel):
    def __init__(self, batch_size=32):
        self.model = tf.keras.models.load_model(
            get_file(URLS["ArcFaceOctupletLossTF"], FILE_HASHES["ArcFaceOctupletLossTF"])
        )
        self.batch_size = batch_size

    @staticmethod
    def __preprocess(img):
        if img.ndim != 4:
            img = np.expand_dims(img, axis=0)
        return img

    def __call__(self, imgs):
        embs = []
        for i in range(0, imgs.shape[0], self.batch_size):
            embs.append(self._inference(self.__preprocess(imgs[i : i + self.batch_size])))
        return np.concatenate(embs)


class PTModel:
    def _inference(self, img) -> np.ndarray:
        if self.device.type == "cuda":
            img = img.cuda()
        return self.model(img).cpu().detach().numpy()


class FaceTransformerOctupletLoss(PTModel):
    def __init__(self, batch_size=32) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.model = ViT_face(
            loss_type="CosFace",
            GPU_ID=self.device,
            num_class=93431,
            image_size=112,
            patch_size=8,
            dim=512,
            depth=20,
            heads=8,
            mlp_dim=2048,
            dropout=0.1,
            emb_dropout=0.1,
        )
        self.model.load_state_dict(
            torch.load(
                get_file(URLS["FaceTransformerOctupletLossPT"], FILE_HASHES["FaceTransformerOctupletLossPT"]),
                map_location=self.device,
            )
        )
        self.model.eval()
        self.batch_size = batch_size

    def __preprocess(self, img) -> np.ndarray:
        if img.ndim != 4:
            img = np.expand_dims(img, axis=0)
        img = (
            torch.from_numpy(np.transpose(img, [0, 3, 1, 2]).astype("float32") * 255).clamp(0.0, 255.0).to(self.device)
        )
        return img

    def __call__(self, imgs):
        embs = []
        for i in range(0, imgs.shape[0], self.batch_size):
            embs.append(self._inference(self.__preprocess(imgs[i : i + self.batch_size])))
        return np.concatenate(embs)


class TFLiteModel:
    @staticmethod
    def _inference(model, img):
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

        input_details = model.get_input_details()
        output_details = model.get_output_details()
        model.resize_tensor_input(input_details[0]["index"], img.shape)
        model.allocate_tensors()
        model.set_tensor(input_details[0]["index"], img.astype(np.float32))
        model.invoke()
        return [model.get_tensor(elem["index"]) for elem in output_details]


class MobileNetV2(TFLiteModel):
    def __init__(self):
        self.model = tf.lite.Interpreter(model_path=get_file(URLS["MobileNetV2"], FILE_HASHES["MobileNetV2"]))

    def __call__(self, imgs):
        return self._inference(self.model, imgs)[0]


class ResNet50(TFLiteModel):
    def __init__(self):
        self.model = tf.lite.Interpreter(model_path=get_file(URLS["ResNet50"], FILE_HASHES["ResNet50"]))

    def __call__(self, imgs):
        return self._inference(self.model, imgs)[0]


class MTCNN(TFLiteModel):
    def __init__(self) -> None:
        self.p_net_model = tf.lite.Interpreter(model_path=get_file(URLS["p_net"], FILE_HASHES["p_net"]))
        self.r_net_model = tf.lite.Interpreter(model_path=get_file(URLS["r_net"], FILE_HASHES["r_net"]))
        self.o_net_model = tf.lite.Interpreter(model_path=get_file(URLS["o_net"], FILE_HASHES["o_net"]))

    def p_net(self, inp):
        return self._inference(self.p_net_model, inp)

    def r_net(self, inp):
        return self._inference(self.r_net_model, inp)

    def o_net(self, inp):
        return self._inference(self.o_net_model, inp)


class ONNXModel:
    @staticmethod
    def _inference(sess, imgs):
        return sess.run(None, {"input_image": imgs.astype(np.float32)})[0]


class MobileNetV2ONNX(ONNXModel):
    def __init__(self) -> None:
        self.sess = rt.InferenceSession(get_file(URLS["MobileNetV2ONNX"], FILE_HASHES["MobileNetV2ONNX"]), providers=rt.get_available_providers())

    # TODO somehow show if CPU or GPU is used?
    def __call__(self, imgs):
        return self._inference(self.sess, imgs)


class FaceTransformerOctupletLossONNX(ONNXModel):
    def __init__(self) -> None:
        self.sess = rt.InferenceSession(get_file(URLS["FaceTransformerOctupletLossONNX"], FILE_HASHES["FaceTransformerOctupletLossONNX"]), providers=rt.get_available_providers())

    def __call__(self, imgs):
        imgs = (np.transpose(imgs, [0, 3, 1, 2]) * 255.0).clip(0.0, 255.0)
        return self._inference(self.sess, imgs)
