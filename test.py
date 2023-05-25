import torch
from tools.utils import get_file

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

filename = get_file(URLS["FaceTransformerOctupletLoss"], FILE_HASHES["FaceTransformerOctupletLoss"])
model = torch.load(filename)

torch.save(model.state_dict(), "model-weights.pt")
# from tools.vit_face import ViT_face

# model = ViT_face(
#     loss_type = "CosFace",
#     GPU_ID = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
#     num_class = 93431,
#     image_size=112,
#     patch_size=8,
#     dim=512,
#     depth=20,
#     heads=8,
#     mlp_dim=2048,
#     dropout=0.1,
#     emb_dropout=0.1
# )
# model.load_state_dict(torch.load("model-weights.pt"))
# model.eval()

# print(1)

# TODO make tf lite model from ArcFaceOctupletLoss and PTLite Model from FaceTransformerOctupletLoss!