from .proto.torchserve_grpc_client import infer, get_inference_stub

MODEL_URI = "13.49.245.51"
MODEL_NAME = "hardhats_detector"


class HardHatDetector:
    def __init__(self):
        self.detection_limit = 0.5

    def get_predictions(self, frame):
        print("getting hat detections...")
        return infer(get_inference_stub(MODEL_URI), MODEL_NAME, frame)
