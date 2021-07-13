import grpc
import inference_pb2
import inference_pb2_grpc
import management_pb2
import management_pb2_grpc
import cv2
import base64
import numpy as np
import sys


def get_inference_stub(ec2_url):
    channel = grpc.insecure_channel(ec2_url + ":3002")
    stub = inference_pb2_grpc.InferenceAPIsServiceStub(channel)
    return stub


def get_management_stub(ec2_url):
    channel = grpc.insecure_channel(ec2_url + ":3003")
    stub = management_pb2_grpc.ManagementAPIsServiceStub(channel)
    return stub


def infer(stub, model_name, model_input):
    detections = {}
    detections["detections"] = []

    jpg_original = base64.b64decode(model_input)
    jpg_as_np = np.frombuffer(jpg_original, dtype=np.uint8)
    frame = cv2.imdecode(jpg_as_np, flags=cv2.IMREAD_COLOR)
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    data = np.array(cv2.imencode(".png", frameRGB)[1]).tobytes()

    input_data = {"body": data}
    response = stub.Predictions(
        inference_pb2.PredictionsRequest(model_name=model_name, input=input_data)
    )

    try:
        prediction = response.prediction.decode("utf-8")
        print("prediction {}", prediction)
        my_list = prediction.split("[\n  ")
        for i, bb in enumerate(my_list):
            my_det = {}
            if bb and bb != "[]":
                bb = bb.strip("").split(",\n")
                bound_b = [
                    int(float(bb[0])),
                    int(float(bb[1])),
                    int(float(bb[2])),
                    int(float(bb[3])),
                ]
                my_det["label"] = bb[5].split('"')[1]
                my_det["location"] = bound_b
                my_det["probability"] = float(bb[4])
                detections["detections"].append(my_det)
    except grpc.RpcError as e:
        exit(1)

    print("detections {}", detections)
    return detections


def infer_test(stub, model_name, model_input):

    detections = {}
    detections["detections"] = []
    img = cv2.imread(model_input)

    if isinstance(model_input, (bytes, bytearray)):
        jpg_original = base64.b64decode(model_input)
        jpg_as_np = np.frombuffer(jpg_original, dtype=np.uint8)
        frame = cv2.imdecode(jpg_as_np, flags=cv2.IMREAD_COLOR)
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        data = np.array(cv2.imencode(".png", frameRGB)[1]).tobytes()

    else:
        try:
            if model_input.endswith(".png"):
                data = cv2.imencode(".png", img)[1].tobytes()
            else:
                with open(model_input, "rb") as f:
                    data = f.read()
        except grpc.RpcError as e:
            exit(1)

    input_data = {"body": data}
    response = stub.Predictions(
        inference_pb2.PredictionsRequest(model_name=model_name, input=input_data)
    )

    try:
        prediction = response.prediction.decode("utf-8")
        # print(prediction)
        my_list = prediction.split("[\n  ")
        for i, bb in enumerate(my_list):
            my_det = {}
            if bb and bb != "[]":
                bb = bb.strip("").split(",\n")
                bound_b = (
                    (int(float(bb[0])), int(float(bb[1]))),
                    (int(float(bb[2])), int(float(bb[3]))),
                )
                my_det["label"] = bb[5].split('"')[1]
                my_det["location"] = bound_b
                my_det["probability"] = float(bb[4])
                detections["detections"].append(my_det)
                img = cv2.rectangle(img, bound_b[0], bound_b[1], (0, 255, 0), 2)
                cv2.putText(
                    img,
                    bb[5].split('"')[1] + ": " + "{:.2f}".format(float(bb[4])),
                    (int(float(bb[0])), int(float(bb[1])) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (36, 255, 0),
                    2,
                )
        print(detections)
        cv2.imshow("result", img)
        cv2.waitKey(0)
    except grpc.RpcError as e:
        exit(1)


def register(stub, model_name):
    params = {
        "url": "https://torchserve.s3.amazonaws.com/mar_files/{}.mar".format(
            model_name
        ),
        "initial_workers": 1,
        "synchronous": True,
        "model_name": model_name,
    }
    try:
        response = stub.RegisterModel(management_pb2.RegisterModelRequest(**params))
        print(f"Model {model_name} registered successfully")
    except grpc.RpcError as e:
        print(f"Failed to register model {model_name}.")
        print(str(e.details()))
        exit(1)


def unregister(stub, model_name):
    try:
        response = stub.UnregisterModel(
            management_pb2.UnregisterModelRequest(model_name=model_name)
        )
        print(f"Model {model_name} unregistered successfully")
    except grpc.RpcError as e:
        print(f"Failed to unregister model {model_name}.")
        print(str(e.details()))
        exit(1)


if __name__ == "__main__":
    # args:
    # 1-> api name [infer, register, unregister]
    # 2-> model name
    # 3-> model input for prediction
    args = sys.argv[1:]
    if args[0] == "infer":
        infer(get_inference_stub(args[3]), args[1], args[2])
    elif args[0] == "infer_test":
        infer_test(get_inference_stub(args[3]), args[1], args[2])
    else:
        api = globals()[args[0]]
        api(get_management_stub(args[2]), args[1])
