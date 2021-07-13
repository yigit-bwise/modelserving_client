# Model deployment



## Files & Scripts

- *Proto* folder: Under this folder the files for gRPC can be found.

  - *torchserve_grpc_client.py*: In this scrip the function used for inferencing can be found under the name *infer*.

- *Resources* folder:

  - *hardhat* folder: Under this folder the model weights and classes information can be found.

  - *yolov3* folder: Under this folder the scripts necessary for initializing yolov3 can be found.

  - *basehandler.py*: This script is the base default handler for the serving process: initializes the model, and handles inferences

  - *torchserve_handler.py*: This script handles the serving process for our yolov3 model: defines the preprocess and postprocess for the inferences.

  - *util.py*: This script matches classes with labels to get a correctly formatted response.

    

## Build and run the docker image

0. Ports 3000:3003 will need to be opened for the EC2 when initializing.

1. Build the torchserve image from cf github torchserve: https://github.com/pytorch/serve/tree/master/docker
2. Build and run our docker image using:

`docker build -t "docker_image_name" .`

`docker run -d -p 3000:8080 -p 3001:8081 -p 3002:7070 -p 3003:7071 "docker_image_name"`

Note: For using gRPC protocol ports 7070:7071 need to be open.



## Getting predictions

Once the docker image is running, you can send gRPC using: 
`python proto/torchserve_grpc_client.py infer hardhats_detector "path_to_image" "your_ec2_url" `

This part can be run from the local machine. The output should print a list with the coordinates of the detected objects and it's label. It should also display the inference image with the bounding boxes of the detections. 

Note: The name used in the gRPC call, *hardhats_detector*, is the model name and it's defined in the docker file.

