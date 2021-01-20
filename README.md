# Rotated vehicle orientation
This repository hosts all the files and to train and deploy rotated vehicle detection with detectron2

## Training
The model was trained using the DOTA dataset:
https://captain-whu.github.io/DOTA/dataset.html


## Deploying
In order to deploy the trained model in docker, we first need to archive the trained model into a torchserve compatible format.

You can download the trained models from here:
https://drive.google.com/drive/folders/1LdFZ7QY-0Cxo_bjKkHIdnZKxQ078FXxA?usp=sharing

Run the following commands:

```
$ cd serve
$ pip install torch-model-archiver
$ torch-model-archiver --model-name vehicle_orientation --version 0.1 --serialized-file {path to model_final.pth file} --handler vehicle_handler.py --extra-files config.yaml,vehicle_handler.py --export-path model_store -f
```

After the model is archived, we can run the docker-compose with `docker-compose up`


To see if it is working test with the following command:

```
$ curl http://127.0.0.1:3002/predictions/vehicle_orientation -T test_vehicles.png
```

