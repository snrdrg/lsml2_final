# FMNIST image recognition bot in tg

## Project Overview

ViT transformer for classification of fashion images

## Directory Structure

To setup environment you will need to use command "docker compose up"

Solution contains 3 containers:
* lsml2_training - stand-alone container for training model. For experiment tracking COMET ML was used, you will need to add you COMET KEY and COMET WORKSPACE into ./training/config.json before building. Avaliable as localhost:9999
* lsml2_frontend - TG bot, used as frontend. Before building you will need to add API key into ./frontend/api_key.json
* lsml2_backend - model serving, propose REST API endpoint backend:8088/model. Get json {'question': <base64 encoded image>}, return json {'result': prediction}


