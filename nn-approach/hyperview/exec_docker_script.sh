#!/bin/bash

CONTAINER_NAME="efficiency_comparison"
IMAGE_NAME="nvcr.io/nvidia/driver:570-5.15.0-1074-nvidia-ubuntu22.04"
WORKDIR="/workspace"
SYMLINK_PATH="$(readlink -f data)"

# Controlla se il container non esiste
if [ ! "$(docker ps -aq -f name=$CONTAINER_NAME)" ]; then
    echo "Creando un nuovo container chiamato $CONTAINER_NAME..."
    docker run -d --gpus all --name $CONTAINER_NAME \
        -v $(pwd):$WORKDIR \
        -v $SYMLINK_PATH:/data \
        -w $WORKDIR \
        $IMAGE_NAME

    # # Esegui l'installazione di Python e delle dipendenze solo se il container è appena stato creato
    # echo "Installazione di Python e dipendenze nel container..."
    # docker exec -it $CONTAINER_NAME bash -c "
    #     apt update &&
    #     apt install -y python3 python3-pip &&
    #     pip3 install -r $WORKDIR/requirements.txt
    # "
fi
