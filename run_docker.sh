#! /bin/bash

docker run --gpus=all --name sanno-simple-cnaps-$(date '+%s') --rm -u "$(id -u):$(id -g)" -e "PYTHONPATH=." -v "$HOME:$HOME" -w "$(pwd)" --shm-size 1gb sanno-simple-cnaps "$@"
