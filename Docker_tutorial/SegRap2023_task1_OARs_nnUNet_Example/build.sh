#!/usr/bin/env bash
SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

docker build -t segrap2023_oar_segmentationcontainer "$SCRIPTPATH"
