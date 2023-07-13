#!/usr/bin/env bash

./build.sh

docker save segrap2023_segmentationcontainer | gzip -c > SegRap2023_SegmentationContainer.tar.gz
