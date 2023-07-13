#!/usr/bin/env bash

./build.sh

docker save segrap2023_gtv_segmentationcontainer | gzip -c > SegRap2023_GTV_SegmentationContainer.tar.gz
