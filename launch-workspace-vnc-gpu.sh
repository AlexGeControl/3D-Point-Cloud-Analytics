#!/bin/bash
docker run \
  --gpus all \
  -v ${PWD}/workspace/assignments:/workspace/assignments \
  -v ${PWD}/workspace/data:/workspace/data \
  -p 49001:9001 \
  -p 45901:5900 \
  -p 46006:6006 \
  --name point_cloud_analytics_workspace anaconda/point-cloud-analysis:bionic-gpu-current