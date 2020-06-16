#!/bin/bash
docker run \
  --gpus all \
  --dns 8.8.8.8 \
  --dns 8.8.4.4 \
  --dns 180.76.76.76 \
  --dns 114.114.114.114 \
  --dns 192.168.3.1 \
  -v ${PWD}/workspace/assignments:/workspace/assignments \
  -v ${PWD}/workspace/data:/workspace/data \
  -p 49001:9001 \
  -p 45901:5900 \
  -p 46006:6006 \
  --name point_cloud_analytics_workspace shenlanxueyuan/point-cloud-processing:ubuntu-bionic-gpu