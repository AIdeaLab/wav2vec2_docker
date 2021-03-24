#!/bin/sh
mkdir -p ./outputs
while true
do
  mv ./outputs/*/*/checkpoints/checkpoint*.pt /opt/ml/checkpoints/
  sleep 60
done
