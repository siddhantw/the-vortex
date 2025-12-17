#!/bin/bash

# Starting Xvfb - this can't be put into Docker since it doesn't have a service
# Refer : https://stackoverflow.com/questions/32151043/xvfb-docker-cannot-open-display
echo "Starting Xvfb..."
Xvfb :99 -ac -screen 0 1920x1080x24 &

command=$1
shift # short for shift 1
echo "Executing this command : sh $command" "$@"
sh "$command" "$@"
