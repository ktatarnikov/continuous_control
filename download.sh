#!/usr/bin/env bash

set -e

wget --show-progress https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip -o Reacher_Linux_20
wget --show-progress https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip -o Reacher_Linux
unzip Reacher_Linux_20.zip -d reacher20
unzip Reacher_Linux.zip -d reacher

# https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux_NoVis.zip
# https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux_NoVis.zip
