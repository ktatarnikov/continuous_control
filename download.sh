#!/usr/bin/env bash

set -e

wget --show-progress https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip -O ./reacher20/Reacher_Linux_20.zip
wget --show-progress https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip -O ./reacher/Reacher_Linux.zip
wget --show-progress https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux_NoVis.zip -O ./reacher/Reacher_Linux_NoVis.zip
unzip ./reacher20/Reacher_Linux_20.zip -d ./reacher20
unzip ./reacher/Reacher_Linux.zip -d ./reacher
unzip ./reacher/Reacher_Linux_NoVis.zip -d ./reacher
