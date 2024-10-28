#!/bin/bash

pip install matplotlib tensorflow-federated==0.86.0 ipykernel
# The line below is necessary because of a conflict between tff and jupyter.
# tff forces an old version, nevertheless, works fine with a more recent one.
pip install typing-extensions==4.12.0 tensorflow[and-cuda]
