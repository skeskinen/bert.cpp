#!/bin/bash

model=$1

python3 convert-to-ggml.py ${model}/ 0
python3 convert-to-ggml.py ${model}/ 1
../build/bin/quantize ${model}/ggml-model-f16.bin ${model}/ggml-model-q4_0.bin 2
../build/bin/quantize ${model}/ggml-model-f16.bin ${model}/ggml-model-q4_1.bin 3