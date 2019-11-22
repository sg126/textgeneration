#!/usr/bin/env bash

python3 main.py gantext/rnn_cuda.yml
python3 main.py gantext/transformer_cuda.yml
python3 main.py gantext/gpt2_cuda.yml
python3 main.py gantext/bert_cuda.yml
python3 main.py gantext/transformer_xl_cuda.yml
python3 main.py transformer/transformer_cuda.yml
python3 main.py transformer/transformer_xl_cuda.yml
python3 main.py transformer/gpt2_cuda.yml
python3 main.py transformer/bert_cuda.yml