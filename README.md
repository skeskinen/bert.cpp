# bert.cpp

[ggml](https://github.com/ggerganov/ggml) inference of BERT neural net architecture with pooling and normalization from [SentenceTransformers (sbert.net)](https://sbert.net/).
High quality sentence embeddings in pure C++ (or C).

## Description
The main goal of `bert.cpp` is to run the BERT model using 4-bit integer quantization on CPU

* Plain C/C++ implementation without dependencies
* Inherit support for various architectures from ggml (x86 with AVX2, ARM, etc.)
* Choose your model size from 32/16/4 bits per model weigth
* all-MiniLM-L6-v2 with 4bit quantization is only 14MB. Inference RAM usage depends on the length of the input
* Sample cpp server over tcp socket and a python test client
* Benchmarks to validate correctness and speed of inference

## Limitations & TODO
* Tokenizer doesn't correctly handle asian writing (CJK, maybe others)
* Inputs longer than ctx size are not truncated. If you are trying to make embeddings for longer texts make sure to truncate.
* bert.cpp doesn't respect tokenizer, pooling or normalization settings from the model card:
    * All inputs are lowercased and trimmed
    * All outputs are mean pooled and normalized
* The API is in C++ (uses things from std::)

## Usage

### Build
```sh
mkdir build
cd build
cmake ..
make
cd ..
```
### Download models
```sh
pip3 install -r requirements.txt
# python3 models/download-ggml.py list_models
python3 models/download-ggml.py download all-MiniLM-L6-v2 q4_0
```
### Start sample server
```sh
./build/bin/server -m models/all-MiniLM-L6-v2/ggml-model-q4_0.bin --port 8085

# bert_model_load: loading model from 'models/all-MiniLM-L6-v2/ggml-model-q4_0.bin' - please wait ...
# bert_model_load: n_vocab = 30522
# bert_model_load: n_ctx   = 512
# bert_model_load: n_embd  = 384
# bert_model_load: n_intermediate  = 1536
# bert_model_load: n_head  = 12
# bert_model_load: n_layer = 6
# bert_model_load: f16     = 2
# bert_model_load: ggml ctx size =  13.57 MB
# bert_model_load: ............ done
# bert_model_load: model size =    13.55 MB / num tensors = 101
# Server running on port 8085 with 4 threads
# Waiting for a client
```
### Run sample client
```sh
python3 examples/sample_client.py 8085
# Loading texts from sample_client_texts.txt...
# Loaded 1738 lines.
# Starting with a test query "Should I get health insurance?"
# Closest texts:
# 1. Will my Medicare premiums be higher because of my higher income?
#  (similarity score: 0.4844)
# 2. Can I sign up for Medicare Part B if I am working and have health insurance through an employer?
#  (similarity score: 0.4575)
# 3. Should I sign up for Medicare Part B if I have Veterans' Benefits?
#  (similarity score: 0.4052)
# Enter a text to find similar texts (enter 'q' to quit): expensive
# Closest texts:
# 1. It is priced at $ 5,995 for an unlimited number of users tapping into the single processor , or $ 195 per user with a minimum of five users .
#  (similarity score: 0.4597)
# 2. The new system costs between $ 1.1 million and $ 22 million , depending on configuration .
#  (similarity score: 0.4547)
# 3. Each hull will cost about $ 1.4 billion , with each fully outfitted submarine costing about $ 2.2 billion , Young said .
#  (similarity score: 0.4078)
```

## Benchmarks
Running MTEB (Massive Text Embedding Benchmark) with bert.cpp vs. [sbert](https://sbert.net/)(cpu mode) gives comparable results between the two, with quantization having minimal effect on accuracy and eval time being similar or better than sbert with batch_size=1 (bert.cpp doesn't support batching).

See [benchmarks](benchmarks) more info.
### all-MiniLM-L6-v2
| Data Type | STSBenchmark | eval time | EmotionClassification | eval time | 
|-----------|-----------|------------|-----------|------------|
| f16 | 0.8201 | 7.52 | 0.4085 | 12.25 | 
| f32 | 0.8201 | 8.22 | 0.4082 | 13.65 | 
| q4_0 | 0.8175 | 6.87 | 0.3911 | 11.22 | 
| q4_1 | 0.8214 | 13.26 | 0.4015 | 21.37 | 
| sbert | 0.8203 | 2.85 | 0.4085 | 7.28 | 
| sbert-batchless | 0.8203 | 12.48 | 0.4085 | 15.27 | 


### all-MiniLM-L12-v2
| Data Type | STSBenchmark | eval time | EmotionClassification | eval time | 
|-----------|-----------|------------|-----------|------------|
| f16 | 0.8306 | 14.66 | 0.4119 | 23.20 | 
| f32 | 0.8306 | 16.18 | 0.4117 | 25.79 | 
| q4_0 | 0.8310 | 13.31 | 0.4183 | 21.54 | 
| q4_1 | 0.8202 | 25.48 | 0.4010 | 41.75 | 
| sbert | 0.8309 | 4.98 | 0.4117 | 10.45 | 
| sbert-batchless | 0.8309 | 22.22 | 0.4117 | 26.53 | 

### bert-base-uncased
bert-base-uncased is not a very good sentence embeddings model, but it's here to show that bert.cpp correctly runs models that are not from SentenceTransformers. Technically any hf model with architecture `BertModel` or `BertForMaskedLM` should work.
| Data Type | STSBenchmark | eval time | EmotionClassification | eval time | 
|-----------|-----------|------------|-----------|------------|
| f16 | 0.4739 | 37.68 | 0.3361 | 61.54 | 
| f32 | 0.4738 | 57.90 | 0.3361 | 91.37 | 
| q4_0 | 0.4940 | 39.21 | 0.3375 | 65.11 | 
| q4_1 | 0.4681 | 85.11 | 0.3268 | 144.11 | 
| sbert | 0.4729 | 16.71 | 0.3527 | 30.03 | 
| sbert-batchless | 0.4729 | 67.12 | 0.3526 | 77.83 | 

