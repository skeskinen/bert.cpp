# bert.cpp

[ggml](https://github.com/ggerganov/ggml) inference of BERT neural net architecture with pooling and normalization from [SentenceTransformers (sbert.net)](https://sbert.net/).
High quality sentence embeddings in pure C++ (with C API).

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
* bert.cpp doesn't respect tokenizer, pooling or normalization settings from the model card:
    * All inputs are lowercased and trimmed
    * All outputs are mean pooled and normalized
* Batching support is WIP. Lack of real batching means that this library is slower than it could be in usecases where you have multiple sentences

## Usage

### Download models
```sh
pip3 install -r requirements.txt
# python3 models/download-ggml.py list_models
python3 models/download-ggml.py download all-MiniLM-L6-v2 q4_0
```
### Build
To build the dynamic library for usage from e.g. Python:
```sh
mkdir build
cd build
cmake .. -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release
make
cd ..
```

To build the native binaries, like the example server, with static libraries, run:
```sh
mkdir build
cd build
cmake .. -DBUILD_SHARED_LIBS=OFF -DCMAKE_BUILD_TYPE=Release
make
cd ..
```
### Run the python dynamic library example
```sh
python3 examples/sample_dylib.py models/all-MiniLM-L6-v2/ggml-model-f16.bin

# bert_load_from_file: loading model from '../models/all-MiniLM-L6-v2/ggml-model-f16.bin' - please wait ...
# bert_load_from_file: n_vocab = 30522
# bert_load_from_file: n_max_tokens   = 512
# bert_load_from_file: n_embd  = 384
# bert_load_from_file: n_intermediate  = 1536
# bert_load_from_file: n_head  = 12
# bert_load_from_file: n_layer = 6
# bert_load_from_file: f16     = 1
# bert_load_from_file: ggml ctx size =  43.12 MB
# bert_load_from_file: ............ done
# bert_load_from_file: model size =    43.10 MB / num tensors = 101
# bert_load_from_file: mem_per_token 450 KB
# Loading texts from sample_client_texts.txt...
# Loaded 1738 lines.
# Starting with a test query "Should I get health insurance?"
# Closest texts:
# 1. Can I sign up for Medicare Part B if I am working and have health insurance through an employer?
#  (similarity score: 0.4790)
# 2. Will my Medicare premiums be higher because of my higher income?
#  (similarity score: 0.4633)
# 3. Should I sign up for Medicare Part B if I have Veterans' Benefits?
#  (similarity score: 0.4208)
# Enter a text to find similar texts (enter 'q' to quit): poaching
# Closest texts:
# 1. The exotic animal trade is enormous , and it continues to spiral out of control .
#  (similarity score: 0.2825)
# 2. " PeopleSoft management entrenchment tactics continue to destroy the value of the company for its shareholders , " said Deborah Lilienthal , an Oracle spokeswoman .
#  (similarity score: 0.2709)
# 3. " I 've stopped looters , run political parties out of abandoned buildings , caught people with large amounts of cash and weapons , " Williams said .
#  (similarity score: 0.2672)
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

### Converting models to ggml format
Converting models is similar to llama.cpp. Use models/convert-to-ggml.py to make hf models into either f32 or f16 ggml models. Then use ./build/bin/quantize to turn those into Q4_0, 4bit per weight models.

There is also models/run_conversions.sh which creates all 4 versions (f32, f16, Q4_0, Q4_1) at once.
```sh
cd models
# Clone a model from hf
git clone https://huggingface.co/sentence-transformers/multi-qa-MiniLM-L6-cos-v1
# Run conversions to 4 ggml formats (f32, f16, Q4_0, Q4_1)
sh run_conversions.sh multi-qa-MiniLM-L6-cos-v1
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

