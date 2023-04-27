Use `run_mteb.py` to run mteb embeddings benchmark for each model. The script will start the c++ server for each different size of model, so make sure you have all 4 sizes in your models directory. It will also run the benchmarks with SentenceTransformers library to get a baseline results.

The ggml version doesn't have batching so it is at a disadvantage compared to sbert where all the computations are done in batches of 64 input sentences. But if batching is not possible in your application (e.g. the inputs are given by the user) then the batchless performance is more relevant. sbert-batchless runs the benchmark with SentenceTransformers library with `batch_size=1`

Note that the sbert results here are with CPU. Sbert also supports GPU inference, and in that case it would be much faster.

Use `print_tables.py` to format the results like the following tables.

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
For bert-base-uncased, the pooling and normalization are different from the ones used in the actual model. I think that's why ggml scores better than sbert in STSBenchmark and worse in EmotionClassification
| Data Type | STSBenchmark | eval time | EmotionClassification | eval time | 
|-----------|-----------|------------|-----------|------------|
| f16 | 0.4739 | 37.68 | 0.3361 | 61.54 | 
| f32 | 0.4738 | 57.90 | 0.3361 | 91.37 | 
| q4_0 | 0.4940 | 39.21 | 0.3375 | 65.11 | 
| q4_1 | 0.4681 | 85.11 | 0.3268 | 144.11 | 
| sbert | 0.4729 | 16.71 | 0.3527 | 30.03 | 
| sbert-batchless | 0.4729 | 67.12 | 0.3526 | 77.83 | 

