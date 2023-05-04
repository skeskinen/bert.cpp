Use `run_mteb.py` to run mteb embeddings benchmark for each model. The script will start the c++ server for each different size of model, so make sure you have all 4 sizes in your models directory. It will also run the benchmarks with SentenceTransformers library to get a baseline results.

The ggml version doesn't have batching so it is at a disadvantage compared to sbert where all the computations are done in batches of 64 input sentences. But if batching is not possible in your application (e.g. the inputs are given by the user) then the batchless performance is more relevant. sbert-batchless runs the benchmark with SentenceTransformers library with `batch_size=1`

Note that the sbert results here are with CPU. Sbert also supports GPU inference, and in that case it would be much faster.

Use `print_tables.py` to format the results like the following tables.

### all-MiniLM-L6-v2
| Data Type | STSBenchmark | eval time | EmotionClassification | eval time | 
|-----------|-----------|------------|-----------|------------|
| f32 | 0.8201 | 6.83 | 0.4082 | 11.34 | 
| f16 | 0.8201 | 6.17 | 0.4085 | 10.28 | 
| q4_0 | 0.8175 | 5.45 | 0.3911 | 10.63 | 
| q4_1 | 0.8223 | 6.79 | 0.4027 | 11.41 | 
| sbert | 0.8203 | 2.74 | 0.4085 | 5.56 | 
| sbert-batchless | 0.8203 | 13.10 | 0.4085 | 15.52 | 

### all-MiniLM-L12-v2
| Data Type | STSBenchmark | eval time | EmotionClassification | eval time | 
|-----------|-----------|------------|-----------|------------|
| f32 | 0.8306 | 13.36 | 0.4117 | 21.23 | 
| f16 | 0.8306 | 11.51 | 0.4119 | 20.08 | 
| q4_0 | 0.8310 | 11.27 | 0.4183 | 20.81 | 
| q4_1 | 0.8325 | 12.37 | 0.4093 | 19.38 | 
| sbert | 0.8309 | 5.11 | 0.4117 | 8.93 | 
| sbert-batchless | 0.8309 | 22.81 | 0.4117 | 28.04 | 


### bert-base-uncased
For bert-base-uncased, the pooling and normalization are different from the ones used in the actual model. I think that's why ggml scores better than sbert in STSBenchmark and worse in EmotionClassification

| Data Type | STSBenchmark | eval time | EmotionClassification | eval time | 
|-----------|-----------|------------|-----------|------------|
| f32 | 0.4738 | 52.38 | 0.3361 | 88.56 | 
| f16 | 0.4739 | 33.24 | 0.3361 | 55.86 | 
| q4_0 | 0.4940 | 33.93 | 0.3375 | 57.82 | 
| q4_1 | 0.4612 | 36.86 | 0.3318 | 59.63 | 
| sbert | 0.4729 | 16.97 | 0.3527 | 28.77 | 
| sbert-batchless | 0.4729 | 69.97 | 0.3526 | 79.02 | 
