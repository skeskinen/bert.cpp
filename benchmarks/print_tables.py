import os
import json

RESULTS_DIR = "results"
BENCHMARKS = ["STSBenchmark", "EmotionClassification"]
DATA_TYPES = [ "f32", "f16", "q4_0", "q4_1", "sbert", "sbert-batchless"]

# Define a dictionary to store the results
results_dict = {}

# Loop over all the directories and extract the models
models = set()
for dir_name in os.listdir(RESULTS_DIR):
    m = dir_name.split("_")[0]
    models.add(m)

def extract_results(test_data):
    res = {"time": test_data["evaluation_time"]}
    if "cos_sim" in test_data and "spearman" in test_data["cos_sim"]:
        res['score'] = test_data["cos_sim"]["spearman"]
    elif "main_score" in test_data:
        res['score'] = test_data["main_score"]
    else:
        print(f"can't extract results {test_data}")
    return res

for model in models:
    model_results = {}
    for data_type in DATA_TYPES:
        dir_name = f"{RESULTS_DIR}/{model}_{data_type}"
        if not os.path.isdir(dir_name):
            print(f"{dir_name} doesn't exist!")
            continue
        data_type_results = {}
        for benchmark in BENCHMARKS:
            results_path = os.path.join(dir_name, f"{benchmark}.json")
            with open(results_path, "r") as f:
                results = json.load(f)

            data_type_results[benchmark] = extract_results(results['test'])

        model_results[data_type] = data_type_results
    results_dict[model] = model_results

# Print the results as an .md table for each model
for model, model_results in results_dict.items():
    print(f"### {model}")
    print("| Data Type | ", end="")
    for benchmark in BENCHMARKS:
        print(f"{benchmark} | eval time | ", end="")
    print()
    print("|-----------|", end="")
    for _ in BENCHMARKS:
        print("-----------|------------|", end="")
    print()
    for data_type in DATA_TYPES:
        print(f"| {data_type} | ", end="")
        for benchmark in BENCHMARKS:
            results = model_results[data_type][benchmark]
            print(f"{results['score']:.4f} | {results['time']:.2f} | ", end="")
        print()
    print("\n")
