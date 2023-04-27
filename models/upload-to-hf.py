
from transformers import LlamaConfig, LlamaForCausalLM, LlamaTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling

from huggingface_hub import HfApi
import os
import sys

os.chdir(os.path.dirname(__file__))
api = HfApi()

if len(sys.argv) > 1:
    model = sys.argv[1].strip('/')
    sizes = ['f32', 'f16', 'q4_0', 'q4_1']

    for s in sizes:
        api.upload_file(
            path_or_fileobj=f"./{model}/ggml-model-{s}.bin",
            path_in_repo=f"{model}/ggml-model-{s}.bin",
            repo_id="skeskinen/ggml",
            repo_type="model",
    )
else:
    print("give model name")