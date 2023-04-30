import socket
import struct
import numpy as np
from mteb import MTEB
import subprocess
import time
from sentence_transformers import SentenceTransformer
import os

MODEL_NAME = 'all-MiniLM-L6-v2'
HF_PREFIX = ''
if 'all-MiniLM' in MODEL_NAME:
    HF_PREFIX = 'sentence-transformers/'
N_EMBD = 384

modes = ['sbert', 'sbert-batchless', 'f32', 'q4_0', 'q4_1', 'f16']

host = "localhost"
port = 8085

os.environ["TOKENIZERS_PARALLELISM"] = "false" # Get rid of the warning spam.

class CppEmbeddingsServerModel():
    def __init__(self, socket):
        self.socket = socket
    def encode(self, sentences, batch_size=32, **kwargs):
        results = []
        for s in sentences:
            self.socket.sendall(s.encode())
            data = self.socket.recv(N_EMBD*4)
            floats = struct.unpack('f' * N_EMBD, data)
            results.append(np.array(floats))
        return results
    
class BatchlessModel():
    def __init__(self, model) -> None:
        self.model = model

    def encode(self, sentences, batch_size=32, **kwargs):
        return self.model.encode(sentences, batch_size=1, **kwargs)

for mode in modes:
    if mode == 'sbert':
        model = SentenceTransformer(f"{HF_PREFIX}{MODEL_NAME}")
    elif mode == 'sbert-batchless':
        model = BatchlessModel(SentenceTransformer(f"{HF_PREFIX}{MODEL_NAME}"))
    else:
        # Start the server process
        server_process = subprocess.Popen(['../build/bin/server', '-m', f'../models/{MODEL_NAME}/ggml-model-{mode}.bin', '-t', '8', '--port', str(port)])
        time.sleep(3)
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((host, port))
        N_EMBD = struct.unpack('i', sock.recv(4))[0]
        model = CppEmbeddingsServerModel(sock)

    evaluation = MTEB(tasks=[
        "STSBenchmark",
        "EmotionClassification",
        ])
    output_folder = f"results/{MODEL_NAME}_{mode}"

    evaluation.run(model, output_folder=output_folder, eval_splits=["test"], task_langs=["en"])

    if not "sbert" in mode:
        sock.close()
        server_process.terminate()
        time.sleep(3)