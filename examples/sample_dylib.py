import ctypes
from typing import Union, List
import numpy as np
import os
import sys

N_THREADS = 6

if len(sys.argv) > 1:
    model_path = sys.argv[1]
else:
    print("Usage: python3 sample_dylib.py <ggml model path>")
    exit(0)

class BertModel:
    def __init__(self, fname):
        self.lib = ctypes.cdll.LoadLibrary(os.path.join(os.path.dirname(__file__), "../build/libbert.so"))

        self.lib.bert_load_from_file.restype = ctypes.c_void_p
        self.lib.bert_load_from_file.argtypes = [ctypes.c_char_p]

        self.lib.bert_n_embd.restype = ctypes.c_int32
        self.lib.bert_n_embd.argtypes = [ctypes.c_void_p]
        
        self.lib.bert_free.argtypes = [ctypes.c_void_p]

        self.lib.bert_encode_batch.argtypes = [
            ctypes.c_void_p,    # struct bert_ctx * ctx,
            ctypes.c_int32,     # int32_t n_threads,  
            ctypes.c_int32,     # int32_t n_batch_size
            ctypes.c_int32,     # int32_t n_inputs
            ctypes.POINTER(ctypes.c_char_p),                # const char ** texts
            ctypes.POINTER(ctypes.POINTER(ctypes.c_float)), # float ** embeddings
        ]

        self.ctx = self.lib.bert_load_from_file(fname.encode("utf-8"))
        self.n_embd = self.lib.bert_n_embd(self.ctx)

    def __del__(self):
        self.lib.bert_free(self.ctx)

    def encode(self, sentences: Union[str, List[str]], batch_size: int = 16) -> np.ndarray:
        input_is_string = False
        if isinstance(sentences, str):
            sentences = [sentences]
            input_is_string = True

        n = len(sentences)

        embeddings = np.zeros((n, self.n_embd), dtype=np.float32)
        embeddings_pointers = (ctypes.POINTER(ctypes.c_float) * len(embeddings))(*[e.ctypes.data_as(ctypes.POINTER(ctypes.c_float)) for e in embeddings])

        texts = (ctypes.c_char_p * n)()
        for j, sentence in enumerate(sentences):
            texts[j] = sentence.encode("utf-8")

        self.lib.bert_encode_batch(
            self.ctx, N_THREADS, batch_size, len(sentences), texts, embeddings_pointers
        )
        if input_is_string:
            return embeddings[0]
        return embeddings

def main():    
    model = BertModel(model_path)

    txt_file = "sample_client_texts.txt"
    print(f"Loading texts from {txt_file}...")
    with open(os.path.join(os.path.dirname(__file__), txt_file), 'r') as f:
        texts = f.readlines()

    embedded_texts = model.encode(texts)
    
    print(f"Loaded {len(texts)} lines.")

    def print_results(res):
        (closest_texts, closest_similarities) = res
        # Print the closest texts and their similarity scores
        print("Closest texts:")
        for i, text in enumerate(closest_texts):
            print(f"{i+1}. {text} (similarity score: {closest_similarities[i]:.4f})")

    # Define the function to query the k closest texts
    def query(text, k=3):
        # Embed the input text
        embedded_text = model.encode(text)
        # Compute the cosine similarity between the input text and all the embedded texts
        similarities = [np.dot(embedded_text, embedded_text_i) / (np.linalg.norm(embedded_text) * np.linalg.norm(embedded_text_i)) for embedded_text_i in embedded_texts]
        # Sort the similarities in descending order
        sorted_indices = np.argsort(similarities)[::-1]
        # Return the k closest texts and their similarities
        closest_texts = [texts[i] for i in sorted_indices[:k]]
        closest_similarities = [similarities[i] for i in sorted_indices[:k]]
        return closest_texts, closest_similarities

    test_query = "Should I get health insurance?"
    print(f'Starting with a test query "{test_query}"')
    print_results(query(test_query))

    while True:
        # Prompt the user to enter a text
        input_text = input("Enter a text to find similar texts (enter 'q' to quit): ")
        # If the user enters 'q', exit the loop
        if input_text == 'q':
            break
        # Call the query function to find the closest texts
        print_results(query(input_text))



if __name__ == '__main__':
    main()