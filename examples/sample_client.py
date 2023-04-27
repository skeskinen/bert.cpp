import socket
import struct
import numpy as np
import os
import sys

N_EMBD = 384

def embeddings_from_local_server(s, sock):
    sock.sendall(s.encode())
    data = sock.recv(N_EMBD*4)
    floats = struct.unpack('f' * N_EMBD, data)
    return floats

host = "localhost"
port = 8080
if len(sys.argv) > 1:
    port = int(sys.argv[1])

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.connect((host, port))
    N_EMBD = struct.unpack('i', sock.recv(4))[0]

    # Define the function to embed a single text
    def embed_text(text):
        embedding = embeddings_from_local_server(text, sock)
        return np.array(embedding)

    txt_file = "sample_client_texts.txt"
    print(f"Loading texts from {txt_file}...")
    with open(os.path.join(os.path.dirname(__file__), txt_file), 'r') as f:
        texts = f.readlines()

    embedded_texts = [embed_text(text) for text in texts]
    
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
        embedded_text = embed_text(text)
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
