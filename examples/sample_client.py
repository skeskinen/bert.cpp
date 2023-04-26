import socket
import struct
import numpy as np

N_EMBD = 384

def embeddings_from_local_server(s, sock):
    sock.sendall(s.encode())
    data = sock.recv(N_EMBD*4)
    floats = struct.unpack('f' * N_EMBD, data)
    return floats

texts = ["How do I get a replacement Medicare card?",
        "What is the monthly premium for Medicare Part B?",
        "How do I terminate my Medicare Part B (medical insurance)?",
        "How do I sign up for Medicare?",
        "Can I sign up for Medicare Part B if I am working and have health insurance through an employer?",
        "How do I sign up for Medicare Part B if I already have Part A?",
        "What are Medicare late enrollment penalties?",
        "What is Medicare and who can get it?",
        "How can I get help with my Medicare Part A and Part B premiums?",
        "What are the different parts of Medicare?",
        "Will my Medicare premiums be higher because of my higher income?",
        "What is TRICARE ?",
        "Should I sign up for Medicare Part B if I have Veterans' Benefits?"]

host = "localhost"
port = 8080
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.connect((host, port))

    # Define the function to embed a single text
    def embed_text(text):
        embedding = embeddings_from_local_server(text, sock)
        return np.array(embedding)

    # Embed all the texts in the list
    embedded_texts = [embed_text(text) for text in texts]

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
