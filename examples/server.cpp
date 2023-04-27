#include <iostream>
#include <string>
#include <cstring>
#include <unistd.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include "bert.h"
#include "ggml.h"

std::string receive_string(int socket) {
    static char buffer[1 << 15] = {0};
    ssize_t bytes_received = read(socket, buffer, sizeof(buffer));
    return std::string(buffer, bytes_received);
}

void send_floats(int socket, const std::vector<float> floats) {
    send(socket, floats.data(), floats.size() * sizeof(float), 0);
}

int main(int argc, char ** argv) {
    bert_params params;
    params.model = "../../models/all-MiniLM-L6-v2/ggml-model-f32.bin";

    if (bert_params_parse(argc, argv, params) == false) {
        return 1;
    }

    bert_vocab vocab;
    bert_model model;

    // load the model
    {
        if (!bert_model_load(params.model, model, vocab)) {
            fprintf(stderr, "%s: failed to load model from '%s'\n", __func__, params.model.c_str());
            return 1;
        }
    }
    size_t mem_per_token = 0;
    bert_eval(model, params.n_threads, { 0, 1, 2, 3 }, mem_per_token);

    int server_fd, new_socket;
    struct sockaddr_in address;
    int addrlen = sizeof(address);

    if ((server_fd = socket(AF_INET, SOCK_STREAM, 0)) == 0) {
        std::cerr << "Socket creation failed" << std::endl;
        return -1;
    }

    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(params.port);

    if (bind(server_fd, (struct sockaddr *)&address, sizeof(address)) < 0) {
        std::cerr << "Bind failed" << std::endl;
        return -1;
    }

    if (listen(server_fd, 1) < 0) {
        std::cerr << "Listen failed" << std::endl;
        return -1;
    }

    std::cout << "Server running on port " << params.port << " with " << params.n_threads << " threads" << std::endl;
    int n_embd = bert_n_embd(model);

    while(true) {
        std::cout << "Waiting for a client" << std::endl;
        if ((new_socket = accept(server_fd, (struct sockaddr *)&address, (socklen_t *)&addrlen)) < 0) {
            std::cerr << "Accept failed" << std::endl;
            return -1;
        }
        std::cout << "New connection" << std::endl;
        send(new_socket, &n_embd, sizeof(int), 0);
        while(true) {
            std::string string_in = receive_string(new_socket);

            std::vector<bert_vocab::id> tokens = ::bert_tokenize(vocab, string_in);
            std::cout << tokens.size() << std::endl;
            if (tokens.size() > 2) { // 2 means only cls and sep special tokens.
                auto embeddings = bert_eval(model, params.n_threads, tokens, mem_per_token);
                if (!embeddings.empty()) {
                    send_floats(new_socket, embeddings);
                } else {
                    std::cerr << "Embeddings was empty!" << std::endl;
                }
            } else {
                break;
            }
        }
        close(new_socket);
    }
    close(server_fd);

    return 0;
}
