#include "bert.h"
#include "ggml.h"

#include <iostream>
#include <string>
#include <vector>
#include <cstring>

#ifdef WIN32
#include "winsock2.h"
#include "include_win/unistd.h"
typedef int socklen_t;
#define read _read
#define close _close

#define SOCKET_HANDLE SOCKET
#else
#include <unistd.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#define SOCKET_HANDLE int
#endif



std::string receive_string(SOCKET_HANDLE socket) {
    static char buffer[1 << 15] = {0};
    ssize_t bytes_received = read(socket, buffer, sizeof(buffer));
    return std::string(buffer, bytes_received);
}

void send_floats(SOCKET_HANDLE socket, const std::vector<float> floats) {
    send(socket, (const char *)floats.data(), floats.size() * sizeof(float), 0);
}

int main(int argc, char ** argv) {
    bert_params params;
    params.model = "../../models/all-MiniLM-L6-v2/ggml-model-q4_0.bin";

    if (bert_params_parse(argc, argv, params) == false) {
        return 1;
    }

    bert_ctx * bctx;

    // load the model
    {
        if ((bctx = bert_load_from_file(params.model)) == nullptr) {
            fprintf(stderr, "%s: failed to load model from '%s'\n", __func__, params.model);
            return 1;
        }
    }

    SOCKET_HANDLE server_fd, new_socket;
    struct sockaddr_in address;
    int addrlen = sizeof(address);


#if WIN32
    WORD wVersionRequested;
    WSADATA wsaData;
    int err;

    /* Use the MAKEWORD(lowbyte, highbyte) macro declared in Windef.h */
    wVersionRequested = MAKEWORD(2, 2);

    err = WSAStartup(wVersionRequested, &wsaData);
    if (err != 0) {
        /* Tell the user that we could not find a usable */
        /* Winsock DLL.                                  */
        printf("WSAStartup failed with error: %d\n", err);
        return 1;
    }

#endif


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
    int n_embd = bert_n_embd(bctx);

    while(true) {
        std::cout << "Waiting for a client" << std::endl;
        if ((new_socket = accept(server_fd, (struct sockaddr *)&address, (socklen_t *)&addrlen)) < 0) {
            std::cerr << "Accept failed" << std::endl;
            return -1;
        }
        std::cout << "New connection" << std::endl;
        send(new_socket, (const char *) & n_embd, sizeof(int), 0);
        while(true) {
            std::string string_in = receive_string(new_socket);
            if (string_in.empty()) {
                break;
            }
            std::vector<float> embeddings = std::vector<float>(n_embd);
            bert_encode(bctx, params.n_threads, string_in.data(), embeddings.data());
            send_floats(new_socket, embeddings);
        }
        close(new_socket);
    }
    close(server_fd);
#if WIN32
    WSACleanup();
#endif
    return 0;
}
