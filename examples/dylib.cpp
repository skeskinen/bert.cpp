#include <dlfcn.h>
#include <string>
#include <iostream>

class BertModel {
public:
    BertModel(const std::string& fname) {
        lib_handle_ = dlopen("../build/libbert.so", RTLD_LAZY);
        if (!lib_handle_) {
            std::cerr << "Failed to load library: " << dlerror() << std::endl;
            std::exit(1);
        }

        bert_load_from_file_ = reinterpret_cast<void*(*)(const char*)>(dlsym(lib_handle_, "bert_load_from_file"));
        bert_n_embd_ = reinterpret_cast<int(*)(void*)>(dlsym(lib_handle_, "bert_n_embd"));
        bert_encode_batch_ = reinterpret_cast<void(*)(void*, int, int, int, const char**, float**)>(dlsym(lib_handle_, "bert_encode_batch"));

        if (!bert_load_from_file_ || !bert_n_embd_ || !bert_encode_batch_) {
            std::cerr << "Failed to load symbols: " << dlerror() << std::endl;
            std::exit(1);
        }

        ctx_ = bert_load_from_file_(fname.c_str());
        n_embd_ = bert_n_embd_(ctx_);
    }

    ~BertModel() {
        dlclose(lib_handle_);
    }

    // ...
private:
    void* lib_handle_;
    void* ctx_;
    int n_embd_;
    void* (*bert_load_from_file_)(const char*);
    int (*bert_n_embd_)(void*);
    void (*bert_encode_batch_)(void*, int, int, int, const char**, float**);
};

int main() {
    BertModel model("../models/all-MiniLM-L6-v2/ggml-model-f16.bin");
    /*
    Potential api, not implemented:
    auto embeddings = model.encode("siikahan se siellä");
    for (auto embedding : embeddings) {
        std::cout << embedding << " ";
    }
    std::cout << std::endl;
    */
    return 0;
}
