#ifndef BERT_H
#define BERT_H

#include <vector>
#include <string>
#include <map>
#include <thread>

// TODO: make the API in C
// #ifdef __cplusplus
// extern "C" {
// #endif

struct bert_vocab
{
    using id = int32_t;
    using token = std::string;

    std::map<token, id> token_to_id;
    std::map<token, id> subword_token_to_id;

    token id_to_token(const id &arg) const
    {
        auto it = _id_to_token.find(arg);
        if (it != _id_to_token.end())
        {
            return it->second;
        }
        it = _id_to_subword_token.find(arg);
        if (it != _id_to_subword_token.end())
        {
            return it->second;
        }
        return "[UNK TOKEN from bert_vocab]";
    }
    std::map<id, token> _id_to_token;
    std::map<id, token> _id_to_subword_token;
};

std::vector<bert_vocab::id> bert_tokenize(const bert_vocab &vocab, const std::string &text);

struct bert_params
{
    int32_t seed = -1; // RNG seed
    int32_t n_threads = std::min(4, (int32_t)std::thread::hardware_concurrency());
    // int32_t n_predict = 200; // new tokens to predict
    int32_t port = 8080; // server mode port to bind
    // sampling parameters
    /*
    int32_t top_k = 40;
    float   top_p = 0.9f;
    float   temp  = 0.9f;
    */

    // int32_t n_batch = 8; // batch size for prompt processing
    std::string model = "models/gpt-2-117M/ggml-model.bin"; // model path
    std::string prompt;
};

bool bert_params_parse(int argc, char **argv, bert_params &params);

// default hparams (all-MiniLM-L6-v2)
struct bert_hparams
{
    int32_t n_vocab = 30522;
    int32_t n_ctx = 512;
    int32_t n_embd = 256;
    int32_t n_intermediate = 1536;
    int32_t n_head = 12;
    int32_t n_layer = 6;
    int32_t f16 = 1;
};

struct bert_layer
{
    // normalization
    struct ggml_tensor *ln_att_w;
    struct ggml_tensor *ln_att_b;

    struct ggml_tensor *ln_out_w;
    struct ggml_tensor *ln_out_b;

    // attention
    struct ggml_tensor *q_w;
    struct ggml_tensor *q_b;
    struct ggml_tensor *k_w;
    struct ggml_tensor *k_b;
    struct ggml_tensor *v_w;
    struct ggml_tensor *v_b;

    struct ggml_tensor *o_w;
    struct ggml_tensor *o_b;

    // ff
    struct ggml_tensor *ff_i_w;
    struct ggml_tensor *ff_i_b;

    struct ggml_tensor *ff_o_w;
    struct ggml_tensor *ff_o_b;
};

struct bert_model
{
    bert_hparams hparams;

    // embeddings
    // struct ggml_tensor * position_ids;
    struct ggml_tensor *word_embeddings;
    struct ggml_tensor *token_type_embeddings;
    struct ggml_tensor *position_embeddings;

    struct ggml_tensor *ln_e_w;
    struct ggml_tensor *ln_e_b;

    std::vector<bert_layer> layers;

    // key + value memory
    // struct ggml_tensor * memory_k;
    // struct ggml_tensor * memory_v;

    //
    struct ggml_context *ctx;
    std::map<std::string, struct ggml_tensor *> tensors;
};

bool bert_model_load(const std::string &fname, bert_model &model, bert_vocab &vocab);

// evaluate the transformer
//
//   - model:     the model
//   - n_threads: number of threads to use
//   - tokens:  the embeddings of the tokens in the context
//

std::vector<float> bert_eval(
    const bert_model &model,
    const int n_threads,
    const std::vector<bert_vocab::id> &tokens,
    size_t &mem_per_token);

int bert_n_embd(const bert_model& model);

// #ifdef __cplusplus
// }
// #endif

#endif // BERT_H
