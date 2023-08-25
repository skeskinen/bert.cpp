#include "bert.h"
#include "ggml.h"

#include <unistd.h>
#include <stdio.h>
#include <vector>

int main(int argc, char ** argv) {
    ggml_time_init();
    const int64_t t_main_start_us = ggml_time_us();

    bert_params params;
    params.model = "../../models/all-MiniLM-L6-v2/ggml-model-f32.bin";

    if (bert_params_parse(argc, argv, params) == false) {
        return 1;
    }

    int64_t t_load_us = 0;

    bert_ctx * bctx;

    // load the model
    {
        const int64_t t_start_us = ggml_time_us();

        if ((bctx = bert_load_from_file(params.model)) == nullptr) {
            fprintf(stderr, "%s: failed to load model from '%s'\n", __func__, params.model);
            return 1;
        }

        t_load_us = ggml_time_us() - t_start_us;
    }

    int64_t t_eval_us  = 0;
    int64_t t_start_us = ggml_time_us();
    int N = bert_n_max_tokens(bctx);
    // tokenize the prompt
    std::vector<bert_vocab_id> tokens(N);
    int n_tokens;
    bert_tokenize(bctx, params.prompt, tokens.data(), &n_tokens, N);
    tokens.resize(n_tokens);

    printf("%s: number of tokens in prompt = %zu\n", __func__, tokens.size());
    printf("\n");

    printf("[");
    for (auto& tok : tokens) {
        printf("%d, ", tok);
    }
    printf("]\n");

    for (auto& tok : tokens) {
        printf("%d -> %s\n", tok, bert_vocab_id_to_token(bctx, tok));
    }
    std::vector<float> embeddings(bert_n_embd(bctx));
    bert_eval(bctx, params.n_threads, tokens.data(), n_tokens, embeddings.data());
    t_eval_us += ggml_time_us() - t_start_us;
    
    printf("[");
    for(auto e : embeddings) {
        printf("%1.4f, ", e);
    }
    printf("]\n");

    // report timing
    {
        const int64_t t_main_end_us = ggml_time_us();

        printf("\n\n");
        //printf("%s: mem per token = %8zu bytes\n", __func__, mem_per_token);
        printf("%s:     load time = %8.2f ms\n", __func__, t_load_us/1000.0f);
        printf("%s:  eval time = %8.2f ms / %.2f ms per token\n", __func__, t_eval_us/1000.0f, t_eval_us/1000.0f/tokens.size());
        printf("%s:    total time = %8.2f ms\n", __func__, (t_main_end_us - t_main_start_us)/1000.0f);
    }

    return 0;
}