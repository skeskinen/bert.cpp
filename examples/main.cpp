#include "bert.h"
#include "ggml.h"

#include <unistd.h>

int main(int argc, char ** argv) {
    const int64_t t_main_start_us = ggml_time_us();

    bert_params params;
    params.model = "../../models/all-MiniLM-L6-v2/ggml-model-f32.bin";

    if (bert_params_parse(argc, argv, params) == false) {
        return 1;
    }

    if (params.seed < 0) {
        params.seed = time(NULL);
    }

    if (params.prompt.empty()) {
        params.prompt = "Hello world";
    }

    int64_t t_load_us = 0;

    bert_vocab vocab;
    bert_model model;

    // load the model
    {
        const int64_t t_start_us = ggml_time_us();

        if (!bert_model_load(params.model, model, vocab)) {
            fprintf(stderr, "%s: failed to load model from '%s'\n", __func__, params.model.c_str());
            return 1;
        }

        t_load_us = ggml_time_us() - t_start_us;
    }

    int64_t t_eval_us  = 0;

    // tokenize the prompt
    std::vector<bert_vocab::id> tokens = ::bert_tokenize(vocab, params.prompt);

    printf("%s: number of tokens in prompt = %zu\n", __func__, tokens.size());
    printf("\n");

    printf("[");
    for (auto& tok : tokens) {
        printf("%d, ", tok);
    }
    printf("]\n");

    for (auto& tok : tokens) {
        printf("%d -> %s\n", tok, vocab.id_to_token(tok).c_str());
    }

    size_t mem_per_token = 0;
    bert_eval(model, params.n_threads, { 0, 1, 2, 3 }, mem_per_token);

    const int64_t t_start_us = ggml_time_us();
    std::vector<float> embeddings = bert_eval(model, params.n_threads, tokens, mem_per_token);
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

    ggml_free(model.ctx);

    return 0;
}