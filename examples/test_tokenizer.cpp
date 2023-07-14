#include "bert.h"
#include "ggml.h"

#include <unistd.h>
#include <map>
#include <algorithm>
#include <stdio.h>
#include <string>
#include <vector>
#define ANSI_COLOR_RED     "\x1b[31m"
#define ANSI_COLOR_RESET   "\x1b[0m"
#define ANSI_COLOR_GREEN   "\x1b[32m"

void tokenizer_test(bert_ctx * ctx, const std::string& input, const std::vector<bert_vocab_id>& expected) {
    int N = bert_n_max_tokens(ctx);
    std::vector<bert_vocab_id> result(N);
    int n_tokens;
    bert_tokenize(ctx, input.c_str(), result.data(), &n_tokens, N);
    result.resize(n_tokens);
    if (result != expected) {
        printf("tokenizer test failed: '%.*s'\n", 16000, input.data());

        printf("[");
        for (auto& tok : result) {
            printf("%d, ", tok);
        }
        printf("]\n");

        for (size_t i = 0; i < result.size(); i++) {
            bert_vocab_id a = expected[std::min(i, expected.size()-1)];
            bert_vocab_id b = result[i];
            const char *color_start = (a == b) ? ANSI_COLOR_GREEN : ANSI_COLOR_RED;
            const char *color_end = ANSI_COLOR_RESET;

            printf("%s%d -> %s : %d -> %s%s\n", color_start, a, bert_vocab_id_to_token(ctx, a), b, bert_vocab_id_to_token(ctx, b), color_end);
        }
    } else {
        printf("Success '%.*s...'\n", 16, input.data());
    }
}



int main(int argc, char ** argv) {

    bert_params params;
    params.model = "models/all-MiniLM-L6-v2/ggml-model.bin";

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
    /*
    for (auto &kv : vocab.id_to_token) {
        printf("%d -> %s\n", kv.first, kv.second.c_str());
    }
    */

    // tokenizer tests:
    
    tokenizer_test(bctx, "Québec", {101, 5447, 102});
    tokenizer_test(bctx, "syömme \t  täällä    tänään", {101, 25353, 5358, 4168, 11937, 25425, 9092, 14634, 102});
    tokenizer_test(bctx, "I'm going to the store to buy 3 apples and a banana! You're welcome to come along if you'd like. The time is 2:30 p.m. and it's partly cloudy outside. I'll be back soon, so don't go anywhere.", {101, 1045, 1005, 1049, 2183, 2000, 1996, 3573, 2000, 4965, 1017, 18108, 1998, 1037, 15212, 999, 2017, 1005, 2128, 6160, 2000, 2272, 2247, 2065, 2017, 1005, 1040, 2066, 1012, 1996, 2051, 2003, 1016, 1024, 2382, 1052, 1012, 1049, 1012, 1998, 2009, 1005, 1055, 6576, 24706, 2648, 1012, 1045, 1005, 2222, 2022, 2067, 2574, 1010, 2061, 2123, 1005, 1056, 2175, 5973, 1012, 102});
    tokenizer_test(bctx, "\"5 2 + 3 * 4 -\"; int stack[1000], top = -1; int calculate(int a, int b, char operator) { return operator == '+' ? a + b : operator == '-' ? a - b : operator == '*' ? a * b : a / b; } void push(int x) { stack[++top] = x; } int pop() { return stack[top--]; } int evaluatePostfix(char* expression) { for (int i = 0; expression[i]; i++) { if (isdigit(expression[i])) push(expression[i] - '0'); else { int a = pop(), b = pop(); push(calculate(b, a, expression[i])); } } return pop(); } int result = evaluatePostfix(input);", {101, 1000, 1019, 1016, 1009, 1017, 1008, 1018, 1011, 1000, 1025, 20014, 9991, 1031, 6694, 1033, 1010, 2327, 1027, 1011, 1015, 1025, 20014, 18422, 1006, 20014, 1037, 1010, 20014, 1038, 1010, 25869, 6872, 1007, 1063, 2709, 6872, 1027, 1027, 1005, 1009, 1005, 1029, 1037, 1009, 1038, 1024, 6872, 1027, 1027, 1005, 1011, 1005, 1029, 1037, 1011, 1038, 1024, 6872, 1027, 1027, 1005, 1008, 1005, 1029, 1037, 1008, 1038, 1024, 1037, 1013, 1038, 1025, 1065, 11675, 5245, 1006, 20014, 1060, 1007, 1063, 9991, 1031, 1009, 1009, 2327, 1033, 1027, 1060, 1025, 1065, 20014, 3769, 1006, 1007, 1063, 2709, 9991, 1031, 2327, 1011, 1011, 1033, 1025, 1065, 20014, 16157, 19894, 8873, 2595, 1006, 25869, 1008, 3670, 1007, 1063, 2005, 1006, 20014, 1045, 1027, 1014, 1025, 3670, 1031, 1045, 1033, 1025, 1045, 1009, 1009, 1007, 1063, 2065, 1006, 2003, 4305, 23806, 1006, 3670, 1031, 1045, 1033, 1007, 1007, 5245, 1006, 3670, 1031, 1045, 1033, 1011, 1005, 1014, 1005, 1007, 1025, 2842, 1063, 20014, 1037, 1027, 3769, 1006, 1007, 1010, 1038, 1027, 3769, 1006, 1007, 1025, 5245, 1006, 18422, 1006, 1038, 1010, 1037, 1010, 3670, 1031, 1045, 1033, 1007, 1007, 1025, 1065, 1065, 2709, 3769, 1006, 1007, 1025, 1065, 20014, 2765, 1027, 16157, 19894, 8873, 2595, 1006, 7953, 1007, 1025, 102});
}