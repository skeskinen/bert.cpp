#include "bert.h"
#include "ggml/ggml.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>
#include <iostream>
#include <unistd.h>
#include <regex>
#include <thread>

void bert_print_usage(char **argv, const bert_params &params)
{
    fprintf(stderr, "usage: %s [options]\n", argv[0]);
    fprintf(stderr, "\n");
    fprintf(stderr, "options:\n");
    fprintf(stderr, "  -h, --help            show this help message and exit\n");
    fprintf(stderr, "  -s SEED, --seed SEED  RNG seed (default: -1)\n");
    fprintf(stderr, "  -t N, --threads N     number of threads to use during computation (default: %d)\n", params.n_threads);
    fprintf(stderr, "  -p PROMPT, --prompt PROMPT\n");
    fprintf(stderr, "                        prompt to start generation with (default: random)\n");
    fprintf(stderr, "  --port p     port to bind in server mode (default: %d)\n", params.port);
    fprintf(stderr, "  -m FNAME, --model FNAME\n");
    fprintf(stderr, "                        model path (default: %s)\n", params.model.c_str());
    fprintf(stderr, "\n");
}

bool bert_params_parse(int argc, char **argv, bert_params &params)
{
    for (int i = 1; i < argc; i++)
    {
        std::string arg = argv[i];

        if (arg == "-s" || arg == "--seed")
        {
            params.seed = std::stoi(argv[++i]);
        }
        else if (arg == "-t" || arg == "--threads")
        {
            params.n_threads = std::stoi(argv[++i]);
        }
        else if (arg == "-p" || arg == "--prompt")
        {
            params.prompt = argv[++i];
        }
        else if (arg == "--port")
        {
            params.port = std::stoi(argv[++i]);
        }
        else if (arg == "-m" || arg == "--model")
        {
            params.model = argv[++i];
        }
        else if (arg == "-h" || arg == "--help")
        {
            bert_print_usage(argv, params);
            exit(0);
        }
        else
        {
            fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
            bert_print_usage(argv, params);
            exit(0);
        }
    }

    return true;
}

static size_t utf8_len(char src)
{
    const size_t lookup[] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4};
    uint8_t highbits = static_cast<uint8_t>(src) >> 4;
    return lookup[highbits];
}

std::string stripAccents(const std::string &inputString)
{
    std::string resultString;
    std::unordered_map<std::string, char> accentMap = {{"À", 'A'},{"Á", 'A'},
        {"Â", 'A'},{"Ã", 'A'},{"Ä", 'A'},{"Å", 'A'},{"à", 'a'},{"á", 'a'},
        {"â", 'a'},{"ã", 'a'},{"ä", 'a'},{"å", 'a'},{"È", 'E'},{"É", 'E'},
        {"Ê", 'E'},{"Ë", 'E'},{"è", 'e'},{"é", 'e'},{"ê", 'e'},{"ë", 'e'},
        {"Ì", 'I'},{"Í", 'I'},{"Î", 'I'},{"Ï", 'I'},{"ì", 'i'},{"í", 'i'},
        {"î", 'i'},{"ï", 'i'},{"Ò", 'O'},{"Ó", 'O'},{"Ô", 'O'},{"Õ", 'O'},
        {"Ö", 'O'},{"ò", 'o'},{"ó", 'o'},{"ô", 'o'},{"õ", 'o'},{"ö", 'o'},
        {"Ù", 'U'},{"Ú", 'U'},{"Û", 'U'},{"Ü", 'U'},{"ù", 'u'},{"ú", 'u'},
        {"û", 'u'},{"ü", 'u'},{"Ý", 'Y'},{"ý", 'y'},{"Ç", 'C'},{"ç", 'c'},
        {"Ñ", 'N'},{"ñ", 'n'},
    };

    for (size_t i = 0; i < inputString.length();)
    {
        int len = utf8_len(inputString[i]);
        std::string curChar = inputString.substr(i, len);
        auto iter = accentMap.find(curChar);
        if (iter != accentMap.end())
        {
            resultString += iter->second;
        }
        else
        {
            resultString += curChar;
        }
        i += len;
    }

    return resultString;
}

std::string bert_normalize_prompt(const std::string &text)
{
    // TODO: handle chinese characters? https://github.com/huggingface/tokenizers/blob/ef5f50605ddf9f8caef1598c0e4853862b9707a7/tokenizers/src/normalizers/bert.rs#L98
    std::string text2 = stripAccents(text);
    for (size_t i = 0; i < text2.size(); i += utf8_len(text2[i]))
    {
        char c = text2[i];
        if (c >= 'A' && c <= 'Z')
            text2[i] = c - 'A' + 'a';
    }
    return text2;
}

std::vector<bert_vocab::id> bert_tokenize(const bert_vocab &vocab, const std::string &text)
{
    int cls_tok_id = 101;
    int sep_tok_id = 102;
    std::vector<std::string> words;
    // first split the text into words
    {
        std::string str = bert_normalize_prompt(text);
        // std::string pat = R"('s|'t|'re|'ve|'m|'ll|'d| ?[[:alpha:]]+| ?[[:digit:]]+| ?[^\s[:alpha:][:digit:]]+|\s+(?!\S)|\s+)";
        std::string pat = R"([[:punct:]]|[[:alpha:]]+|[[:digit:]]+)";

        std::regex re(pat);
        std::smatch m;

        while (std::regex_search(str, m, re))
        {
            for (std::string x : m)
            {
                words.push_back(x);
            }
            str = m.suffix();
        }
    }

    // find the longest tokens that form the words:
    std::vector<bert_vocab::id> tokens{cls_tok_id};
    for (const auto &word : words)
    {
        if (word.size() == 0)
            continue;

        // printf("processing word '%s\'\n", word.c_str());
        int i = 0;
        int n = word.size();
        auto *token_map = &vocab.token_to_id;
    loop:
        while (i < n)
        {
            int j = n;
            while (j > i)
            {
                // printf("trying to find: '%s'\n", word.substr(i, j-i).c_str());
                auto it = token_map->find(word.substr(i, j - i));
                if (it != token_map->end())
                {
                    // printf("matched!\n");
                    tokens.push_back(it->second);
                    i = j;
                    token_map = &vocab.subword_token_to_id;
                    goto loop;
                }
                --j;
            }
            if (i == n)
            {
                break;
            }
            if (j == i)
            {
                fprintf(stderr, "%s: unknown token '%s'\n", __func__, word.substr(i, 1).data());
                token_map = &vocab.subword_token_to_id;
                ++i;
            }
        }
    }
    tokens.push_back(sep_tok_id);
    return tokens;
}

// load the model's weights from a file
bool bert_model_load(const std::string &fname, bert_model &model, bert_vocab &vocab)
{
    printf("%s: loading model from '%s' - please wait ...\n", __func__, fname.c_str());

    auto fin = std::ifstream(fname, std::ios::binary);
    if (!fin)
    {
        fprintf(stderr, "%s: failed to open '%s'\n", __func__, fname.c_str());
        return false;
    }

    // verify magic
    {
        uint32_t magic;
        fin.read((char *)&magic, sizeof(magic));
        if (magic != 0x67676d6c)
        {
            fprintf(stderr, "%s: invalid model file '%s' (bad magic)\n", __func__, fname.c_str());
            return false;
        }
    }

    // load hparams
    {
        auto &hparams = model.hparams;

        fin.read((char *)&hparams.n_vocab, sizeof(hparams.n_vocab));
        fin.read((char *)&hparams.n_ctx, sizeof(hparams.n_ctx));
        fin.read((char *)&hparams.n_embd, sizeof(hparams.n_embd));
        fin.read((char *)&hparams.n_intermediate, sizeof(hparams.n_intermediate));
        fin.read((char *)&hparams.n_head, sizeof(hparams.n_head));
        fin.read((char *)&hparams.n_layer, sizeof(hparams.n_layer));
        fin.read((char *)&hparams.f16, sizeof(hparams.f16));

        printf("%s: n_vocab = %d\n", __func__, hparams.n_vocab);
        printf("%s: n_ctx   = %d\n", __func__, hparams.n_ctx);
        printf("%s: n_embd  = %d\n", __func__, hparams.n_embd);
        printf("%s: n_intermediate  = %d\n", __func__, hparams.n_intermediate);
        printf("%s: n_head  = %d\n", __func__, hparams.n_head);
        printf("%s: n_layer = %d\n", __func__, hparams.n_layer);
        printf("%s: f16     = %d\n", __func__, hparams.f16);
    }

    // load vocab
    {
        int32_t n_vocab = model.hparams.n_vocab;

        /*
        fin.read((char *) &n_vocab, sizeof(n_vocab));

        if (n_vocab != model.hparams.n_vocab) {
            fprintf(stderr, "%s: invalid model file '%s' (bad vocab size %d != %d)\n",
                    __func__, fname.c_str(), n_vocab, model.hparams.n_vocab);
            return false;
        }
        */

        std::string word;
        for (int i = 0; i < n_vocab; i++)
        {
            uint32_t len;
            fin.read((char *)&len, sizeof(len));

            word.resize(len);
            fin.read((char *)word.data(), len);

            if (word[0] == '#' && word[1] == '#')
            {
                vocab.subword_token_to_id[word.substr(2)] = i;
                vocab._id_to_subword_token[i] = word;
            }

            if (vocab.token_to_id.count(word) == 0)
            {
                vocab.token_to_id[word] = i;
                vocab._id_to_token[i] = word;
            }
        }
    }

    // for the big tensors, we have the option to store the data in 16-bit floats or quantized
    // in order to save memory and also to speed up the computation
    ggml_type wtype = GGML_TYPE_COUNT;
    switch (model.hparams.f16)
    {
    case 0:
        wtype = GGML_TYPE_F32;
        break;
    case 1:
        wtype = GGML_TYPE_F16;
        break;
    case 2:
        wtype = GGML_TYPE_Q4_0;
        break;
    case 3:
        wtype = GGML_TYPE_Q4_1;
        break;
    default:
    {
        fprintf(stderr, "%s: invalid model file '%s' (bad f16 value %d)\n",
                __func__, fname.c_str(), model.hparams.f16);
        return false;
    }
    }

    auto &ctx = model.ctx;

    size_t ctx_size = 0;

    {
        const auto &hparams = model.hparams;

        const int n_embd = hparams.n_embd;
        const int n_layer = hparams.n_layer;
        const int n_ctx = hparams.n_ctx;
        const int n_intermediate = hparams.n_intermediate;
        const int n_vocab = hparams.n_vocab;

        // Calculate size requirements

        ctx_size += n_embd * n_vocab * ggml_type_sizef(wtype); // word_embeddings
        ctx_size += n_embd * 2 * ggml_type_sizef(wtype); // token_type_embeddings
        ctx_size += n_embd * n_ctx * ggml_type_sizef(wtype); // position_embeddings

        ctx_size += 2 * n_embd * ggml_type_sizef(GGML_TYPE_F32); // ln_e_*

        ctx_size += 4 * n_layer * (n_embd * ggml_type_sizef(GGML_TYPE_F32)); // ln_*

        ctx_size += 4 * n_layer * (n_embd * n_embd * ggml_type_sizef(wtype)); // kqvo weights
        ctx_size += 4 * n_layer * (n_embd * ggml_type_sizef(GGML_TYPE_F32)); // kqvo bias

        ctx_size += 2 * n_layer * (n_embd * n_intermediate * ggml_type_sizef(wtype)); // ff_*_w
        ctx_size += n_layer * (n_intermediate * ggml_type_sizef(GGML_TYPE_F32)); // ff_i_b
        ctx_size += n_layer * (n_embd * ggml_type_sizef(GGML_TYPE_F32)); // ff_o_b

        ctx_size += (5 + 16 * n_layer) * 256; // object overhead

        printf("%s: ggml ctx size = %6.2f MB\n", __func__, ctx_size / (1024.0 * 1024.0));
    }

    // create the ggml context
    {
        struct ggml_init_params params = {
            .mem_size = ctx_size,
            .mem_buffer = NULL,
            .no_alloc = false,
        };

        model.ctx = ggml_init(params);
        if (!model.ctx)
        {
            fprintf(stderr, "%s: ggml_init() failed\n", __func__);
            return false;
        }
    }

    // prepare memory for the weights
    {
        const auto &hparams = model.hparams;

        const int n_embd = hparams.n_embd;
        const int n_layer = hparams.n_layer;
        const int n_intermediate = hparams.n_intermediate;
        const int n_ctx = hparams.n_ctx;
        const int n_vocab = hparams.n_vocab;

        model.layers.resize(n_layer);

        model.word_embeddings = ggml_new_tensor_2d(ctx, wtype, n_embd, n_vocab);
        model.token_type_embeddings = ggml_new_tensor_2d(ctx, wtype, n_embd, 2);
        model.position_embeddings = ggml_new_tensor_2d(ctx, wtype, n_embd, n_ctx);

        model.ln_e_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);
        model.ln_e_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);

        // map by name
        model.tensors["embeddings.word_embeddings.weight"] = model.word_embeddings;
        model.tensors["embeddings.token_type_embeddings.weight"] = model.token_type_embeddings;
        model.tensors["embeddings.position_embeddings.weight"] = model.position_embeddings;

        model.tensors["embeddings.LayerNorm.weight"] = model.ln_e_w;
        model.tensors["embeddings.LayerNorm.bias"] = model.ln_e_b;

        for (int i = 0; i < n_layer; ++i)
        {
            auto &layer = model.layers[i];

            layer.ln_att_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);
            layer.ln_att_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);
            layer.ln_out_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);
            layer.ln_out_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);

            layer.q_w = ggml_new_tensor_2d(ctx, wtype, n_embd, n_embd);
            layer.q_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);
            layer.k_w = ggml_new_tensor_2d(ctx, wtype, n_embd, n_embd);
            layer.k_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);
            layer.v_w = ggml_new_tensor_2d(ctx, wtype, n_embd, n_embd);
            layer.v_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);
            layer.o_w = ggml_new_tensor_2d(ctx, wtype, n_embd, n_embd);
            layer.o_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);

            layer.ff_i_w = ggml_new_tensor_2d(ctx, wtype, n_embd, n_intermediate);
            layer.ff_i_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_intermediate);

            layer.ff_o_w = ggml_new_tensor_2d(ctx, wtype, n_intermediate, n_embd);
            layer.ff_o_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);

            // map by name

            model.tensors["encoder.layer." + std::to_string(i) + ".attention.self.query.weight"] = layer.q_w;
            model.tensors["encoder.layer." + std::to_string(i) + ".attention.self.query.bias"] = layer.q_b;
            model.tensors["encoder.layer." + std::to_string(i) + ".attention.self.key.weight"] = layer.k_w;
            model.tensors["encoder.layer." + std::to_string(i) + ".attention.self.key.bias"] = layer.k_b;
            model.tensors["encoder.layer." + std::to_string(i) + ".attention.self.value.weight"] = layer.v_w;
            model.tensors["encoder.layer." + std::to_string(i) + ".attention.self.value.bias"] = layer.v_b;
            model.tensors["encoder.layer." + std::to_string(i) + ".attention.output.LayerNorm.weight"] = layer.ln_att_w;
            model.tensors["encoder.layer." + std::to_string(i) + ".attention.output.LayerNorm.bias"] = layer.ln_att_b;
            model.tensors["encoder.layer." + std::to_string(i) + ".attention.output.dense.weight"] = layer.o_w;
            model.tensors["encoder.layer." + std::to_string(i) + ".attention.output.dense.bias"] = layer.o_b;

            model.tensors["encoder.layer." + std::to_string(i) + ".intermediate.dense.weight"] = layer.ff_i_w;
            model.tensors["encoder.layer." + std::to_string(i) + ".intermediate.dense.bias"] = layer.ff_i_b;

            model.tensors["encoder.layer." + std::to_string(i) + ".output.LayerNorm.weight"] = layer.ln_out_w;
            model.tensors["encoder.layer." + std::to_string(i) + ".output.LayerNorm.bias"] = layer.ln_out_b;
            model.tensors["encoder.layer." + std::to_string(i) + ".output.dense.weight"] = layer.ff_o_w;
            model.tensors["encoder.layer." + std::to_string(i) + ".output.dense.bias"] = layer.ff_o_b;
        }
    }

    // key + value memory
    /*
    {
        const auto & hparams = model.hparams;

        const int n_embd  = hparams.n_embd;
        const int n_layer = hparams.n_layer;
        const int n_ctx   = hparams.n_ctx;

        const int n_mem      = n_layer*n_ctx;
        const int n_elements = n_embd*n_mem;

        model.memory_k = ggml_new_tensor_1d(ctx, GGML_TYPE_F16, n_elements);
        model.memory_v = ggml_new_tensor_1d(ctx, GGML_TYPE_F16, n_elements);

        const size_t memory_size = ggml_nbytes(model.memory_k) + ggml_nbytes(model.memory_v);

        printf("%s: memory_size = %8.2f MB, n_mem = %d\n", __func__, memory_size/1024.0/1024.0, n_mem);
    }
    */

    // load weights
    {
        int n_tensors = 0;
        size_t total_size = 0;

        printf("%s: ", __func__);

        while (true)
        {
            int32_t n_dims;
            int32_t length;
            int32_t ftype;

            fin.read(reinterpret_cast<char *>(&n_dims), sizeof(n_dims));
            fin.read(reinterpret_cast<char *>(&length), sizeof(length));
            fin.read(reinterpret_cast<char *>(&ftype), sizeof(ftype));

            if (fin.eof())
            {
                break;
            }

            int64_t nelements = 1;
            int64_t ne[2] = {1, 1};
            for (int i = 0; i < n_dims; ++i)
            {
                int32_t ne_cur;
                fin.read(reinterpret_cast<char *>(&ne_cur), sizeof(ne_cur));
                ne[i] = ne_cur;
                nelements *= ne[i];
            }

            std::string name(length, 0);
            fin.read(&name[0], length);

            if (model.tensors.find(name.data()) == model.tensors.end())
            {
                fprintf(stderr, "%s: unknown tensor '%s' in model file\n", __func__, name.data());
                return false;
            }

            auto tensor = model.tensors[name.data()];
            if (ggml_nelements(tensor) != nelements)
            {
                fprintf(stderr, "%s: tensor '%s' has wrong size in model file\n", __func__, name.data());
                return false;
            }

            if (tensor->ne[0] != ne[0] || tensor->ne[1] != ne[1])
            {
                fprintf(stderr, "%s: tensor '%s' has wrong shape in model file: got [%lld, %lld], expected [%lld, %lld]\n",
                        __func__, name.data(), tensor->ne[0], tensor->ne[1], ne[0], ne[1]);
                return false;
            }

            if (0)
            {
                static const char *ftype_str[] = {
                    "f32",
                    "f16",
                    "q4_0",
                    "q4_1",
                };
                printf("%24s - [%5lld, %5lld], type = %6s, %6.2f MB, %9zu bytes\n", name.data(), ne[0], ne[1], ftype_str[ftype], ggml_nbytes(tensor) / 1024.0 / 1024.0, ggml_nbytes(tensor));
            }

            size_t bpe = 0;

            switch (ftype)
            {
            case 0:
                bpe = ggml_type_size(GGML_TYPE_F32);
                break;
            case 1:
                bpe = ggml_type_size(GGML_TYPE_F16);
                break;
            case 2:
                bpe = ggml_type_size(GGML_TYPE_Q4_0);
                assert(ne[0] % 64 == 0);
                break;
            case 3:
                bpe = ggml_type_size(GGML_TYPE_Q4_1);
                assert(ne[0] % 64 == 0);
                break;
            default:
            {
                fprintf(stderr, "%s: unknown ftype %d in model file\n", __func__, ftype);
                return false;
            }
            };

            if ((nelements * bpe) / ggml_blck_size(tensor->type) != ggml_nbytes(tensor))
            {
                fprintf(stderr, "%s: tensor '%s' has wrong size in model file: got %zu, expected %llu\n",
                        __func__, name.data(), ggml_nbytes(tensor), nelements * bpe);
                return false;
            }

            fin.read(reinterpret_cast<char *>(tensor->data), ggml_nbytes(tensor));

            // printf("%42s - [%5d, %5d], type = %6s, %6.2f MB\n", name.data(), ne[0], ne[1], ftype == 0 ? "float" : "f16", ggml_nbytes(tensor)/1024.0/1024.0);
            total_size += ggml_nbytes(tensor);
            if (++n_tensors % 8 == 0)
            {
                printf(".");
                fflush(stdout);
            }
        }

        printf(" done\n");

        printf("%s: model size = %8.2f MB / num tensors = %d\n", __func__, total_size / 1024.0 / 1024.0, n_tensors);
    }

    fin.close();

    return true;
}

/*
void pretty_print_tensor(const void *data, const int64_t *shape, const size_t *strides, int cur_dim, const std::string &indent)
{
    if (cur_dim == 0)
    {
        printf("%s[", indent.c_str());
        if (shape[0] > 6)
        {
            for (size_t i = 0; i < 3; i++)
            {
                printf("%1.4f, ", *((float *)(data + i * strides[0])));
            }
            printf("..., ");
            for (size_t i = shape[0] - 3; i < shape[0]; i++)
            {
                printf("%1.4f, ", *((float *)(data + i * strides[0])));
            }
        }
        else
        {
            for (size_t i = 0; i < shape[0]; i++)
            {
                printf("%1.4f, ", *((float *)(data + i * strides[0])));
            }
        }
        printf("]");
        return;
    }

    printf("%s[\n", indent.c_str());
    std::string inner_indent = indent + "\t";
    for (int i = 0; i < shape[cur_dim]; i++)
    {
        pretty_print_tensor(data + i * strides[cur_dim], shape, strides, cur_dim - 1, inner_indent);

        if (i < shape[cur_dim] - 1)
        {
            printf(",");
        }
        printf("\n");
    }
    printf("%s]", indent.c_str());
}
*/

std::vector<float> bert_eval(
    const bert_model &model,
    const int n_threads,
    const std::vector<bert_vocab::id> &tokens,
    size_t &mem_per_token)
{

    const int N = tokens.size();

    const auto &hparams = model.hparams;

    const int n_embd = hparams.n_embd;
    const int n_layer = hparams.n_layer;
    const int n_ctx = hparams.n_ctx;
    const int n_head = hparams.n_head;

    const int d_head = n_embd / n_head;

    std::vector<float> result;
    if (tokens.size() > (size_t)n_ctx)
    {
        fprintf(stderr, "Too many tokens, maximum is %d\n", n_ctx);
        return result;
    }

    static size_t buf_size = 256u * 1024 * 1024;
    static void *buf = malloc(buf_size);

    if (mem_per_token > 0 && mem_per_token * N > buf_size)
    {
        const size_t buf_size_new = 1.1 * (mem_per_token * N); // add 10% to account for ggml object overhead
        // printf("\n%s: reallocating buffer from %zu to %zu bytes\n", __func__, buf_size, buf_size_new);

        // reallocate
        buf_size = buf_size_new;
        buf = realloc(buf, buf_size);
        if (buf == nullptr)
        {
            fprintf(stderr, "%s: failed to allocate %zu bytes\n", __func__, buf_size);
            return result;
        }
    }

    struct ggml_init_params params = {
        .mem_size = buf_size,
        .mem_buffer = buf,
        .no_alloc = false,
    };

    struct ggml_context *ctx0 = ggml_init(params);
    struct ggml_cgraph gf = {};
    gf.n_threads = n_threads;

    // Embeddings. word_embeddings + token_type_embeddings + position_embeddings
    struct ggml_tensor *token_layer = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, N);
    memcpy(token_layer->data, tokens.data(), N * ggml_element_size(token_layer));

    struct ggml_tensor *token_types = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, N);
    ggml_set_zero(token_types);

    struct ggml_tensor *positions = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, N);
    for (int i = 0; i < N; i++)
    {
        ggml_set_i32_1d(positions, i, i);
    }

    struct ggml_tensor *inpL = ggml_get_rows(ctx0, model.word_embeddings, token_layer);

    inpL = ggml_add(ctx0,
                    ggml_get_rows(ctx0, model.token_type_embeddings, token_types),
                    inpL);
    inpL = ggml_add(ctx0,
                    ggml_get_rows(ctx0, model.position_embeddings, positions),
                    inpL);

    // embd norm
    {
        inpL = ggml_norm(ctx0, inpL);

        inpL = ggml_add(ctx0,
                        ggml_mul(ctx0,
                                 ggml_repeat(ctx0, model.ln_e_w, inpL),
                                 inpL),
                        ggml_repeat(ctx0, model.ln_e_b, inpL));
    }
    // layers
    for (int il = 0; il < n_layer; il++)
    {
        struct ggml_tensor *cur = inpL;

        // self-attention
        {
            struct ggml_tensor *Qcur = cur;
            Qcur = ggml_reshape_3d(ctx0,
                                   ggml_add(ctx0, ggml_repeat(ctx0, model.layers[il].q_b, Qcur),
                                            ggml_mul_mat(ctx0, model.layers[il].q_w, Qcur)),
                                   d_head, n_head, N);
            struct ggml_tensor *Q = ggml_permute(ctx0, Qcur, 0, 2, 1, 3);

            struct ggml_tensor *Kcur = cur;
            Kcur = ggml_reshape_3d(ctx0,
                                   ggml_add(ctx0, ggml_repeat(ctx0, model.layers[il].k_b, Kcur),
                                            ggml_mul_mat(ctx0, model.layers[il].k_w, Kcur)),
                                   d_head, n_head, N);
            struct ggml_tensor *K = ggml_permute(ctx0, Kcur, 0, 2, 1, 3);

            struct ggml_tensor *Vcur = cur;
            Vcur = ggml_reshape_3d(ctx0,
                                   ggml_add(ctx0, ggml_repeat(ctx0, model.layers[il].v_b, Vcur),
                                            ggml_mul_mat(ctx0, model.layers[il].v_w, Vcur)),
                                   d_head, n_head, N);
            struct ggml_tensor *V = ggml_permute(ctx0, Vcur, 0, 2, 1, 3);

            struct ggml_tensor *KQ = ggml_mul_mat(ctx0, K, Q);
            // KQ = soft_max(KQ / sqrt(head width))
            KQ = ggml_soft_max(ctx0,
                               ggml_scale(ctx0,
                                          KQ,
                                          ggml_new_f32(ctx0, 1.0f / sqrt((float)d_head))));

            V = ggml_cont(ctx0, ggml_transpose(ctx0, V));
            struct ggml_tensor *KQV = ggml_mul_mat(ctx0, V, KQ);
            KQV = ggml_permute(ctx0, KQV, 0, 2, 1, 3);

            cur = ggml_cpy(ctx0,
                           KQV,
                           ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd, N));
        }
        // attention output
        cur = ggml_add(ctx0,
                       ggml_repeat(ctx0, model.layers[il].o_b, cur),
                       ggml_mul_mat(ctx0, model.layers[il].o_w, cur));

        // re-add the layer input
        cur = ggml_add(ctx0, cur, inpL);

        // attention norm
        {
            cur = ggml_norm(ctx0, cur);

            cur = ggml_add(ctx0,
                           ggml_mul(ctx0,
                                    ggml_repeat(ctx0, model.layers[il].ln_att_w, cur),
                                    cur),
                           ggml_repeat(ctx0, model.layers[il].ln_att_b, cur));
        }
        struct ggml_tensor *att_output = cur;
        // intermediate_output = self.intermediate(attention_output)
        cur = ggml_mul_mat(ctx0, model.layers[il].ff_i_w, cur);
        cur = ggml_add(ctx0,
                       ggml_repeat(ctx0, model.layers[il].ff_i_b, cur),
                       cur);
        cur = ggml_gelu(ctx0, cur);

        // layer_output = self.output(intermediate_output, attention_output)
        cur = ggml_mul_mat(ctx0, model.layers[il].ff_o_w, cur);
        cur = ggml_add(ctx0,
                       ggml_repeat(ctx0, model.layers[il].ff_o_b, cur),
                       cur);
        // attentions bypass the intermediate layer
        cur = ggml_add(ctx0, att_output, cur);

        // output norm
        {
            cur = ggml_norm(ctx0, cur);

            cur = ggml_add(ctx0,
                           ggml_mul(ctx0,
                                    ggml_repeat(ctx0, model.layers[il].ln_out_w, cur),
                                    cur),
                           ggml_repeat(ctx0, model.layers[il].ln_out_b, cur));
        }
        inpL = cur;
    }
    inpL = ggml_cont(ctx0, ggml_transpose(ctx0, inpL));
    // pooler
    struct ggml_tensor *sum = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, N, 1);
    ggml_set_f32(sum, 1.0f / N);
    inpL = ggml_mul_mat(ctx0, inpL, sum);

    // normalizer
    ggml_tensor *length = ggml_sqrt(ctx0,
                                    ggml_sum(ctx0, ggml_sqr(ctx0, inpL)));
    inpL = ggml_scale(ctx0, inpL, ggml_div(ctx0, ggml_new_f32(ctx0, 1.0f), length));

    ggml_tensor *output = inpL;
    // run the computation
    ggml_build_forward_expand(&gf, output);
    ggml_graph_compute(ctx0, &gf);

    // float *dat = ggml_get_data_f32(output);
    // pretty_print_tensor(dat, output->ne, output->nb, output->n_dims - 1, "");

    // if (n_past%100 == 0) {
    //     ggml_graph_print   (&gf);
    //     ggml_graph_dump_dot(&gf, NULL, "gpt-2.dot");
    // }

    result.resize(n_embd);
    memcpy(result.data(), (float *)ggml_get_data(output), sizeof(float) * n_embd);

    if (mem_per_token == 0)
    {
        mem_per_token = ggml_used_mem(ctx0) / N;
    }
    // printf("used_mem = %zu\n", ggml_used_mem(ctx0));

    ggml_free(ctx0);

    return result;
}

int bert_n_embd(const bert_model &model)
{
    return model.hparams.n_embd;
}
