#ifndef TOKENIZER_H
#define TOKENIZER_H

#include "model.h"
#include <stdint.h>

typedef struct {
    char  **vocab;       /* vocab[i] = string for token i */
    float  *scores;      /* BPE merge scores */
    int     vocab_size;
    int    *sorted_idx;  /* indices sorted by vocab string for binary search */
    uint32_t bos_id;
    uint32_t eos_id;
    int     gpt_style;    /* 1 = GPT2/Qwen byte-level BPE, 0 = SentencePiece */
    int     use_rank;     /* use merge-rank table instead of scores */
    uint64_t *pair_keys;  /* hash: (a+1)<<32 | (b+1) */
    uint32_t *pair_ranks; /* merge rank */
    uint32_t *pair_result;/* merged token id */
    uint32_t pair_mask;   /* table size - 1 */
    int    *ctrl_ids;     /* added/control token ids for literal matching */
    char  **ctrl_strs;    /* their vocab strings (borrowed) */
    int     n_ctrl;
} tokenizer_t;

/* Load tokenizer data from GGUF metadata pointers in model.
 * Returns 0 on success. */
int tokenizer_load(tokenizer_t *t, const model_t *m);

/* Encode a text string into token IDs.
 * tokens must have space for at least max_tokens entries.
 * Returns number of tokens produced. */
int tokenizer_encode(const tokenizer_t *t, const char *text, int *tokens, int max_tokens, int add_bos);

/* Decode a single token ID to its string representation.
 * Returns pointer to static/vocab string (do not free). */
const char *tokenizer_decode(const tokenizer_t *t, int prev_token, int token);

/* Free tokenizer resources */
void tokenizer_free(tokenizer_t *t);

#endif /* TOKENIZER_H */
