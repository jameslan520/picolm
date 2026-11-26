#include "tokenizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>

/* ---- GGUF string reader (reused from model.c logic) ---- */

static uint64_t read_u64_at(const uint8_t **p) {
    uint64_t v;
    memcpy(&v, *p, 8);
    *p += 8;
    return v;
}

/* ---- Sorted index for binary search ---- */

static char **g_vocab_for_sort; /* global for qsort comparison */

static int cmp_sorted(const void *a, const void *b) {
    int ia = *(const int *)a;
    int ib = *(const int *)b;
    return strcmp(g_vocab_for_sort[ia], g_vocab_for_sort[ib]);
}

static int vocab_lookup(const tokenizer_t *t, const char *str, int len) {
    /* Binary search in sorted vocabulary */
    int lo = 0, hi = t->vocab_size - 1;
    while (lo <= hi) {
        int mid = (lo + hi) / 2;
        int idx = t->sorted_idx[mid];
        int cmp = strncmp(t->vocab[idx], str, (size_t)len);
        if (cmp == 0) {
            /* Check exact length match */
            if (t->vocab[idx][len] == '\0') return idx;
            if (t->vocab[idx][len] > '\0') { hi = mid - 1; }
            else { lo = mid + 1; }
        } else if (cmp < 0) {
            lo = mid + 1;
        } else {
            hi = mid - 1;
        }
    }
    return -1; /* not found */
}

/* ---- GPT2-style byte rendering (bytes_to_unicode) ----
 *
 * Printable bytes (33-126, 161-172, 174-255) map to themselves.
 * All other bytes map to codepoints 256+n (n = index among the
 * non-printable bytes).  In particular byte 0x20 (space) -> 288
 * (U+0120 "G-dot") and 0x0A -> 266 (U+010A "C-dot"). */

static int g_np_init = 0;
static uint8_t  g_np[68];       /* non-printable bytes in ascending order */
static int16_t  g_np_idx[256];  /* byte -> index in g_np, -1 if printable */

static void np_init(void) {
    if (g_np_init) return;
    int n = 0;
    for (int b = 0; b < 256; b++) {
        int printable = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174);
        if (printable) {
            g_np_idx[b] = -1;
        } else {
            g_np_idx[b] = (int16_t)n;
            g_np[n++] = (uint8_t)b;
        }
    }
    g_np_init = 1;
}

/* Render one input byte into its GPT2 vocab form (UTF-8).
 * Returns number of bytes written (1 or 2). */
static int render_byte(unsigned char b, char *out) {
    np_init();
    if (g_np_idx[b] < 0) { out[0] = (char)b; return 1; }
    int cp = 256 + g_np_idx[b];
    out[0] = (char)(0xC0 | (cp >> 6));
    out[1] = (char)(0x80 | (cp & 0x3F));
    return 2;
}

/* ---- BPE pair rank hash table ---- */

static int pair_lookup(const tokenizer_t *t, int a, int b, int *rank, int *result) {
    if (!t->pair_keys) return 0;
    uint64_t key = ((uint64_t)(a + 1) << 32) | (uint32_t)(b + 1);
    uint32_t idx = (uint32_t)((key * 0x9E3779B97F4A7C15ULL) >> 32) & t->pair_mask;
    while (t->pair_keys[idx]) {
        if (t->pair_keys[idx] == key) {
            if (rank)   *rank   = (int)t->pair_ranks[idx];
            if (result) *result = (int)t->pair_result[idx];
            return 1;
        }
        idx = (idx + 1) & t->pair_mask;
    }
    return 0;
}

/* ---- Public API ---- */

int tokenizer_load(tokenizer_t *t, const model_t *m) {
    int vs = m->config.vocab_size;
    t->vocab_size = vs;
    t->bos_id = m->tok_bos_id;
    t->eos_id = m->tok_eos_id;
    t->gpt_style = 0;
    t->use_rank = 0;
    t->pair_keys = NULL;
    t->pair_ranks = NULL;
    t->pair_result = NULL;
    t->pair_mask = 0;
    t->ctrl_ids = NULL;
    t->ctrl_strs = NULL;
    t->n_ctrl = 0;

    np_init();

    /* Allocate vocab and scores arrays */
    t->vocab = (char **)calloc((size_t)vs, sizeof(char *));
    t->scores = (float *)calloc((size_t)vs, sizeof(float));
    t->sorted_idx = (int *)malloc((size_t)vs * sizeof(int));
    if (!t->vocab || !t->scores || !t->sorted_idx) {
        fprintf(stderr, "OOM allocating tokenizer\n");
        return -1;
    }

    /* Read vocab strings from GGUF metadata array */
    if (m->tok_tokens_data && m->tok_n_tokens > 0) {
        const uint8_t *p = (const uint8_t *)m->tok_tokens_data;
        uint64_t n = m->tok_n_tokens;
        if ((int)n > vs) n = (uint64_t)vs;

        for (uint64_t i = 0; i < n; i++) {
            uint64_t slen = read_u64_at(&p);
            /* Allocate and copy the string with null terminator */
            t->vocab[i] = (char *)malloc((size_t)slen + 1);
            if (t->vocab[i]) {
                memcpy(t->vocab[i], p, (size_t)slen);
                t->vocab[i][slen] = '\0';
            }
            p += slen;
        }
    }

    /* Fill any remaining entries with empty strings */
    for (int i = 0; i < vs; i++) {
        if (!t->vocab[i]) {
            t->vocab[i] = (char *)calloc(1, 1);
        }
    }

    /* Read scores */
    if (m->tok_scores_data && m->tok_n_scores > 0) {
        uint64_t n = m->tok_n_scores;
        if ((int)n > vs) n = (uint64_t)vs;
        memcpy(t->scores, m->tok_scores_data, (size_t)n * sizeof(float));
    }

    /* Build sorted index */
    for (int i = 0; i < vs; i++) {
        t->sorted_idx[i] = i;
    }
    g_vocab_for_sort = t->vocab;
    qsort(t->sorted_idx, (size_t)vs, sizeof(int), cmp_sorted);

    /* Detect tokenizer style: GPT2-family (byte-level, "G-dot" marker)
     * vs SentencePiece (underscore marker). */
    if (m->tok_pre_data && m->tok_pre_len > 0) {
        static const char *gpt_pres[] = {
            "gpt2", "qwen2", "jina", "refact", "starcoder", "starcoder2",
            "redpajama", "penrose", "beta", NULL
        };
        for (int i = 0; gpt_pres[i]; i++) {
            size_t gl = strlen(gpt_pres[i]);
            if (m->tok_pre_len == gl &&
                memcmp(m->tok_pre_data, gpt_pres[i], gl) == 0) {
                t->gpt_style = 1;
                break;
            }
        }
        /* "llama", "default", ... stay SentencePiece (0) */
    } else {
        /* No pre-tokenizer info: scan vocab for marker prefixes */
        int ng = 0, ns = 0;
        int scan = vs < 4096 ? vs : 4096;
        for (int i = 0; i < scan; i++) {
            const unsigned char *v = (const unsigned char *)t->vocab[i];
            if (v[0] == 0xC4 && v[1] == 0xA0) ng++;          /* G-dot */
            else if (v[0] == 0xE2 && v[1] == 0x96 && v[2] == 0x81) ns++; /* U+2581 */
        }
        t->gpt_style = (ng > ns);
    }

    /* Build pair-rank hash table from the merges list (used when merge
     * scores are absent, e.g. Qwen2/GPT2 tokenizers). */
    if (m->tok_merges_data && m->tok_n_merges > 0) {
        size_t n = (size_t)m->tok_n_merges;
        size_t sz = 256;
        while (sz < n * 2) sz <<= 1;
        t->pair_keys   = (uint64_t *)calloc(sz, sizeof(uint64_t));
        t->pair_ranks  = (uint32_t *)calloc(sz, sizeof(uint32_t));
        t->pair_result = (uint32_t *)calloc(sz, sizeof(uint32_t));
        if (t->pair_keys && t->pair_ranks && t->pair_result) {
            t->pair_mask = (uint32_t)(sz - 1);
            const uint8_t *p = (const uint8_t *)m->tok_merges_data;
            for (uint32_t rank = 0; rank < (uint32_t)n; rank++) {
                uint64_t slen;
                memcpy(&slen, p, 8);
                p += 8;
                const char *s = (const char *)p;
                p += slen;
                const char *sp = (const char *)memchr(s, ' ', (size_t)slen);
                if (!sp) continue;
                int la = (int)(sp - s);
                int lb = (int)(slen - (uint64_t)la - 1);
                if (la <= 0 || lb <= 0 || la + lb >= 500) continue;
                int a = vocab_lookup(t, s, la);
                int b = vocab_lookup(t, sp + 1, lb);
                if (a < 0 || b < 0) continue;
                char merged[512];
                memcpy(merged, s, (size_t)la);
                memcpy(merged + la, sp + 1, (size_t)lb);
                merged[la + lb] = '\0';
                int res = vocab_lookup(t, merged, la + lb);
                if (res < 0) continue;
                uint64_t key = ((uint64_t)(a + 1) << 32) | (uint32_t)(b + 1);
                uint32_t idx = (uint32_t)((key * 0x9E3779B97F4A7C15ULL) >> 32) & t->pair_mask;
                while (t->pair_keys[idx]) {
                    if (t->pair_keys[idx] == key) break; /* keep first rank */
                    idx = (idx + 1) & t->pair_mask;
                }
                if (t->pair_keys[idx]) continue; /* duplicate merge */
                t->pair_keys[idx] = key;
                t->pair_ranks[idx] = rank;
                t->pair_result[idx] = (uint32_t)res;
            }
            int has_scores = (m->tok_scores_data && m->tok_n_scores > 0);
            /* GPT-style tokenizers have no scores: always use ranks.
             * SentencePiece uses scores when present, ranks otherwise. */
            t->use_rank = t->gpt_style || !has_scores;
        }
    }

    /* Collect added/control tokens for literal matching in encode */
    if (m->tok_types_data && m->tok_n_types > 0) {
        const uint32_t *tt = (const uint32_t *)m->tok_types_data;
        uint64_t ntn = m->tok_n_types;
        if (ntn > (uint64_t)vs) ntn = (uint64_t)vs;
        t->ctrl_ids = (int *)malloc((size_t)ntn * sizeof(int));
        t->ctrl_strs = (char **)malloc((size_t)ntn * sizeof(char *));
        if (t->ctrl_ids && t->ctrl_strs) {
            for (uint64_t i = 0; i < ntn; i++) {
                uint32_t ty = tt[i];
                if (ty == 2 || ty == 3 || ty == 4) { /* CONTROL, SINGLE_WORD, UNUSED */
                    t->ctrl_ids[t->n_ctrl] = (int)i;
                    t->ctrl_strs[t->n_ctrl] = t->vocab[i];
                    t->n_ctrl++;
                }
            }
        } else {
            free(t->ctrl_ids); free(t->ctrl_strs);
            t->ctrl_ids = NULL; t->ctrl_strs = NULL;
        }
        if (ntn >= 151646 && getenv("PICOLM_DEBUG_TOKENS"))
            fprintf(stderr, "dbg: token_type[151643..45]=%u/%u/%u ctrl_tokens=%d\n",
                    tt[151643], tt[151644], tt[151645], t->n_ctrl);
    }
    fprintf(stderr, "Tokenizer loaded: %d tokens, bos=%u, eos=%u, style=%s, merges=%s\n",
            vs, t->bos_id, t->eos_id,
            t->gpt_style ? "gpt2" : "sentencepiece",
            t->pair_keys ? "rank-table" : "none");
    return 0;
}

int tokenizer_encode(const tokenizer_t *t, const char *text, int *tokens, int max_tokens, int add_bos) {
    int n_tokens = 0;

    if (add_bos && n_tokens < max_tokens && !t->gpt_style) {
        tokens[n_tokens++] = (int)t->bos_id;
    }

    if (!text || !*text) return n_tokens;

    int *merge_buf = NULL;
    int merge_len = 0;

    if (t->gpt_style) {
        /* GPT2/Qwen byte-level BPE: render each input byte to its vocab
         * form (space -> G-dot etc.), look up initial symbols, then merge. */
        int text_len = (int)strlen(text);
        merge_buf = (int *)malloc((size_t)text_len * sizeof(int));
        if (!merge_buf) return n_tokens;
        int gi = 0;
        while (gi < text_len) {
            /* Literal match for added/control tokens first */
            int m_id = -1, m_len = 0;
            for (int ci = 0; ci < t->n_ctrl; ci++) {
                int cl = (int)strlen(t->ctrl_strs[ci]);
                if (cl > m_len && gi + cl <= text_len &&
                    memcmp(text + gi, t->ctrl_strs[ci], (size_t)cl) == 0) {
                    m_id = t->ctrl_ids[ci];
                    m_len = cl;
                }
            }
            if (m_id >= 0) {
                merge_buf[merge_len++] = m_id;
                gi += m_len;
                continue;
            }
            char rb[8];
            int rl = render_byte((unsigned char)text[gi], rb);
            int tok = vocab_lookup(t, rb, rl);
            if (tok < 0 && rl > 1) {
                char raw[2] = { text[gi], 0 };
                tok = vocab_lookup(t, raw, 1);
            }
            if (tok >= 0) merge_buf[merge_len++] = tok;
            gi++;
        }
    } else {
        /* SentencePiece convention: replace spaces with U+2581 and
         * prepend U+2581 at the start of the text. */
        int text_len = (int)strlen(text);
        int norm_cap = text_len * 3 + 4;
        char *norm = (char *)malloc((size_t)norm_cap);
        if (!norm) return n_tokens;
        int norm_len = 0;

        norm[norm_len++] = (char)0xE2;
        norm[norm_len++] = (char)0x96;
        norm[norm_len++] = (char)0x81;

        for (int i = 0; i < text_len; i++) {
            if (text[i] == ' ') {
                norm[norm_len++] = (char)0xE2;
                norm[norm_len++] = (char)0x96;
                norm[norm_len++] = (char)0x81;
            } else {
                norm[norm_len++] = text[i];
            }
        }
        norm[norm_len] = '\0';

        merge_buf = (int *)malloc((size_t)(norm_len + 1) * sizeof(int));
        if (!merge_buf) { free(norm); return n_tokens; }

        for (int i = 0; i < norm_len; ) {
            int clen = 1;
            unsigned char c = (unsigned char)norm[i];
            if (c >= 0xF0) clen = 4;
            else if (c >= 0xE0) clen = 3;
            else if (c >= 0xC0) clen = 2;

            if (i + clen > norm_len) clen = norm_len - i;

            int tok = vocab_lookup(t, norm + i, clen);
            if (tok >= 0) {
                merge_buf[merge_len++] = tok;
                i += clen;
            } else {
                char byte_tok[8];
                snprintf(byte_tok, sizeof(byte_tok), "<0x%02X>", (unsigned char)norm[i]);
                tok = vocab_lookup(t, byte_tok, (int)strlen(byte_tok));
                if (tok >= 0) {
                    merge_buf[merge_len++] = tok;
                }
                i++;
            }
        }
        free(norm);
    }

    /* BPE merge loop */
    if (t->use_rank && t->pair_keys) {
        /* Rank-based merging (GPT2/Qwen, or SentencePiece without scores) */
        while (merge_len >= 2) {
            int best_rank = -1, best_idx = -1, best_res = -1;
            for (int i = 0; i < merge_len - 1; i++) {
                int rk, res;
                if (pair_lookup(t, merge_buf[i], merge_buf[i + 1], &rk, &res)) {
                    if (best_idx < 0 || rk < best_rank) {
                        best_rank = rk;
                        best_idx = i;
                        best_res = res;
                    }
                }
            }
            if (best_idx < 0) break; /* no more merges possible */
            merge_buf[best_idx] = best_res;
            for (int i = best_idx + 1; i < merge_len - 1; i++) {
                merge_buf[i] = merge_buf[i + 1];
            }
            merge_len--;
        }
    } else {
        /* Score-based merging (SentencePiece) */
        while (merge_len >= 2) {
            float best_score = -1e30f;
            int best_idx = -1;
            int best_tok = -1;

            for (int i = 0; i < merge_len - 1; i++) {
                const char *s1 = t->vocab[merge_buf[i]];
                const char *s2 = t->vocab[merge_buf[i + 1]];
                int l1 = (int)strlen(s1);
                int l2 = (int)strlen(s2);

                char merged[256];
                if (l1 + l2 >= (int)sizeof(merged)) continue;
                memcpy(merged, s1, (size_t)l1);
                memcpy(merged + l1, s2, (size_t)l2);
                merged[l1 + l2] = '\0';

                int tok = vocab_lookup(t, merged, l1 + l2);
                if (tok >= 0 && t->scores[tok] > best_score) {
                    best_score = t->scores[tok];
                    best_idx = i;
                    best_tok = tok;
                }
            }

            if (best_idx < 0) break; /* no more merges possible */

            merge_buf[best_idx] = best_tok;
            for (int i = best_idx + 1; i < merge_len - 1; i++) {
                merge_buf[i] = merge_buf[i + 1];
            }
            merge_len--;
        }
    }

    /* Copy to output */
    for (int i = 0; i < merge_len && n_tokens < max_tokens; i++) {
        tokens[n_tokens++] = merge_buf[i];
    }

    free(merge_buf);
    return n_tokens;
}

const char *tokenizer_decode(const tokenizer_t *t, int prev_token, int token) {
    if (token < 0 || token >= t->vocab_size) return "";

    const char *str = t->vocab[token];

    /* Handle byte tokens: <0xHH> -> single byte */
    if (str[0] == '<' && str[1] == '0' && str[2] == 'x' && str[5] == '>') {
        static char byte_buf[2];
        unsigned int val = 0;
        for (int i = 3; i < 5; i++) {
            val <<= 4;
            char c = str[i];
            if (c >= '0' && c <= '9') val += (unsigned)(c - '0');
            else if (c >= 'A' && c <= 'F') val += (unsigned)(c - 'A' + 10);
            else if (c >= 'a' && c <= 'f') val += (unsigned)(c - 'a' + 10);
        }
        byte_buf[0] = (char)val;
        byte_buf[1] = '\0';
        return byte_buf;
    }

    /* GPT2-style: walk UTF-8 codepoints and map back to raw bytes.
     * Codepoints 256..323 are the rendered non-printable bytes
     * (space, newline, tab, ...); codepoints < 256 are literal bytes;
     * anything larger is real Unicode text (CJK etc.) passed through. */
    if (t->gpt_style) {
        static char gbuf[512];
        const unsigned char *s = (const unsigned char *)str;
        int o = 0, i = 0;
        np_init();
        while (s[i] && o < 511) {
            int start = i;
            uint32_t cp;
            int cl;
            unsigned char c0 = s[i];
            if (c0 < 0x80) {
                cp = c0; cl = 1;
            } else if ((c0 & 0xE0) == 0xC0 && (s[i + 1] & 0xC0) == 0x80) {
                cp = ((uint32_t)(c0 & 0x1F) << 6) | (s[i + 1] & 0x3F);
                cl = 2;
            } else if ((c0 & 0xF0) == 0xE0 && (s[i + 1] & 0xC0) == 0x80 &&
                       (s[i + 2] & 0xC0) == 0x80) {
                cp = ((uint32_t)(c0 & 0x0F) << 12) |
                     ((uint32_t)(s[i + 1] & 0x3F) << 6) |
                     (uint32_t)(s[i + 2] & 0x3F);
                cl = 3;
            } else {
                cp = c0; cl = 1; /* invalid UTF-8: pass through */
            }
            i += cl;
            if (cp >= 256 && cp <= 323) {
                gbuf[o++] = (char)g_np[cp - 256];
            } else if (cp < 256) {
                gbuf[o++] = (char)(unsigned char)cp;
            } else {
                /* Real Unicode text: copy the original UTF-8 bytes */
                for (int k = start; k < i && o < 511; k++) gbuf[o++] = (char)s[k];
            }
        }
        gbuf[o] = '\0';
        return gbuf;
    }

    /* Handle SentencePiece leading space marker U+2581 -> " " */
    if ((unsigned char)str[0] == 0xE2 && (unsigned char)str[1] == 0x96 &&
        (unsigned char)str[2] == 0x81) {
        static char space_buf[256];
        if (prev_token == (int)t->bos_id) {
            int len = (int)strlen(str + 3);
            if (len >= (int)sizeof(space_buf)) len = (int)sizeof(space_buf) - 1;
            memcpy(space_buf, str + 3, (size_t)len);
            space_buf[len] = '\0';
            return space_buf;
        }
        space_buf[0] = ' ';
        int len = (int)strlen(str + 3);
        if (len >= (int)sizeof(space_buf) - 1) len = (int)sizeof(space_buf) - 2;
        memcpy(space_buf + 1, str + 3, (size_t)len);
        space_buf[1 + len] = '\0';
        return space_buf;
    }

    return str;
}

void tokenizer_free(tokenizer_t *t) {
    if (t->vocab) {
        for (int i = 0; i < t->vocab_size; i++) {
            free(t->vocab[i]);
        }
        free(t->vocab);
        t->vocab = NULL;
    }
    free(t->scores);
    t->scores = NULL;
    free(t->sorted_idx);
    t->sorted_idx = NULL;
    free(t->pair_keys);
    t->pair_keys = NULL;
    free(t->pair_ranks);
    t->pair_ranks = NULL;
    free(t->pair_result);
    t->pair_result = NULL;
    free(t->ctrl_ids);
    t->ctrl_ids = NULL;
    free(t->ctrl_strs);
    t->ctrl_strs = NULL;
}
