// test_index.c -- exercise the semantic index without a bot attached.
//
// The risky code in this feature is not the tool wiring, it is parsing 768
// floats out of a response, normalising them, and ranking them. This calls
// those three directly against the real embedding endpoint and checks that
// the ranking is the one a person would give.
#define main agent_main
#include "bmagent.c"
#undef main

static const char *FACTS[] = {
    "Reza's unikernel boots in 31 milliseconds under Firecracker",
    "The cat is called Marmalade and she is fourteen years old",
    "BareMetal Cloud caps every instance at 16 MiB of RAM",
    "Reza prefers his terminal text in amber on a dark background",
    "The dentist appointment is on Tuesday at half past nine",
};
#define NFACTS ((int)(sizeof(FACTS) / sizeof(FACTS[0])))

static const struct { const char *q; int expect; } QUERIES[] = {
    { "how quickly does the machine start?",        0 },
    { "what pet do I have?",                        1 },
    { "how much memory can I get?",                 2 },
    { "what colour scheme do I like?",              3 },
    { "when am I seeing the dentist?",              4 },
};
#define NQ ((int)(sizeof(QUERIES) / sizeof(QUERIES[0])))

int main(void) {
    cJSON_Hooks hooks = { .malloc_fn = arena_alloc, .free_fn = arena_free };
    cJSON_InitHooks(&hooks);
    curl_global_init(CURL_GLOBAL_DEFAULT);
    g_llm_key = getenv("GEMINI_API_KEY");
    if (!g_llm_key || !*g_llm_key) { fprintf(stderr, "set GEMINI_API_KEY\n"); return 2; }
    calibrate_cycles();

    printf("index capacity %d x %d dims = %zu bytes static\n",
           VEC_MAX, VEC_DIM, sizeof(g_vec) + sizeof(g_vtext) + sizeof(g_vchat));

    // Store the facts against one chat, and a decoy against another, to check
    // that a search cannot cross from one person's notes into another's.
    static float v[EMBED_BATCH][VEC_DIM];
    for (int i = 0; i < NFACTS; i++) {
        const char *one = FACTS[i];
        if (embed_texts(&one, 1, v) != 1) { fprintf(stderr, "embed failed at %d\n", i); return 1; }
        double n = 0;
        for (int d = 0; d < VEC_DIM; d++) n += (double)v[0][d] * (double)v[0][d];
        if (n < 0.99 || n > 1.01) { fprintf(stderr, "FAIL: norm %.4f is not 1\n", n); return 1; }
        index_add(1111, FACTS[i], v[0]);
    }
    {
        const char *decoy = "Somebody else's secret: the launch code is 4815162342";
        if (embed_texts(&decoy, 1, v) == 1) index_add(2222, decoy, v[0]);
    }
    printf("%d vectors resident\n\n", g_vcount);

    int fails = 0;
    for (int i = 0; i < NQ; i++) {
        const char *q = QUERIES[i].q;
        if (embed_texts(&q, 1, v) != 1) { fprintf(stderr, "query embed failed\n"); return 1; }
        int idx[8]; double sc[8];
        int n = index_search(1111, v[0], 3, idx, sc);
        int top = n ? idx[0] : -1;
        int ok = (top >= 0 && strcmp(g_vtext[top], FACTS[QUERIES[i].expect]) == 0);
        if (!ok) fails++;
        printf("%-38s -> %-52.52s %.3f  %s\n", q, top >= 0 ? g_vtext[top] : "(nothing)",
               n ? sc[0] : 0.0, ok ? "ok" : "WRONG");
        for (int j = 0; j < n; j++)
            if (g_vchat[idx[j]] != 1111) { printf("  LEAK: returned another chat's note\n"); fails++; }
    }
    printf("\nsearch over %d vectors took %.1f us\n", g_vcount,
           g_tsc_hz ? (double)g_vlast_search_cycles * 1e6 / (double)g_tsc_hz : 0.0);
    printf("%s: %d/%d queries returned the right note\n", fails ? "FAIL" : "PASS", NQ - fails, NQ);
    return fails ? 1 : 0;
}
