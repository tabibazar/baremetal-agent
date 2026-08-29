// Call recall_similar directly, in a fresh process, against the live Redis.
// The index starts empty -- exactly the state after a restart -- so anything
// found had to be re-embedded from Redis by the backfill path first.
#define main agent_main
#include "bmagent.c"
#undef main

int main(int argc, char **argv) {
    cJSON_Hooks h = { .malloc_fn = arena_alloc, .free_fn = arena_free };
    cJSON_InitHooks(&h);
    curl_global_init(CURL_GLOBAL_DEFAULT);
    g_llm_key  = getenv("GEMINI_API_KEY");
    g_kv_url   = getenv("KV_URL");
    g_kv_token = getenv("KV_TOKEN");
    g_memory_enabled = 1;
    g_current_chat = strtoll(argv[1], NULL, 10);
    calibrate_cycles();

    printf("index starts with %d vectors (fresh process)\n\n", g_vcount);
    for (int i = 2; i < argc; i++) {
        cJSON *a = cJSON_CreateObject();
        cJSON_AddStringToObject(a, "query", argv[i]);
        cJSON_AddNumberToObject(a, "k", 2);
        time_t t0 = time(NULL);
        tool_recall_similar(a);
        printf("Q: %s\n   (%lds) %s\n\n", argv[i], (long)(time(NULL) - t0), g_result);
    }
    return 0;
}
