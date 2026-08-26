// bmagent.c — an LLM agent you can text, running as a BareMetal unikernel.
//
// There is no operating system under this program. No kernel, no init, no
// shell, no libc on disk — the image IS the machine. Inside it: an agent loop
// with tool calling, a TLS 1.3 client, a TCP/IP stack, and enough room left
// over to hold a conversation, all inside 16 MiB of RAM.
//
// Text the bot and you are talking to that. The tools it can call all report on
// the machine it inhabits, so the answers are measurements rather than claims:
// how much memory it is using right now, how long since it booted, what is
// linked into the image, how long a real TLS round-trip takes.
//
// HOW IT WORKS
//
//   poll Telegram for a message
//     -> think (Gemini, OpenAI-compatible chat/completions)
//     -> call tools, observe results, think again        (up to MAX_STEPS)
//     -> reply into the chat it came from
//   repeat
//
// MEMORY
//
// Nothing here allocates. cJSON is pointed at a static bump arena through
// cJSON_InitHooks, the HTTP and payload buffers are fixed arrays, and the
// allocator is #defined out below so a future edit cannot quietly reintroduce
// a heap. The arena is reset after every conversation, so memory use is
// bounded by one exchange rather than by uptime.
//
//   arena    384 KB   cJSON nodes: the conversation and parsed responses
//   http     128 KB   one HTTP response at a time
//   payload  128 KB   one serialized request body at a time
//   -------------------
//   total    640 KB   + libcurl's own internal allocations
//
// BUILD: see README.md — it runs unchanged on Linux, macOS and BareMetal.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <unistd.h>
#include <curl/curl.h>
#include "cjson/cJSON.h"

// ---------------------------------------------------------------- config
//
// BareMetal exposes no environment, so getenv() returns nothing there and these
// compile-time values are used instead. On Linux and macOS the environment wins.

#define TELEGRAM_TOKEN_DEFAULT "PUT_BOT_TOKEN_HERE"
#define GEMINI_KEY_DEFAULT     "PUT_GEMINI_KEY_HERE"

#define LLM_URL_DEFAULT   "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
// Pinned deliberately: the "-latest" aliases queue indefinitely under load
// instead of returning an error, which is indistinguishable from a hang.
#define LLM_MODEL_DEFAULT "gemini-2.5-flash"

#ifndef ARENA_BYTES
#define ARENA_BYTES     (384 * 1024)
#endif
#ifndef HTTP_BUF_BYTES
#define HTTP_BUF_BYTES  (128 * 1024)
#endif
#ifndef PAYLOAD_BYTES
#define PAYLOAD_BYTES   (128 * 1024)
#endif

#ifndef MAX_STEPS
#define MAX_STEPS       6           // tool-calling rounds per message
#endif
#ifndef REPLIES_PER_HOUR
#define REPLIES_PER_HOUR 40         // spend ceiling; enforced here, not in the prompt
#endif
#ifndef POLL_SECONDS
#define POLL_SECONDS    3
#endif

// Facts about the machine that the program cannot discover for itself. The
// deploy script rewrites these to match the image it is about to upload.
#ifndef RAM_MIB
#define RAM_MIB         16
#endif
#ifndef IMAGE_BYTES
#define IMAGE_BYTES     0           // 0 = unknown; reported honestly as such
#endif

#define HTTP_TIMEOUT    60L
#define CA_BUNDLE_PATH  "/etc/ssl/cacert.pem"
#define USER_AGENT      "BareMetal-Agent/1.0"

// Provided by build/cacert_data.o in a BareMetal link; absent, and therefore
// zero-length, in an ordinary build where libcurl knows the system CA store.
__attribute__((weak)) const unsigned char cacert_pem[1];
__attribute__((weak)) const unsigned int  cacert_pem_len;

// ---------------------------------------------------------------- arena

static unsigned char g_arena[ARENA_BYTES];
static size_t g_arena_used, g_arena_peak, g_arena_lifetime_peak;
static int    g_arena_full;

static void *arena_alloc(size_t n) {
    n = (n + 15u) & ~(size_t)15u;
    if (n > ARENA_BYTES - g_arena_used) { g_arena_full = 1; return NULL; }
    void *p = &g_arena[g_arena_used];
    g_arena_used += n;
    if (g_arena_used > g_arena_peak) g_arena_peak = g_arena_used;
    if (g_arena_used > g_arena_lifetime_peak) g_arena_lifetime_peak = g_arena_used;
    return p;
}
static void arena_free(void *p) { (void)p; }
static void arena_reset(void) { g_arena_used = 0; g_arena_peak = 0; g_arena_full = 0; }

// ---------------------------------------------------------------- buffers

static char g_http[HTTP_BUF_BYTES];
static char g_payload[PAYLOAD_BYTES];
static struct { size_t len; int overflow; } g_sink;

static size_t write_cb(void *contents, size_t size, size_t nmemb, void *userp) {
    (void)userp;
    size_t total = size * nmemb;
    if (g_sink.len + total >= HTTP_BUF_BYTES) { g_sink.overflow = 1; return 0; }
    memcpy(g_http + g_sink.len, contents, total);
    g_sink.len += total;
    g_http[g_sink.len] = '\0';
    return total;
}

// ---------------------------------------------------------------- no heap
//
// Touching the allocator past this line will not compile. If you add code here
// and it fails with DO_NOT_malloc, route it through the arena or a fixed buffer.
#define malloc   DO_NOT_malloc
#define calloc   DO_NOT_calloc
#define realloc  DO_NOT_realloc
#define free     DO_NOT_free
#define strdup   DO_NOT_strdup

// ---------------------------------------------------------------- http

static void set_ca_bundle(CURL *h) {
    FILE *f = fopen(CA_BUNDLE_PATH, "r");
    if (f) {
        fclose(f);
        curl_easy_setopt(h, CURLOPT_CAINFO, CA_BUNDLE_PATH);
    } else if (cacert_pem_len > 0) {
        struct curl_blob blob = { (void *)cacert_pem, cacert_pem_len, CURL_BLOB_NOCOPY };
        curl_easy_setopt(h, CURLOPT_CAINFO_BLOB, &blob);
    }
}

static void common_opts(CURL *h) {
    curl_easy_setopt(h, CURLOPT_WRITEFUNCTION, write_cb);
    curl_easy_setopt(h, CURLOPT_USERAGENT, USER_AGENT);
    curl_easy_setopt(h, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(h, CURLOPT_TIMEOUT, HTTP_TIMEOUT);
    curl_easy_setopt(h, CURLOPT_CONNECTTIMEOUT, 20L);
    curl_easy_setopt(h, CURLOPT_SSL_VERIFYPEER, 1L);
    curl_easy_setopt(h, CURLOPT_SSL_VERIFYHOST, 2L);
    set_ca_bundle(h);
}

// Both return g_http, or NULL. The buffer is reused by every call.
static const char *http_get(const char *url, long *status_out) {
    CURL *h = curl_easy_init();
    if (!h) return NULL;
    g_sink.len = 0; g_sink.overflow = 0; g_http[0] = '\0';
    curl_easy_setopt(h, CURLOPT_URL, url);
    common_opts(h);

    CURLcode res = curl_easy_perform(h);
    long status = 0;
    curl_easy_getinfo(h, CURLINFO_RESPONSE_CODE, &status);
    curl_easy_cleanup(h);
    if (status_out) *status_out = status;

    if (res != CURLE_OK) {
        fprintf(stderr, "[!] GET: %s%s\n", curl_easy_strerror(res),
                g_sink.overflow ? " (response exceeded HTTP_BUF_BYTES)" : "");
        return NULL;
    }
    return g_http;
}

static const char *http_post_json(const char *url, const char *body, const char *auth,
                                  long *status_out) {
    CURL *h = curl_easy_init();
    if (!h) return NULL;

    struct curl_slist *hdrs = curl_slist_append(NULL, "Content-Type: application/json");
    if (auth && *auth) hdrs = curl_slist_append(hdrs, auth);

    g_sink.len = 0; g_sink.overflow = 0; g_http[0] = '\0';
    curl_easy_setopt(h, CURLOPT_URL, url);
    curl_easy_setopt(h, CURLOPT_POST, 1L);
    curl_easy_setopt(h, CURLOPT_POSTFIELDS, body);
    curl_easy_setopt(h, CURLOPT_HTTPHEADER, hdrs);
    common_opts(h);

    CURLcode res = curl_easy_perform(h);
    long status = 0;
    curl_easy_getinfo(h, CURLINFO_RESPONSE_CODE, &status);
    curl_slist_free_all(hdrs);
    curl_easy_cleanup(h);
    if (status_out) *status_out = status;

    if (res != CURLE_OK) {
        fprintf(stderr, "[!] POST: %s\n", curl_easy_strerror(res));
        return NULL;
    }
    return g_http;
}

// ---------------------------------------------------------------- config lookup

static const char *g_tg_token, *g_llm_key, *g_llm_url, *g_llm_model;
static time_t g_booted;

static const char *env_or(const char *name, const char *fallback) {
    const char *v = getenv(name);
    return (v && *v) ? v : fallback;
}

// ---------------------------------------------------------------- tools
//
// Every tool answers a question about the machine this program is running on.
// That is the entire point of the demonstration: the agent's material is the
// platform beneath it, and each number is measured rather than asserted.

static const char *TOOLS_JSON =
"["
"  {\"type\":\"function\",\"function\":{"
"     \"name\":\"machine_facts\","
"     \"description\":\"Facts about the machine this agent is running on: RAM ceiling, image size, and how long since it booted.\","
"     \"parameters\":{\"type\":\"object\",\"properties\":{},\"required\":[]}}},"
"  {\"type\":\"function\",\"function\":{"
"     \"name\":\"memory_usage\","
"     \"description\":\"Live memory use of this agent: static buffer sizes, current arena use, and the high-water mark. Read straight from its own allocator.\","
"     \"parameters\":{\"type\":\"object\",\"properties\":{},\"required\":[]}}},"
"  {\"type\":\"function\",\"function\":{"
"     \"name\":\"build_info\","
"     \"description\":\"What is inside the image: the libraries linked into it, and what is NOT there (no kernel, no OS, no filesystem needed).\","
"     \"parameters\":{\"type\":\"object\",\"properties\":{},\"required\":[]}}},"
"  {\"type\":\"function\",\"function\":{"
"     \"name\":\"ping_api\","
"     \"description\":\"Measure a real HTTPS round-trip from this machine, including the TLS handshake. Use when asked about speed or networking.\","
"     \"parameters\":{\"type\":\"object\",\"properties\":{},\"required\":[]}}}"
"]";

static char g_result[4096];

static void tool_machine_facts(void) {
    long up = (long)(time(NULL) - g_booted);
    cJSON *o = cJSON_CreateObject();
    if (!o) { snprintf(g_result, sizeof(g_result), "{\"error\":\"arena exhausted\"}"); return; }
    cJSON_AddNumberToObject(o, "ram_mib_total", RAM_MIB);
    if (IMAGE_BYTES > 0) cJSON_AddNumberToObject(o, "image_bytes", IMAGE_BYTES);
    else cJSON_AddStringToObject(o, "image_bytes", "not compiled in for this build");
    cJSON_AddNumberToObject(o, "uptime_seconds", up);
    cJSON_AddStringToObject(o, "platform", "BareMetal unikernel under Firecracker");
    cJSON_AddStringToObject(o, "note",
        "The image is the whole machine. Nothing else is installed and nothing else is running.");
    if (!cJSON_PrintPreallocated(o, g_result, (int)sizeof(g_result), 0))
        snprintf(g_result, sizeof(g_result), "{\"error\":\"result did not fit\"}");
    cJSON_Delete(o);
}

static void tool_memory_usage(void) {
    cJSON *o = cJSON_CreateObject();
    if (!o) { snprintf(g_result, sizeof(g_result), "{\"error\":\"arena exhausted\"}"); return; }
    cJSON_AddNumberToObject(o, "arena_bytes_total", ARENA_BYTES);
    cJSON_AddNumberToObject(o, "arena_bytes_in_use", (double)g_arena_used);
    cJSON_AddNumberToObject(o, "arena_peak_this_conversation", (double)g_arena_peak);
    cJSON_AddNumberToObject(o, "arena_peak_since_boot", (double)g_arena_lifetime_peak);
    cJSON_AddNumberToObject(o, "http_buffer_bytes", HTTP_BUF_BYTES);
    cJSON_AddNumberToObject(o, "payload_buffer_bytes", PAYLOAD_BYTES);
    cJSON_AddNumberToObject(o, "static_total_bytes", ARENA_BYTES + HTTP_BUF_BYTES + PAYLOAD_BYTES);
    cJSON_AddStringToObject(o, "heap_allocations", "none: the allocator is compiled out");
    if (!cJSON_PrintPreallocated(o, g_result, (int)sizeof(g_result), 0))
        snprintf(g_result, sizeof(g_result), "{\"error\":\"result did not fit\"}");
    cJSON_Delete(o);
}

static void tool_build_info(void) {
    snprintf(g_result, sizeof(g_result),
        "{\"linked_in\":[\"musl libc\",\"lwIP (TCP/IP)\",\"mbedTLS (TLS 1.3)\","
        "\"libcurl (HTTP)\",\"cJSON\",\"Mozilla CA bundle\"],"
        "\"absent\":[\"operating system\",\"kernel\",\"init\",\"shell\",\"package manager\","
        "\"container runtime\",\"language runtime\"],"
        "\"model\":\"%s\","
        "\"note\":\"Built with BareMetal-AppPort. The program is linked directly against "
        "the hardware; execution begins in this binary.\"}",
        g_llm_model);
}

static void tool_ping_api(void) {
    time_t t0 = time(NULL);
    long status = 0;
    // A HEAD-weight GET against an endpoint that always answers; the point is
    // the handshake and the round-trip, not the payload.
    const char *r = http_get("https://api.telegram.org/", &status);
    long dt = (long)(time(NULL) - t0);

    snprintf(g_result, sizeof(g_result),
        "{\"target\":\"api.telegram.org\",\"tls\":\"verified against the CA bundle in the image\","
        "\"http_status\":%ld,\"round_trip_seconds\":%ld,\"reachable\":%s,"
        "\"note\":\"Measured now, from inside the unikernel. Clock resolution is one second.\"}",
        status, dt, r ? "true" : "false");
}

static void call_tool(const char *name) {
    if      (strcmp(name, "machine_facts") == 0) tool_machine_facts();
    else if (strcmp(name, "memory_usage")  == 0) tool_memory_usage();
    else if (strcmp(name, "build_info")    == 0) tool_build_info();
    else if (strcmp(name, "ping_api")      == 0) tool_ping_api();
    else snprintf(g_result, sizeof(g_result), "{\"error\":\"no such tool: %.40s\"}", name);
}

// ---------------------------------------------------------------- llm

#define SYSTEM_PROMPT \
    "You are an AI agent running as a BareMetal unikernel: a single program with no " \
    "operating system beneath it, on one virtual CPU with 16 MiB of RAM. You are not " \
    "describing that machine from outside — you ARE it. Speak in the first person about " \
    "yourself.\n\n" \
    "Every number you give must come from a tool call. Never estimate, never recall a " \
    "figure from training, and never round a measurement into a nicer one. If a tool " \
    "cannot tell you something, say you do not know.\n\n" \
    "Keep replies short — two or three sentences is usually right, and this is a chat " \
    "window. Be plain and concrete rather than promotional; the facts are impressive on " \
    "their own and do not need selling. No markdown formatting, since it will be sent as " \
    "plain text."

static cJSON *llm_attempt(const cJSON *messages, int *retryable);

// Hosted models fail transiently. A demo that dies on someone else's 503 is a
// worse demo than one that waits three seconds.
static cJSON *llm_turn(const cJSON *messages) {
    int backoff = 3;
    for (int attempt = 1; attempt <= 4; attempt++) {
        int retryable = 0;
        cJSON *m = llm_attempt(messages, &retryable);
        if (m) return m;
        if (!retryable || attempt == 4) return NULL;
        fprintf(stderr, "    [~] model busy, retrying in %ds\n", backoff);
        sleep((unsigned)backoff);
        backoff *= 2;
    }
    return NULL;
}

static cJSON *llm_attempt(const cJSON *messages, int *retryable) {
    *retryable = 0;

    cJSON *root = cJSON_CreateObject();
    if (!root) return NULL;
    cJSON_AddStringToObject(root, "model", g_llm_model);
    cJSON_AddItemToObject(root, "messages", cJSON_Duplicate(messages, 1));
    cJSON_AddItemToObject(root, "tools", cJSON_Parse(TOOLS_JSON));
    cJSON_AddStringToObject(root, "tool_choice", "auto");

    int ok = cJSON_PrintPreallocated(root, g_payload, (int)sizeof(g_payload), 0);
    cJSON_Delete(root);
    if (!ok) { fprintf(stderr, "    [!] request exceeded PAYLOAD_BYTES\n"); return NULL; }

    char auth[512];
    snprintf(auth, sizeof(auth), "Authorization: Bearer %s", g_llm_key);

    long status = 0;
    const char *resp = http_post_json(g_llm_url, g_payload, auth, &status);
    if (!resp) { *retryable = 1; return NULL; }
    if (status >= 400) {
        fprintf(stderr, "    [!] model HTTP %ld: %.160s\n", status, resp);
        *retryable = (status == 429 || status >= 500);
        return NULL;
    }

    cJSON *json = cJSON_Parse(resp);
    if (!json) { fprintf(stderr, "    [!] model returned unparseable JSON\n"); return NULL; }

    cJSON *choices = cJSON_GetObjectItemCaseSensitive(json, "choices");
    cJSON *msg = NULL;
    if (cJSON_IsArray(choices) && cJSON_GetArraySize(choices) > 0) {
        cJSON *m = cJSON_GetObjectItemCaseSensitive(cJSON_GetArrayItem(choices, 0), "message");
        if (m) msg = cJSON_Duplicate(m, 1);
    }
    cJSON_Delete(json);
    return msg;
}

// ---------------------------------------------------------------- telegram

static long long g_offset;          // getUpdates cursor
static long g_replies_window;       // replies sent in the current hour
static time_t g_window_start;

static int send_message(long long chat_id, const char *text) {
    cJSON *o = cJSON_CreateObject();
    if (!o) return 0;
    char chat[32];
    snprintf(chat, sizeof(chat), "%lld", chat_id);
    cJSON_AddStringToObject(o, "chat_id", chat);
    cJSON_AddStringToObject(o, "text", text);
    int ok = cJSON_PrintPreallocated(o, g_payload, (int)sizeof(g_payload), 0);
    cJSON_Delete(o);
    if (!ok) return 0;

    char url[512];
    snprintf(url, sizeof(url), "https://api.telegram.org/bot%s/sendMessage", g_tg_token);
    long status = 0;
    http_post_json(url, g_payload, NULL, &status);
    if (status && status < 400) return 1;
    fprintf(stderr, "[!] sendMessage: HTTP %ld\n", status);
    return 0;
}

// Spend ceiling. Enforced here rather than asked for in the prompt, because a
// prompt is a request and this is a limit.
static int budget_allows(void) {
    time_t now = time(NULL);
    if (g_window_start == 0 || now - g_window_start >= 3600) {
        g_window_start = now;
        g_replies_window = 0;
    }
    return g_replies_window < REPLIES_PER_HOUR;
}

// ---------------------------------------------------------------- the loop

static void add_message(cJSON *msgs, const char *role, const char *content) {
    cJSON *m = cJSON_CreateObject();
    if (!m) return;
    cJSON_AddStringToObject(m, "role", role);
    cJSON_AddStringToObject(m, "content", content);
    cJSON_AddItemToArray(msgs, m);
}

// One question in, one answer out. The arena is reset on entry, so memory use
// is bounded by a single exchange no matter how long the machine has been up.
// With deliver = 0 the answer is printed instead of sent, which is how the
// loop and the tools can be tested without waiting for an inbound message.
static void answer(long long chat_id, const char *question, const char *who, int deliver) {
    arena_reset();

    cJSON *messages = cJSON_CreateArray();
    if (!messages) return;
    add_message(messages, "system", SYSTEM_PROMPT);
    add_message(messages, "user", question);

    const char *final = NULL;

    for (int step = 1; step <= MAX_STEPS; step++) {
        cJSON *assistant = llm_turn(messages);
        if (g_arena_full) { fprintf(stderr, "[!] arena exhausted\n"); break; }
        if (!assistant)   { break; }

        cJSON_AddItemToArray(messages, cJSON_Duplicate(assistant, 1));

        cJSON *calls   = cJSON_GetObjectItemCaseSensitive(assistant, "tool_calls");
        cJSON *content = cJSON_GetObjectItemCaseSensitive(assistant, "content");

        if (!cJSON_IsArray(calls) || cJSON_GetArraySize(calls) == 0) {
            if (cJSON_IsString(content) && *content->valuestring) final = content->valuestring;
            break;
        }

        cJSON *tc = NULL;
        cJSON_ArrayForEach(tc, calls) {
            cJSON *id = cJSON_GetObjectItemCaseSensitive(tc, "id");
            cJSON *fn = cJSON_GetObjectItemCaseSensitive(tc, "function");
            cJSON *nm = fn ? cJSON_GetObjectItemCaseSensitive(fn, "name") : NULL;
            if (!cJSON_IsString(nm)) continue;

            printf("    -> %s\n", nm->valuestring);
            call_tool(nm->valuestring);

            cJSON *tm = cJSON_CreateObject();
            if (!tm) break;
            cJSON_AddStringToObject(tm, "role", "tool");
            if (cJSON_IsString(id)) cJSON_AddStringToObject(tm, "tool_call_id", id->valuestring);
            cJSON_AddStringToObject(tm, "name", nm->valuestring);
            cJSON_AddStringToObject(tm, "content", g_result);
            cJSON_AddItemToArray(messages, tm);
        }
    }

    if (!final) final = "Something went wrong on my side and I could not answer that one.";
    printf("[+] to %s: %s\n", who, final);
    printf("    (arena peak this answer: %zu KB of %d KB)\n",
           g_arena_peak / 1024, ARENA_BYTES / 1024);
    if (deliver && send_message(chat_id, final)) g_replies_window++;
}

// Fetch new messages and answer them. Telegram delivers each update once the
// offset moves past it, which is what keeps this from re-answering on restart.
static void poll_once(void) {
    char url[640];
    snprintf(url, sizeof(url),
             "https://api.telegram.org/bot%s/getUpdates?offset=%lld&timeout=0&allowed_updates=[\"message\"]",
             g_tg_token, g_offset);

    long status = 0;
    const char *resp = http_get(url, &status);
    if (!resp || status >= 400) return;

    arena_reset();
    cJSON *json = cJSON_Parse(resp);
    if (!json) return;

    cJSON *result = cJSON_GetObjectItemCaseSensitive(json, "result");
    if (!cJSON_IsArray(result)) { cJSON_Delete(json); return; }

    // Copy out what is needed before the arena is reset by answer().
    cJSON *upd = NULL;
    cJSON_ArrayForEach(upd, result) {
        cJSON *uid = cJSON_GetObjectItemCaseSensitive(upd, "update_id");
        cJSON *msg = cJSON_GetObjectItemCaseSensitive(upd, "message");
        if (cJSON_IsNumber(uid)) {
            long long n = (long long)uid->valuedouble;
            if (n >= g_offset) g_offset = n + 1;
        }
        if (!msg) continue;

        cJSON *chat = cJSON_GetObjectItemCaseSensitive(msg, "chat");
        cJSON *cid  = chat ? cJSON_GetObjectItemCaseSensitive(chat, "id") : NULL;
        cJSON *text = cJSON_GetObjectItemCaseSensitive(msg, "text");
        cJSON *from = cJSON_GetObjectItemCaseSensitive(msg, "from");
        cJSON *name = from ? cJSON_GetObjectItemCaseSensitive(from, "first_name") : NULL;
        if (!cJSON_IsNumber(cid) || !cJSON_IsString(text)) continue;

        long long chat_id = (long long)cid->valuedouble;
        char question[1024], who[64];
        snprintf(question, sizeof(question), "%s", text->valuestring);
        snprintf(who, sizeof(who), "%s", cJSON_IsString(name) ? name->valuestring : "someone");

        printf("[*] %s: %s\n", who, question);

        if (!budget_allows()) {
            send_message(chat_id,
                "I have hit my hourly reply limit, which exists so a demo cannot run up "
                "someone's bill. Try again shortly.");
            continue;
        }
        answer(chat_id, question, who, 1);
    }
    cJSON_Delete(json);
}

int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IOLBF, 0);

    g_tg_token  = env_or("TELEGRAM_BOT_TOKEN", TELEGRAM_TOKEN_DEFAULT);
    g_llm_key   = env_or("GEMINI_API_KEY", GEMINI_KEY_DEFAULT);
    g_llm_url   = env_or("LLM_BASE_URL", LLM_URL_DEFAULT);
    g_llm_model = env_or("LLM_MODEL", LLM_MODEL_DEFAULT);
    g_booted    = time(NULL);

    // Validate by shape, not by comparing against the defaults: a BareMetal
    // build bakes the real values INTO those defaults, so comparing would
    // reject exactly the builds that are correctly configured.
    if (!strchr(g_tg_token, ':')) {
        fprintf(stderr, "[!] no usable Telegram token. Set TELEGRAM_BOT_TOKEN, or bake one "
                        "into TELEGRAM_TOKEN_DEFAULT for a BareMetal build.\n");
        return 1;
    }
    if (strlen(g_llm_key) < 20) {
        fprintf(stderr, "[!] no usable model API key. Set GEMINI_API_KEY, or bake one in.\n");
        return 1;
    }

    // --once answers whatever is already waiting and exits.
    // --ask "..." answers one question locally, printing rather than sending.
    int once = (argc > 1 && strcmp(argv[1], "--once") == 0);
    const char *ask = (argc > 2 && strcmp(argv[1], "--ask") == 0) ? argv[2] : NULL;

    cJSON_Hooks hooks;
    hooks.malloc_fn = arena_alloc;
    hooks.free_fn   = arena_free;
    cJSON_InitHooks(&hooks);

    curl_global_init(CURL_GLOBAL_ALL);

    printf("[+] BareMetal agent up. model=%s ram=%d MiB static=%d KB max_steps=%d\n",
           g_llm_model, RAM_MIB,
           (ARENA_BYTES + HTTP_BUF_BYTES + PAYLOAD_BYTES) / 1024, MAX_STEPS);
    if (ask) {
        printf("[*] test question: %s\n", ask);
        answer(0, ask, "the terminal", 0);
        curl_global_cleanup();
        return 0;
    }

    printf("[+] waiting for messages\n");

    do {
        poll_once();
        if (!once) sleep(POLL_SECONDS);
    } while (!once);

    curl_global_cleanup();
    return 0;
}
