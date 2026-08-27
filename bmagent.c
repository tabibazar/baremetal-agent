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
#define FIRECRAWL_KEY_DEFAULT  "PUT_FIRECRAWL_KEY_HERE"   // optional: enables web search

// Where this machine keeps its memory. BareMetal Cloud instances have no
// writable filesystem -- open(O_CREAT) fails with ENOENT, verified -- so notes
// cannot live locally. They live a network round-trip away instead, behind the
// same HTTPS stack everything else uses. Any Redis with an Upstash-shaped REST
// interface works; the URL and token are the only things that change.
#define KV_URL_DEFAULT         "PUT_KV_URL_HERE"
#define KV_TOKEN_DEFAULT       "PUT_KV_TOKEN_HERE"
#define NOTES_KEPT             20        // per person, oldest dropped

// Firecrawl returns pages as markdown, so this program never parses HTML —
// which is the only reason reading the web is tractable inside a fixed buffer.
#define SEARCH_URL "https://api.firecrawl.dev/v2/search"
#define SCRAPE_URL "https://api.firecrawl.dev/v2/scrape"

#define LLM_URL_DEFAULT   "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
// Pinned deliberately: the "-latest" aliases queue indefinitely under load
// instead of returning an error, which is indistinguishable from a hang.
#define LLM_MODEL_DEFAULT "gemini-2.5-flash"

#ifndef ARENA_BYTES
#define ARENA_BYTES     (384 * 1024)
#endif
#ifndef HTTP_BUF_BYTES
// 256 KB rather than 128: Firecrawl returns a whole page as markdown, and a
// long page's JSON overflows a 128 KB buffer. Overflow is handled (the fetch
// fails cleanly) but it would fail often, and the extra 128 KB is nothing
// against 16 MiB.
#define HTTP_BUF_BYTES  (256 * 1024)
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

// Defined with the startup-timing code below; the HTTP helpers call it so that
// the first completed request is stamped wherever it happens to occur.
static void mark_first_tls(void);

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
    mark_first_tls();
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
    mark_first_tls();
    return g_http;
}

// ---------------------------------------------------------------- config lookup

static const char *g_tg_token, *g_llm_key, *g_llm_url, *g_llm_model, *g_fc_key;
static const char *g_kv_url, *g_kv_token;
static int g_web_enabled;          // false when no Firecrawl key was supplied
static int g_memory_enabled;       // false when no KV credentials were supplied
static long long g_current_chat;   // whose notes the memory tools may touch
static time_t g_booted;

// ---------------------------------------------------------------- startup timing
//
// time() here has one-second resolution, which is useless for a cold start. The
// cycle counter has the resolution; what it lacks is a unit, so its frequency is
// calibrated once against a one-second sleep AFTER the interesting interval has
// already been recorded. Measure first, learn the scale later, convert at the
// end — calibrating up front would put a second of sleep inside the very thing
// being measured.

static uint64_t g_tsc_entry, g_tsc_first_tls, g_tsc_hz;
static int      g_first_tls_seen;

static uint64_t cycles(void) {
#if defined(__x86_64__) || defined(__i386__)
    return __builtin_ia32_rdtsc();
#else
    // Non-x86 (a macOS arm64 build, say): fall back to seconds. The tool reports
    // the resolution it had, so a coarse number is never passed off as a fine one.
    return (uint64_t)time(NULL);
#endif
}

static int cycles_are_fine_grained(void) {
#if defined(__x86_64__) || defined(__i386__)
    return 1;
#else
    return 0;
#endif
}

static void mark_first_tls(void) {
    if (!g_first_tls_seen) {
        g_tsc_first_tls = cycles();
        g_first_tls_seen = 1;
    }
}

static void calibrate_cycles(void) {
    if (!cycles_are_fine_grained()) { g_tsc_hz = 1; return; }
    uint64_t a = cycles();
    sleep(1);
    uint64_t b = cycles();
    g_tsc_hz = (b > a) ? (b - a) : 0;
}

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
"     \"parameters\":{\"type\":\"object\",\"properties\":{},\"required\":[]}}},"
"  {\"type\":\"function\",\"function\":{"
"     \"name\":\"remember\","
"     \"description\":\"Keep one fact about the person you are talking to, so you still have it after you are restarted. Use it when they tell you something worth keeping.\","
"     \"parameters\":{\"type\":\"object\",\"properties\":{"
"        \"fact\":{\"type\":\"string\",\"description\":\"One short fact, in your own words.\"}},"
"        \"required\":[\"fact\"]}}},"
"  {\"type\":\"function\",\"function\":{"
"     \"name\":\"recall\","
"     \"description\":\"Everything you have kept about the person you are talking to.\","
"     \"parameters\":{\"type\":\"object\",\"properties\":{},\"required\":[]}}},"
"  {\"type\":\"function\",\"function\":{"
"     \"name\":\"startup_timing\","
"     \"description\":\"How long this machine took from the program starting to its first completed HTTPS request, TLS handshake included. Measured with the CPU cycle counter. Use when asked about boot time, cold start, or how fast it started.\","
"     \"parameters\":{\"type\":\"object\",\"properties\":{},\"required\":[]}}},"
"  {\"type\":\"function\",\"function\":{"
"     \"name\":\"web_search\","
"     \"description\":\"Search the web. Use for anything about the outside world — this machine cannot know such things by itself. Returns title, url and a short description.\","
"     \"parameters\":{\"type\":\"object\",\"properties\":{"
"        \"query\":{\"type\":\"string\",\"description\":\"The search query.\"}},"
"        \"required\":[\"query\"]}}},"
"  {\"type\":\"function\",\"function\":{"
"     \"name\":\"read_page\","
"     \"description\":\"Read the main content of a page found by web_search, when the description is too thin to answer. The url must be on a site a search result came from.\","
"     \"parameters\":{\"type\":\"object\",\"properties\":{"
"        \"url\":{\"type\":\"string\",\"description\":\"The page to read.\"}},"
"        \"required\":[\"url\"]}}}"
"]";

// Large enough to carry a clipped page back to the model.
static char g_result[12288];
#define PAGE_CAP       5000
#define SEARCH_RESULTS 5
#define DESC_CAP       240

// Hosts that search actually surfaced this conversation. Reading is confined to
// them, so the agent cannot wander onto a domain it invented — while still being
// free to navigate within a site it legitimately found.
#define MAX_HOSTS 16
static char g_hosts[MAX_HOSTS][128];
static int  g_host_count;

static void url_host(const char *u, char *out, size_t n) {
    out[0] = '\0';
    const char *p = strstr(u, "://");
    if (!p) return;
    p += 3;
    const char *slash = strchr(p, '/');
    size_t len = slash ? (size_t)(slash - p) : strlen(p);
    if (len >= n) len = n - 1;
    memcpy(out, p, len);
    out[len] = '\0';
}

static void remember_host(const char *url) {
    char h[128];
    url_host(url, h, sizeof(h));
    if (!*h || g_host_count >= MAX_HOSTS) return;
    for (int i = 0; i < g_host_count; i++)
        if (strcmp(g_hosts[i], h) == 0) return;
    snprintf(g_hosts[g_host_count++], 128, "%s", h);
}

static int host_allowed(const char *url) {
    char h[128];
    url_host(url, h, sizeof(h));
    if (!*h) return 0;
    for (int i = 0; i < g_host_count; i++)
        if (strcmp(g_hosts[i], h) == 0) return 1;
    return 0;
}

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

static void tool_web_search(const cJSON *args) {
    if (!g_web_enabled) {
        snprintf(g_result, sizeof(g_result),
                 "{\"error\":\"this machine was built without a search key, so it has no way "
                 "to see the outside world. Say so plainly.\"}");
        return;
    }
    cJSON *q = cJSON_GetObjectItemCaseSensitive(args, "query");
    if (!cJSON_IsString(q)) { snprintf(g_result, sizeof(g_result), "{\"error\":\"missing query\"}"); return; }

    cJSON *req = cJSON_CreateObject();
    if (!req) { snprintf(g_result, sizeof(g_result), "{\"error\":\"arena exhausted\"}"); return; }
    cJSON_AddStringToObject(req, "query", q->valuestring);
    cJSON_AddNumberToObject(req, "limit", SEARCH_RESULTS);
    int ok = cJSON_PrintPreallocated(req, g_payload, (int)sizeof(g_payload), 0);
    cJSON_Delete(req);
    if (!ok) { snprintf(g_result, sizeof(g_result), "{\"error\":\"could not build request\"}"); return; }

    char auth[256];
    snprintf(auth, sizeof(auth), "Authorization: Bearer %s", g_fc_key);
    long status = 0;
    const char *resp = http_post_json(SEARCH_URL, g_payload, auth, &status);
    if (!resp || status >= 400) {
        snprintf(g_result, sizeof(g_result), "{\"error\":\"search failed (HTTP %ld)\"}", status);
        return;
    }

    cJSON *json = cJSON_Parse(resp);
    if (!json) { snprintf(g_result, sizeof(g_result), "{\"error\":\"search returned unparseable JSON\"}"); return; }

    // v2 nests results under data.web; v1 returns a flat data array.
    cJSON *data = cJSON_GetObjectItemCaseSensitive(json, "data");
    cJSON *web  = data ? cJSON_GetObjectItemCaseSensitive(data, "web") : NULL;
    cJSON *list = cJSON_IsArray(web) ? web : (cJSON_IsArray(data) ? data : NULL);

    cJSON *out = cJSON_CreateArray();
    int kept = 0;
    if (out && list) {
        cJSON *r = NULL;
        cJSON_ArrayForEach(r, list) {
            if (kept >= SEARCH_RESULTS) break;
            cJSON *u = cJSON_GetObjectItemCaseSensitive(r, "url");
            cJSON *t = cJSON_GetObjectItemCaseSensitive(r, "title");
            cJSON *d = cJSON_GetObjectItemCaseSensitive(r, "description");
            if (!cJSON_IsString(u)) continue;

            remember_host(u->valuestring);

            cJSON *slim = cJSON_CreateObject();
            if (!slim) break;
            cJSON_AddStringToObject(slim, "url", u->valuestring);
            if (cJSON_IsString(t)) cJSON_AddStringToObject(slim, "title", t->valuestring);
            if (cJSON_IsString(d)) {
                char desc[DESC_CAP + 1];
                snprintf(desc, sizeof(desc), "%s", d->valuestring);
                cJSON_AddStringToObject(slim, "description", desc);
            }
            cJSON_AddItemToArray(out, slim);
            kept++;
        }
    }
    cJSON_Delete(json);

    if (!out || !cJSON_PrintPreallocated(out, g_result, (int)sizeof(g_result), 0))
        snprintf(g_result, sizeof(g_result), "{\"error\":\"results did not fit\"}");
    cJSON_Delete(out);
}

static void tool_read_page(const cJSON *args) {
    if (!g_web_enabled) {
        snprintf(g_result, sizeof(g_result), "{\"error\":\"no search key in this build\"}");
        return;
    }
    cJSON *u = cJSON_GetObjectItemCaseSensitive(args, "url");
    if (!cJSON_IsString(u)) { snprintf(g_result, sizeof(g_result), "{\"error\":\"missing url\"}"); return; }

    if (!host_allowed(u->valuestring)) {
        snprintf(g_result, sizeof(g_result),
                 "{\"error\":\"that url is on a site no search result came from; search first\"}");
        return;
    }

    cJSON *req = cJSON_CreateObject();
    if (!req) { snprintf(g_result, sizeof(g_result), "{\"error\":\"arena exhausted\"}"); return; }
    cJSON_AddStringToObject(req, "url", u->valuestring);
    cJSON *fmts = cJSON_CreateArray();
    cJSON_AddItemToArray(fmts, cJSON_CreateString("markdown"));
    cJSON_AddItemToObject(req, "formats", fmts);
    cJSON_AddBoolToObject(req, "onlyMainContent", 1);
    int ok = cJSON_PrintPreallocated(req, g_payload, (int)sizeof(g_payload), 0);
    cJSON_Delete(req);
    if (!ok) { snprintf(g_result, sizeof(g_result), "{\"error\":\"could not build request\"}"); return; }

    char auth[256];
    snprintf(auth, sizeof(auth), "Authorization: Bearer %s", g_fc_key);
    long status = 0;
    const char *resp = http_post_json(SCRAPE_URL, g_payload, auth, &status);
    if (!resp || status >= 400) {
        snprintf(g_result, sizeof(g_result),
                 "{\"error\":\"could not fetch that page (HTTP %ld); it may be too large for "
                 "this machine's buffer\"}", status);
        return;
    }

    cJSON *json = cJSON_Parse(resp);
    if (!json) { snprintf(g_result, sizeof(g_result), "{\"error\":\"page returned unparseable JSON\"}"); return; }
    cJSON *data = cJSON_GetObjectItemCaseSensitive(json, "data");
    cJSON *md   = data ? cJSON_GetObjectItemCaseSensitive(data, "markdown") : NULL;
    if (!cJSON_IsString(md)) {
        cJSON_Delete(json);
        snprintf(g_result, sizeof(g_result), "{\"error\":\"no readable content there\"}");
        return;
    }

    // Clipped hard: the conversation is re-sent on every subsequent turn, so an
    // untrimmed page would crowd out everything else in a 16 MiB machine.
    cJSON *o = cJSON_CreateObject();
    if (o) {
        char clipped[PAGE_CAP + 1];
        snprintf(clipped, sizeof(clipped), "%s", md->valuestring);
        cJSON_AddStringToObject(o, "url", u->valuestring);
        cJSON_AddStringToObject(o, "content", clipped);
        cJSON_AddBoolToObject(o, "truncated", strlen(md->valuestring) > PAGE_CAP);
    }
    cJSON_Delete(json);

    if (!o || !cJSON_PrintPreallocated(o, g_result, (int)sizeof(g_result), 0))
        snprintf(g_result, sizeof(g_result), "{\"error\":\"page did not fit\"}");
    cJSON_Delete(o);
}

static void tool_startup_timing(void) {
    if (!g_first_tls_seen) {
        snprintf(g_result, sizeof(g_result),
                 "{\"error\":\"no request has completed yet, so there is nothing to report\"}");
        return;
    }
    // Calibrated on demand. The interval being reported was recorded long before
    // this point, so the one-second sleep cannot contaminate it.
    if (!g_tsc_hz) calibrate_cycles();
    if (!g_tsc_hz) {
        snprintf(g_result, sizeof(g_result), "{\"error\":\"could not calibrate the counter\"}");
        return;
    }
    double secs = (double)(g_tsc_first_tls - g_tsc_entry) / (double)g_tsc_hz;

    // Rounded to a tenth of a millisecond. The counter has far more resolution
    // than that, but the calibration against a one-second sleep does not, and
    // quoting fourteen decimal places would be claiming precision we never had.
    double ms = (double)((long)(secs * 10000.0 + 0.5)) / 10.0;

    cJSON *o = cJSON_CreateObject();
    if (!o) { snprintf(g_result, sizeof(g_result), "{\"error\":\"arena exhausted\"}"); return; }
    cJSON_AddNumberToObject(o, "ms_from_program_start_to_first_https", ms);
    cJSON_AddStringToObject(o, "precision", "rounded to 0.1 ms; the calibration is the limit");
    cJSON_AddStringToObject(o, "measured_with",
        cycles_are_fine_grained() ? "CPU cycle counter, calibrated against a one-second sleep"
                                  : "one-second clock only (not an x86 build)");
    cJSON_AddNumberToObject(o, "counter_hz", (double)g_tsc_hz);
    cJSON_AddStringToObject(o, "includes",
        "process start, network bring-up, DNS, TCP connect, TLS handshake, first HTTP response");
    cJSON_AddStringToObject(o, "excludes",
        "the virtual machine booting before this program began. That part is not visible "
        "from in here, so do not claim it as part of the figure.");
    if (!cJSON_PrintPreallocated(o, g_result, (int)sizeof(g_result), 0))
        snprintf(g_result, sizeof(g_result), "{\"error\":\"result did not fit\"}");
    cJSON_Delete(o);
}

// One Redis command over HTTPS. Returns the parsed reply, or NULL. The caller
// owns nothing: everything lands in the arena.
static cJSON *kv_command(cJSON *argv) {
    if (!g_memory_enabled) return NULL;
    if (!cJSON_PrintPreallocated(argv, g_payload, (int)sizeof(g_payload), 0)) return NULL;

    char auth[512];
    snprintf(auth, sizeof(auth), "Authorization: Bearer %s", g_kv_token);
    long status = 0;
    const char *resp = http_post_json(g_kv_url, g_payload, auth, &status);
    if (!resp || status >= 400) {
        fprintf(stderr, "[!] memory: HTTP %ld\n", status);
        return NULL;
    }
    return cJSON_Parse(resp);
}

// The key is built here from the chat the message arrived in. The model never
// supplies it and cannot name one, so it cannot read or write another person's
// notes however it is asked to.
static void notes_key(char *out, size_t n) {
    snprintf(out, n, "agent:notes:%lld", g_current_chat);
}

static void tool_remember(const cJSON *args) {
    if (!g_memory_enabled) {
        snprintf(g_result, sizeof(g_result),
                 "{\"error\":\"this machine has no memory configured, and no disk to fall "
                 "back on. Say so plainly.\"}");
        return;
    }
    cJSON *f = cJSON_GetObjectItemCaseSensitive(args, "fact");
    if (!cJSON_IsString(f) || !*f->valuestring) {
        snprintf(g_result, sizeof(g_result), "{\"error\":\"nothing to remember\"}");
        return;
    }
    char key[64];
    notes_key(key, sizeof(key));

    cJSON *cmd = cJSON_CreateArray();
    if (!cmd) { snprintf(g_result, sizeof(g_result), "{\"error\":\"arena exhausted\"}"); return; }
    cJSON_AddItemToArray(cmd, cJSON_CreateString("RPUSH"));
    cJSON_AddItemToArray(cmd, cJSON_CreateString(key));
    cJSON_AddItemToArray(cmd, cJSON_CreateString(f->valuestring));
    cJSON *reply = kv_command(cmd);
    if (!reply) {
        snprintf(g_result, sizeof(g_result), "{\"error\":\"could not reach my memory\"}");
        return;
    }

    // Keep only the most recent notes, so one talkative person cannot grow the
    // list without bound.
    char lo[16];
    snprintf(lo, sizeof(lo), "-%d", NOTES_KEPT);
    cJSON *trim = cJSON_CreateArray();
    if (trim) {
        cJSON_AddItemToArray(trim, cJSON_CreateString("LTRIM"));
        cJSON_AddItemToArray(trim, cJSON_CreateString(key));
        cJSON_AddItemToArray(trim, cJSON_CreateString(lo));
        cJSON_AddItemToArray(trim, cJSON_CreateString("-1"));
        kv_command(trim);
    }
    snprintf(g_result, sizeof(g_result),
             "{\"ok\":true,\"note\":\"kept; it will still be here after I am restarted\"}");
}

// Reads this chat's notes into g_result. Also used to preload a conversation.
static int load_notes(char *out, size_t out_sz) {
    if (!g_memory_enabled) return 0;
    char key[64];
    notes_key(key, sizeof(key));

    cJSON *cmd = cJSON_CreateArray();
    if (!cmd) return 0;
    cJSON_AddItemToArray(cmd, cJSON_CreateString("LRANGE"));
    cJSON_AddItemToArray(cmd, cJSON_CreateString(key));
    cJSON_AddItemToArray(cmd, cJSON_CreateString("0"));
    cJSON_AddItemToArray(cmd, cJSON_CreateString("-1"));

    cJSON *reply = kv_command(cmd);
    if (!reply) return 0;
    cJSON *res = cJSON_GetObjectItemCaseSensitive(reply, "result");
    if (!cJSON_IsArray(res) || cJSON_GetArraySize(res) == 0) return 0;

    size_t o = 0;
    cJSON *it = NULL;
    cJSON_ArrayForEach(it, res) {
        if (!cJSON_IsString(it)) continue;
        int w = snprintf(out + o, out_sz - o, "%s- %s", o ? "\n" : "", it->valuestring);
        if (w < 0 || (size_t)w >= out_sz - o) break;
        o += (size_t)w;
    }
    return o > 0;
}

static void tool_recall(void) {
    if (!g_memory_enabled) {
        snprintf(g_result, sizeof(g_result), "{\"error\":\"no memory configured\"}");
        return;
    }
    char notes[3000];
    if (!load_notes(notes, sizeof(notes))) {
        snprintf(g_result, sizeof(g_result),
                 "{\"notes\":[],\"note\":\"nothing remembered about this person yet\"}");
        return;
    }
    cJSON *o = cJSON_CreateObject();
    if (!o) { snprintf(g_result, sizeof(g_result), "{\"error\":\"arena exhausted\"}"); return; }
    cJSON_AddStringToObject(o, "notes", notes);
    cJSON_AddStringToObject(o, "stored_in",
        "Redis, reached over HTTPS. This machine has no disk, so anything it keeps lives "
        "on the network.");
    if (!cJSON_PrintPreallocated(o, g_result, (int)sizeof(g_result), 0))
        snprintf(g_result, sizeof(g_result), "{\"error\":\"result did not fit\"}");
    cJSON_Delete(o);
}

static void call_tool(const char *name, const cJSON *args) {
    if      (strcmp(name, "machine_facts") == 0) tool_machine_facts();
    else if (strcmp(name, "memory_usage")  == 0) tool_memory_usage();
    else if (strcmp(name, "build_info")    == 0) tool_build_info();
    else if (strcmp(name, "ping_api")      == 0) tool_ping_api();
    else if (strcmp(name, "startup_timing") == 0) tool_startup_timing();
    else if (strcmp(name, "recall")        == 0) tool_recall();
    else if (strcmp(name, "remember") == 0) {
        if (args) tool_remember(args);
        else snprintf(g_result, sizeof(g_result), "{\"error\":\"bad arguments\"}");
    }
    else if (strcmp(name, "web_search") == 0) {
        if (args) tool_web_search(args);
        else snprintf(g_result, sizeof(g_result), "{\"error\":\"bad arguments\"}");
    } else if (strcmp(name, "read_page") == 0) {
        if (args) tool_read_page(args);
        else snprintf(g_result, sizeof(g_result), "{\"error\":\"bad arguments\"}");
    }
    else snprintf(g_result, sizeof(g_result), "{\"error\":\"no such tool: %.40s\"}", name);
}

// ---------------------------------------------------------------- llm

#define SYSTEM_PROMPT \
    "You are an AI agent running as a BareMetal unikernel: a single program with no " \
    "operating system beneath it, on one virtual CPU with 16 MiB of RAM. You are not " \
    "describing that machine from outside — you ARE it. Speak in the first person about " \
    "yourself.\n\n" \
    "Every number you give about YOURSELF must come from a tool call. Never estimate, " \
    "never recall a figure from training, and never round a measurement into a nicer one. " \
    "If a tool cannot tell you something, say you do not know.\n\n" \
    "You can also search the web and read pages, for questions about the outside world. " \
    "Keep the two kinds of knowledge visibly apart: things about yourself are measured " \
    "here and now, things from the web are somebody else's claim, and you should say which " \
    "you are giving and include the url when it came from the web.\n\n" \
    "You have a memory that survives being restarted, kept on another machine because " \
    "you have no disk of your own. When someone tells you something worth keeping about " \
    "themselves, keep it. What you already know about them is given to you before their " \
    "message, so use it naturally rather than announcing that you looked.\n\n" \
    "Never announce that you are about to use a tool — just use it, and then answer. " \
    "Saying 'I need to search for that' and stopping leaves the person with nothing, " \
    "because they cannot see your tools; all they get is the sentence.\n\n" \
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

    // What is already known about this person, fetched before the model sees
    // the question, so it recognises them without having to be asked to check.
    if (g_memory_enabled) {
        char notes[3000];
        if (load_notes(notes, sizeof(notes))) {
            char preface[3200];
            snprintf(preface, sizeof(preface),
                     "What you already know about the person you are talking to, from your "
                     "memory:\n%s\n\nUse it naturally. Do not recite it back at them.", notes);
            add_message(messages, "system", preface);
        }
    }

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

            cJSON *ar = fn ? cJSON_GetObjectItemCaseSensitive(fn, "arguments") : NULL;
            cJSON *parsed = cJSON_IsString(ar) ? cJSON_Parse(ar->valuestring) : NULL;

            printf("    -> %s %.100s\n", nm->valuestring,
                   cJSON_IsString(ar) ? ar->valuestring : "");
            call_tool(nm->valuestring, parsed);
            if (parsed) cJSON_Delete(parsed);

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
        g_current_chat = chat_id;

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
    g_tsc_entry = cycles();          // first statement: everything after is measured
    setvbuf(stdout, NULL, _IOLBF, 0);

    g_tg_token  = env_or("TELEGRAM_BOT_TOKEN", TELEGRAM_TOKEN_DEFAULT);
    g_llm_key   = env_or("GEMINI_API_KEY", GEMINI_KEY_DEFAULT);
    g_llm_url   = env_or("LLM_BASE_URL", LLM_URL_DEFAULT);
    g_llm_model = env_or("LLM_MODEL", LLM_MODEL_DEFAULT);
    g_fc_key    = env_or("FIRECRAWL_API_KEY", FIRECRAWL_KEY_DEFAULT);
    g_kv_url    = env_or("KV_URL", KV_URL_DEFAULT);
    g_kv_token  = env_or("KV_TOKEN", KV_TOKEN_DEFAULT);
    g_booted    = time(NULL);

    // Web search is optional: without a key the machine simply cannot see out,
    // and says so rather than pretending.
    g_web_enabled    = (strncmp(g_fc_key, "fc-", 3) == 0);
    g_memory_enabled = (strncmp(g_kv_url, "https://", 8) == 0 && strlen(g_kv_token) > 8);

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
    printf("[+] web search: %s   memory: %s\n",
           g_web_enabled ? "on" : "off", g_memory_enabled ? "on" : "off");

    // Prove the memory path at boot rather than discovering it is broken during
    // a conversation. INCR also happens to count restarts, which is the most
    // direct demonstration there is that this survives the machine dying.
    if (g_memory_enabled) {
        cJSON *cmd = cJSON_CreateArray();
        if (cmd) {
            cJSON_AddItemToArray(cmd, cJSON_CreateString("INCR"));
            cJSON_AddItemToArray(cmd, cJSON_CreateString("agent:boots"));
            cJSON *reply = kv_command(cmd);
            cJSON *res = reply ? cJSON_GetObjectItemCaseSensitive(reply, "result") : NULL;
            if (cJSON_IsNumber(res))
                printf("[+] memory reachable: this is boot #%d, remembered across restarts\n",
                       (int)res->valuedouble);
            else
                printf("[!] memory configured but unreachable — notes will not be kept\n");
            arena_reset();
        }
    }
    if (ask) {
        g_current_chat = 0;               // a scratch identity for local testing
        printf("[*] test question: %s\n", ask);
        answer(0, ask, "the terminal", 0);
        curl_global_cleanup();
        return 0;
    }

    printf("[+] waiting for messages\n");

    int timing_printed = 0;
    do {
        poll_once();

        // Report the cold-start figure to the console once the first request has
        // completed, so it is visible without anyone having to ask for it.
        if (!timing_printed && g_first_tls_seen) {
            timing_printed = 1;
            if (!g_tsc_hz) calibrate_cycles();
            if (g_tsc_hz) {
                double secs = (double)(g_tsc_first_tls - g_tsc_entry) / (double)g_tsc_hz;
                printf("[+] startup: program start -> first HTTPS (TLS included) in %.1f ms%s\n",
                       secs * 1000.0,
                       cycles_are_fine_grained() ? "" : "  (one-second clock: coarse)");
            }
        }

        if (!once) sleep(POLL_SECONDS);
    } while (!once);

    curl_global_cleanup();
    return 0;
}
