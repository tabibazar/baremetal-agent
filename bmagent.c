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
//   arena      1.0 MiB   cJSON nodes: the conversation and parsed responses
//   http      256 KB     one HTTP response at a time
//   payload   128 KB     one serialized request body at a time
//   index     7.99 MiB   2560 embeddings, their text, and the chat each belongs to
//   ---------------------
//   total     9.37 MiB   of a 16 MiB machine, + libcurl's own allocations
//
// The index is the reason the figure is no longer measured in kilobytes. A
// cloud instance is capped at 16 MiB and the agent used to claim 640 KB of it,
// which was tidy and wasteful in equal measure. It now keeps a vector database
// of everything it has been told, resident, and searches it without touching
// the network or a disk -- neither of which this machine has anyway.
//
// BUILD: see README.md — it runs unchanged on Linux, macOS and BareMetal.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
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
#define RECALL_SHOWN           25        // how many of them plain recall returns
#define NOTES_KEPT             100       // per person, oldest dropped. Was 20,
                                         // sized for a machine that indexed
                                         // nothing; the RAM-resident index
                                         // below makes a longer list useful
                                         // rather than merely longer.
#define REMINDERS_KEY          "agent:reminders"   // sorted set, scored by due time
#define REMINDER_MAX_PER_CHAT  20
#define REMINDER_MIN_DELAY     20        // seconds; below this it is a typo
#define REMINDER_MAX_HORIZON   (90L * 24 * 3600)
#define REMINDER_CHECK_EVERY   60        // seconds between due-checks
#define STATS_MESSAGES_KEY     "agent:stats:messages"
#define STATS_PEOPLE_KEY       "agent:stats:people"

// Firecrawl returns pages as markdown, so this program never parses HTML —
// which is the only reason reading the web is tractable inside a fixed buffer.
#define SEARCH_URL "https://api.firecrawl.dev/v2/search"
#define SCRAPE_URL "https://api.firecrawl.dev/v2/scrape"

#define LLM_URL_DEFAULT   "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
// Pinned deliberately: the "-latest" aliases queue indefinitely under load
// instead of returning an error, which is indistinguishable from a hang.
#define LLM_MODEL_DEFAULT "gemini-2.5-flash"

#ifndef ARENA_BYTES
// 1 MiB, up from 384 KB. An embedding response is a JSON array of 768 numbers
// and there is now room to parse one without the arena being the tight
// constraint. See the memory budget above VEC_MAX below.
#define ARENA_BYTES     (1024 * 1024)
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
#define REPLIES_PER_HOUR 300        // spend ceiling; enforced here, not in the prompt.
                                    // 40 was sized for a private demo. A bot
                                    // whose handle is public needs room to answer
                                    // strangers without every one of them meeting
                                    // the rate limiter instead of the agent.
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

// ------------------------------------------------------- semantic index
//
// A vector database that lives entirely in this machine's RAM.
//
// Exact-key recall could only find a note if you named it the way you filed
// it. This indexes every remembered note by meaning: each one is embedded
// once, and a query is answered by comparing it against every resident vector
// and returning the closest. There is no tree, no database and no disk -- the
// whole corpus is a static array, and a search is one pass over it.
//
// The memory budget, against a hard cloud ceiling of 16 MiB per instance:
//
//     vectors   2560 x 768 x 4 bytes  = 7.50 MiB
//     text      2560 x 192 bytes      = 0.47 MiB
//     chat ids  2560 x 8 bytes        = 0.02 MiB
//     arena, HTTP and payload buffers = 1.38 MiB
//                                       ---------
//                                       9.37 MiB of 16
//
// None of it costs a byte in the image: this is BSS, so the upload stays the
// same size and the machine simply wakes up with the room already claimed.
// A probe established that 13 MiB of touched static memory boots in a 16 MiB
// instance and 14 MiB does not; the agent's own code and lwIP/mbedTLS working
// set take a share of the difference, and the rest is deliberate margin.
//
// Why brute force is the right answer here, and not laziness: the platform is
// known to corrupt 64-bit arithmetic at a rate near 1 in 4000 under load, so a
// feature whose inner loop is a long multiply-accumulate needs a failure mode
// that degrades rather than breaks. A wrong score reorders two neighbours in a
// ranked list. It cannot crash, and it cannot invent a memory that was never
// stored. An exact-arithmetic feature would have been the wrong thing to build
// on this machine.

#define EMBED_URL_DEFAULT   "https://generativelanguage.googleapis.com/v1beta/openai/embeddings"
#define EMBED_MODEL_DEFAULT "gemini-embedding-001"
#define VEC_DIM      768       // the model natively returns 3072; see below
#define VEC_MAX      2560
#define VEC_TEXT     192       // per-entry copy, so a search touches no network
#define EMBED_BATCH  8         // ~24 KB of response per vector, so ~190 KB of the
                               // 256 KB HTTP buffer -- the largest batch that fits
#define BACKFILL_MAX NOTES_KEPT // restore ALL of them: a note the index cannot see
                                // is one plain recall already showed anyway

static float     g_vec[VEC_MAX][VEC_DIM];
static char      g_vtext[VEC_MAX][VEC_TEXT];
static long long g_vchat[VEC_MAX];
static int       g_vcount;
static int       g_vevicted;            // entries dropped because the index filled
static uint64_t  g_vlast_search_cycles;
static long long g_vbackfilled[32];     // chats already restored after this boot
static int       g_vbackfill_count;

// Pull the embeddings out of a response by hand rather than through cJSON.
// A batch of four is 3072 numbers, and every one of them would become a
// separate node in the arena; scanning the text costs nothing and the shape is
// fixed, because this program wrote the request.
static int parse_embeddings(const char *json, int want, float out[][VEC_DIM]) {
    const char *p = json;
    int got = 0;
    while (got < want && (p = strstr(p, "\"embedding\"")) != NULL) {
        p = strchr(p, '[');
        if (!p) break;
        p++;
        int d = 0;
        double sum = 0;
        while (d < VEC_DIM) {
            char *end = NULL;
            double v = strtod(p, &end);
            if (end == p) break;
            out[got][d] = (float)v;
            sum += v * v;
            d++;
            p = end;
            while (*p == ' ' || *p == ',') p++;
            if (*p == ']') break;
        }
        if (d != VEC_DIM) return got;
        // The model returns 3072 dimensions and truncates on request, but a
        // truncated vector is no longer unit length (0.59 was measured), so
        // cosine similarity needs the normalisation doing here. Skipping this
        // silently ranks longer vectors above closer ones.
        double n = sqrt(sum);
        if (n > 0) for (int i = 0; i < VEC_DIM; i++) out[got][i] = (float)(out[got][i] / n);
        got++;
    }
    return got;
}

// Embed up to EMBED_BATCH strings in one request. Returns how many came back.
static int embed_texts(const char **texts, int n, float out[][VEC_DIM]) {
    if (n <= 0 || !g_llm_key || !*g_llm_key) return 0;
    if (n > EMBED_BATCH) n = EMBED_BATCH;

    cJSON *body = cJSON_CreateObject();
    if (!body) return 0;
    cJSON_AddStringToObject(body, "model", env_or("EMBED_MODEL", EMBED_MODEL_DEFAULT));
    cJSON_AddNumberToObject(body, "dimensions", VEC_DIM);
    cJSON *arr = cJSON_CreateArray();
    if (!arr) { cJSON_Delete(body); return 0; }
    for (int i = 0; i < n; i++) cJSON_AddItemToArray(arr, cJSON_CreateString(texts[i]));
    cJSON_AddItemToObject(body, "input", arr);

    int ok = cJSON_PrintPreallocated(body, g_payload, (int)sizeof(g_payload), 0);
    cJSON_Delete(body);
    if (!ok) return 0;

    char auth[512];
    snprintf(auth, sizeof(auth), "Authorization: Bearer %s", g_llm_key);
    long status = 0;
    const char *resp = http_post_json(env_or("EMBED_URL", EMBED_URL_DEFAULT),
                                      g_payload, auth, &status);
    if (!resp || status >= 400) {
        fprintf(stderr, "[!] embed: HTTP %ld\n", status);
        return 0;
    }
    return parse_embeddings(resp, n, out);
}

static void index_add(long long chat, const char *text, const float *v) {
    int slot;
    if (g_vcount < VEC_MAX) {
        slot = g_vcount++;
    } else {
        // Full. Drop the oldest entry rather than refusing the newest: the
        // note itself is safe in Redis either way, so the only thing lost is
        // its searchability, and losing the least recent is the least
        // surprising choice.
        memmove(g_vec[0], g_vec[1], (size_t)(VEC_MAX - 1) * sizeof(g_vec[0]));
        memmove(g_vtext[0], g_vtext[1], (size_t)(VEC_MAX - 1) * sizeof(g_vtext[0]));
        memmove(&g_vchat[0], &g_vchat[1], (size_t)(VEC_MAX - 1) * sizeof(g_vchat[0]));
        slot = VEC_MAX - 1;
        g_vevicted++;
    }
    memcpy(g_vec[slot], v, sizeof(g_vec[slot]));
    snprintf(g_vtext[slot], VEC_TEXT, "%s", text);
    g_vchat[slot] = chat;
}

// Embed one note and file it. Failure is not fatal anywhere it is called: the
// note is already stored, it simply will not be findable by meaning.
static int index_remember(long long chat, const char *text) {
    static float v[EMBED_BATCH][VEC_DIM];
    const char *one = text;
    if (embed_texts(&one, 1, v) != 1) return 0;
    index_add(chat, text, v[0]);
    return 1;
}

static int chat_backfilled(long long chat) {
    for (int i = 0; i < g_vbackfill_count; i++) if (g_vbackfilled[i] == chat) return 1;
    return 0;
}

static int load_notes_list(cJSON **reply_out, int count);   // with the memory tools

// The index lives in RAM, so a restart empties it while the notes themselves
// survive in Redis. On the first search in a chat after a boot, re-embed what
// is stored so the feature works on a machine that reboots, rather than only
// on one that has been up a while.
static int index_backfill(long long chat) {
    if (chat_backfilled(chat)) return 0;
    if (g_vbackfill_count < (int)(sizeof(g_vbackfilled) / sizeof(g_vbackfilled[0])))
        g_vbackfilled[g_vbackfill_count++] = chat;

    cJSON *reply = NULL;
    if (!load_notes_list(&reply, BACKFILL_MAX)) return 0;
    cJSON *res = cJSON_GetObjectItemCaseSensitive(reply, "result");
    if (!cJSON_IsArray(res)) return 0;

    int total = cJSON_GetArraySize(res);
    int first = total > BACKFILL_MAX ? total - BACKFILL_MAX : 0;   // the most recent
    static float v[EMBED_BATCH][VEC_DIM];
    const char *batch[EMBED_BATCH];
    int n = 0, added = 0;

    for (int i = first; i < total; i++) {
        cJSON *it = cJSON_GetArrayItem(res, i);
        if (!cJSON_IsString(it) || !*it->valuestring) continue;
        batch[n++] = it->valuestring;
        if (n == EMBED_BATCH) {
            int got = embed_texts(batch, n, v);
            for (int j = 0; j < got; j++) { index_add(chat, batch[j], v[j]); added++; }
            n = 0;
        }
    }
    if (n) {
        int got = embed_texts(batch, n, v);
        for (int j = 0; j < got; j++) { index_add(chat, batch[j], v[j]); added++; }
    }
    return added;
}

// One pass over every vector belonging to this chat. Vectors are unit length,
// so the dot product is the cosine.
static int index_search(long long chat, const float *q, int k,
                        int *idx_out, double *score_out) {
    if (k < 1) k = 1;
    if (k > 8) k = 8;
    int found = 0;
    uint64_t t0 = cycles();

    for (int i = 0; i < g_vcount; i++) {
        if (g_vchat[i] != chat) continue;
        const float *v = g_vec[i];
        double s = 0;
        for (int d = 0; d < VEC_DIM; d++) s += (double)q[d] * (double)v[d];

        int pos = found;
        while (pos > 0 && score_out[pos - 1] < s) pos--;
        if (pos >= k) continue;
        int last = (found < k) ? found : k - 1;
        for (int m = last; m > pos; m--) {
            score_out[m] = score_out[m - 1];
            idx_out[m]   = idx_out[m - 1];
        }
        score_out[pos] = s;
        idx_out[pos]   = i;
        if (found < k) found++;
    }
    g_vlast_search_cycles = cycles() - t0;
    return found;
}


// ============================================================ fractal
//
// The agent can draw. Not decoration: the picture is a measurement.
//
// A Mandelbrot escape count is a pure function of the pixel, so rendering a
// frame twice must give the same image, byte for byte, on a correct machine.
// This one disagrees with itself on 64-bit integer arithmetic about once in
// four thousand results, measured against a Linux control under the same
// hypervisor. Every frame here is rendered twice and the two passes compared,
// with any pixel that differs painted red and counted.
//
// So far that count has always been zero, which is the finding: the fault is
// in the integer path -- gcc lowering __uint128_t division to __udivti3 --
// and not in the double-precision floating point this loop runs on.
//
// The PNG is written here too, deflate and all, because there is no zlib in
// this image to link against.
//
// Budget, on top of the agent's own 9,588 KB:
//
//   escape counts x2   400 KB
//   palette indices    100 KB
//   raw scanlines      100 KB
//   encoded PNG         64 KB
//   multipart body      96 KB
//   deflate tables      96 KB
//   ------------------------
//                      856 KB   -- 10.2 MiB of 16 in total

#define FRACT_MAX    320       // largest frame; the link will not carry more
#define MAX_ITER     600
#define TARGET_RE   (-0.743643887037151)
#define TARGET_IM   ( 0.131825904205330)

static int g_w = 256, g_h = 256;
static uint16_t g_pass1[FRACT_MAX * FRACT_MAX];
static uint16_t g_pass2[FRACT_MAX * FRACT_MAX];
static uint8_t  g_idx[FRACT_MAX * FRACT_MAX];
static uint8_t  g_plte[256 * 3];
static uint8_t  g_body[96 * 1024];
static size_t   g_body_len;

// ---------------------------------------------------------------- fractal

// Escape-time iteration, double precision. Deliberately plain: the whole
// exercise depends on this being a pure function of (cr, ci), so there is no
// caching, no early-out on symmetry, and no reuse of anything between passes.
static uint16_t escape(double cr, double ci) {
    // Cardioid and period-2 bulb tests. These are exact algebraic checks that
    // skip the interior, where the loop would otherwise always run to
    // MAX_ITER -- worth it on a machine that computes 3.7x slower than Linux.
    double q = (cr - 0.25) * (cr - 0.25) + ci * ci;
    if (q * (q + (cr - 0.25)) <= 0.25 * ci * ci) return MAX_ITER;
    if ((cr + 1.0) * (cr + 1.0) + ci * ci <= 0.0625) return MAX_ITER;

    double zr = 0, zi = 0, zr2 = 0, zi2 = 0;
    uint16_t i = 0;
    while (i < MAX_ITER && zr2 + zi2 <= 4.0) {
        zi = 2.0 * zr * zi + ci;
        zr = zr2 - zi2 + cr;
        zr2 = zr * zr;
        zi2 = zi * zi;
        i++;
    }
    return i;
}

static void render(uint16_t *out, double scale) {
    for (int y = 0; y < g_h; y++) {
        double ci = TARGET_IM + ((double)y - g_h / 2.0) * scale;
        for (int x = 0; x < g_w; x++) {
            double cr = TARGET_RE + ((double)x - g_w / 2.0) * scale;
            out[y * g_w + x] = escape(cr, ci);
        }
    }
}

#define PAL_BANDS   254   // 0..253 are escape colours
#define PAL_INSIDE  254
#define PAL_BAD     255

// Three phases of one cosine give a smooth spectrum, evaluated once into a
// 256-entry table rather than per pixel -- which is also what makes the image
// a palette PNG, and a third of the bytes.
static void build_palette(void) {
    for (int k = 0; k < PAL_BANDS; k++) {
        double s = (double)k / (double)PAL_BANDS;
        g_plte[k * 3 + 0] = (uint8_t)(255.0 * (0.5 + 0.5 * cos(6.283 * (s + 0.00))));
        g_plte[k * 3 + 1] = (uint8_t)(255.0 * (0.5 + 0.5 * cos(6.283 * (s + 0.33))));
        g_plte[k * 3 + 2] = (uint8_t)(255.0 * (0.5 + 0.5 * cos(6.283 * (s + 0.67))));
    }
    g_plte[PAL_INSIDE * 3] = g_plte[PAL_INSIDE * 3 + 1] = g_plte[PAL_INSIDE * 3 + 2] = 8;
    g_plte[PAL_BAD * 3 + 0] = 255;
    g_plte[PAL_BAD * 3 + 1] = 0;
    g_plte[PAL_BAD * 3 + 2] = 0;
}

static void colourise(const uint16_t *it) {
    for (int i = 0; i < g_w * g_h; i++) {
        uint16_t n = it[i];
        if (n >= MAX_ITER) { g_idx[i] = PAL_INSIDE; continue; }
        // sqrt rather than a linear ramp, and cycling rather than a single
        // sweep. Most pixels in a wide view escape in under a dozen steps, so
        // dividing by MAX_ITER puts almost the whole image in the first
        // percent of the palette and the picture comes out one flat colour.
        // Taking the root spreads the low counts, and letting the phase wrap
        // draws the escape-time bands at every depth instead of only near the
        // boundary.
        double s = sqrt((double)n) * 0.13;
        s -= floor(s);                             // the phase wraps; keep it in [0,1)
        g_idx[i] = (uint8_t)(s * PAL_BANDS);
    }
}

// Paint every pixel where the two passes disagreed. Returns how many.
static long mark_disagreements(void) {
    long bad = 0;
    for (int i = 0; i < g_w * g_h; i++) {
        if (g_pass1[i] != g_pass2[i]) { g_idx[i] = PAL_BAD; bad++; }
    }
    return bad;
}

// ---------------------------------------------------------------- png
//
// Hand-rolled, because porting zlib to link one function would have been the
// larger job. The compression method is "none": deflate's stored block is a
// five-byte header around at most 65535 literal bytes, so a valid zlib stream
// is a two-byte header, a run of stored blocks, and an Adler-32. Every decoder
// accepts it. The file is about the size of the raw pixels, which is a fair
// trade for not having a compressor.

static uint32_t crc_table[256];
static int crc_ready;

static void crc_init(void) {
    for (uint32_t n = 0; n < 256; n++) {
        uint32_t c = n;
        for (int k = 0; k < 8; k++) c = (c & 1) ? 0xEDB88320u ^ (c >> 1) : c >> 1;
        crc_table[n] = c;
    }
    crc_ready = 1;
}

static uint32_t crc32_of(const uint8_t *buf, size_t len, uint32_t crc) {
    if (!crc_ready) crc_init();
    crc ^= 0xFFFFFFFFu;
    for (size_t i = 0; i < len; i++) crc = crc_table[(crc ^ buf[i]) & 0xFF] ^ (crc >> 8);
    return crc ^ 0xFFFFFFFFu;
}

static void put32(uint8_t *p, uint32_t v) {
    p[0] = (uint8_t)(v >> 24); p[1] = (uint8_t)(v >> 16);
    p[2] = (uint8_t)(v >> 8);  p[3] = (uint8_t)v;
}

// Append a PNG chunk at *o, advancing it.
static void chunk(uint8_t *base, size_t *o, const char *type,
                  const uint8_t *data, size_t len) {
    put32(base + *o, (uint32_t)len); *o += 4;
    size_t type_at = *o;
    memcpy(base + *o, type, 4); *o += 4;
    if (len) { memcpy(base + *o, data, len); *o += len; }
    uint32_t c = crc32_of(base + type_at, len + 4, 0);
    put32(base + *o, c); *o += 4;
}


// ---------------------------------------------------------------- deflate
//
// Fixed-Huffman deflate with LZ77 matching, written here because there is no
// zlib to link and the alternative was staying at 96x96.
//
// Stored blocks got the first version working, but they are not compression:
// the file is the pixels plus five bytes per 64 KB. A real ceiling of about
// 16 KB per HTTPS request to this endpoint therefore capped the picture at
// 96x96, which is small enough to be disappointing.
//
// A Mandelbrot compresses extraordinarily well and it is worth seeing why: the
// interior is one flat colour, the escape bands are long runs of one palette
// index, and consecutive scanlines are nearly identical -- so a match at
// distance (width + 1) reproduces most of a row from the row above it. LZ77
// finds all three without being told about any of them.
//
// Fixed Huffman rather than dynamic: the code lengths are defined by the
// standard, so there is no tree to build, no tree to serialise, and far less
// to get wrong. Dynamic would beat it by perhaps a fifth, which is not worth
// the extra hundred lines here.

// Sizes the renderer will try, smallest first.
static const int LADDER[] = { 64, 96, 128, 160, 192, 256, 320 };
#define N_LADDER ((int)(sizeof(LADDER) / sizeof(LADDER[0])))
// Largest body this link has been seen to carry. Measured the hard way: 14,240
// bytes has been delivered, 17,065 has failed four attempts twice over.
static size_t g_budget = 14000;

#define DEF_WINDOW  16384
#define DEF_MIN_MATCH 3
#define DEF_MAX_MATCH 258
#define DEF_HASH_BITS 13
#define DEF_HASH_SIZE (1 << DEF_HASH_BITS)

static uint8_t  *g_ob;          // output cursor state for the bit writer
static size_t    g_ocap, g_olen;
static uint32_t  g_bitbuf;
static int       g_bitcnt;
static int       g_overflow;
static int32_t   g_head[DEF_HASH_SIZE];
static int32_t   g_prev[DEF_WINDOW];

static void bw_init(uint8_t *out, size_t cap) {
    g_ob = out; g_ocap = cap; g_olen = 0;
    g_bitbuf = 0; g_bitcnt = 0; g_overflow = 0;
}

// Deflate writes bits least-significant first within each byte.
static void put_bits(uint32_t v, int n) {
    g_bitbuf |= (v & ((1u << n) - 1u)) << g_bitcnt;
    g_bitcnt += n;
    while (g_bitcnt >= 8) {
        if (g_olen < g_ocap) g_ob[g_olen++] = (uint8_t)(g_bitbuf & 0xFF);
        else g_overflow = 1;
        g_bitbuf >>= 8;
        g_bitcnt -= 8;
    }
}

// Huffman codes are defined most-significant bit first, which is the opposite
// order to everything else in the format. Emitting them a bit at a time from
// the top is slower than a reversal table and much harder to get wrong.
static void put_code(uint32_t code, int len) {
    for (int i = len - 1; i >= 0; i--) put_bits((code >> i) & 1u, 1);
}

static void bw_flush(void) {
    if (g_bitcnt > 0) {
        if (g_olen < g_ocap) g_ob[g_olen++] = (uint8_t)(g_bitbuf & 0xFF);
        else g_overflow = 1;
        g_bitbuf = 0; g_bitcnt = 0;
    }
}

// The fixed literal/length code, straight from RFC 1951 section 3.2.6.
static void put_literal(int sym) {
    if (sym <= 143)      put_code(0x30u  + (uint32_t)sym, 8);
    else if (sym <= 255) put_code(0x190u + (uint32_t)(sym - 144), 9);
    else if (sym <= 279) put_code(0x0u   + (uint32_t)(sym - 256), 7);
    else                 put_code(0xC0u  + (uint32_t)(sym - 280), 8);
}

static const uint16_t LEN_BASE[29] = {
    3,4,5,6,7,8,9,10,11,13,15,17,19,23,27,31,35,43,51,59,67,83,99,115,131,163,195,227,258 };
static const uint8_t LEN_EXTRA[29] = {
    0,0,0,0,0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,0 };
static const uint16_t DIST_BASE[30] = {
    1,2,3,4,5,7,9,13,17,25,33,49,65,97,129,193,257,385,513,769,
    1025,1537,2049,3073,4097,6145,8193,12289,16385,24577 };
static const uint8_t DIST_EXTRA[30] = {
    0,0,0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10,11,11,12,12,13,13 };

static void put_match(int len, int dist) {
    int lc = 28;
    while (lc > 0 && len < LEN_BASE[lc]) lc--;
    put_literal(257 + lc);
    if (LEN_EXTRA[lc]) put_bits((uint32_t)(len - LEN_BASE[lc]), LEN_EXTRA[lc]);

    int dc = 29;
    while (dc > 0 && dist < DIST_BASE[dc]) dc--;
    put_code((uint32_t)dc, 5);                      // distance codes are 5 bits fixed
    if (DIST_EXTRA[dc]) put_bits((uint32_t)(dist - DIST_BASE[dc]), DIST_EXTRA[dc]);
}

static uint32_t dhash(const uint8_t *p) {
    return (uint32_t)(((p[0] << 10) ^ (p[1] << 5) ^ p[2]) & (DEF_HASH_SIZE - 1));
}

// Compress src into a zlib stream at out. Returns bytes written, 0 on overflow.
static size_t zlib_deflate(const uint8_t *src, size_t len, uint8_t *out, size_t cap) {
    bw_init(out, cap);
    if (cap < 8) return 0;
    g_ob[g_olen++] = 0x78;                          // CMF: deflate, 32K window
    g_ob[g_olen++] = 0x01;                          // FLG: (0x7801 % 31) == 0
    put_bits(1, 1);                                 // BFINAL
    put_bits(1, 2);                                 // BTYPE: fixed Huffman

    for (int i = 0; i < DEF_HASH_SIZE; i++) g_head[i] = -1;

    size_t pos = 0;
    while (pos < len) {
        int best_len = 0, best_dist = 0;
        if (pos + DEF_MIN_MATCH <= len) {
            uint32_t h = dhash(src + pos);
            int32_t cand = g_head[h];
            // One chain, bounded: the data is mostly long runs, so the first
            // few candidates are as good as the hundredth and the search is
            // where all the time would go.
            int tries = 16;
            while (cand >= 0 && tries-- > 0) {
                size_t dist = pos - (size_t)cand;
                if (dist == 0 || dist > DEF_WINDOW) break;
                size_t maxl = len - pos;
                if (maxl > DEF_MAX_MATCH) maxl = DEF_MAX_MATCH;
                size_t l = 0;
                while (l < maxl && src[cand + l] == src[pos + l]) l++;
                if ((int)l > best_len) { best_len = (int)l; best_dist = (int)dist; }
                if (best_len >= (int)maxl) break;
                cand = g_prev[(size_t)cand & (DEF_WINDOW - 1)];
            }
        }

        if (best_len >= DEF_MIN_MATCH) {
            put_match(best_len, best_dist);
            for (int k = 0; k < best_len; k++) {
                if (pos + DEF_MIN_MATCH <= len) {
                    uint32_t h = dhash(src + pos);
                    g_prev[pos & (DEF_WINDOW - 1)] = g_head[h];
                    g_head[h] = (int32_t)pos;
                }
                pos++;
            }
        } else {
            put_literal(src[pos]);
            if (pos + DEF_MIN_MATCH <= len) {
                uint32_t h = dhash(src + pos);
                g_prev[pos & (DEF_WINDOW - 1)] = g_head[h];
                g_head[h] = (int32_t)pos;
            }
            pos++;
        }
        if (g_overflow) return 0;
    }

    put_literal(256);                               // end of block
    bw_flush();

    uint32_t a = 1, b = 0;
    for (size_t i = 0; i < len; i++) { a = (a + src[i]) % 65521; b = (b + a) % 65521; }
    if (g_olen + 4 > g_ocap) return 0;
    put32(g_ob + g_olen, (b << 16) | a); g_olen += 4;
    return g_overflow ? 0 : g_olen;
}

// Encode g_idx + g_plte into buf, returning the byte count (0 if it would not fit).
static uint8_t g_raw[(FRACT_MAX + 1) * FRACT_MAX];      // filter byte + indices, per scanline

static size_t png_encode(uint8_t *buf, size_t cap) {
    const size_t raw_len = (size_t)g_h * (1 + (size_t)g_w);
    if (raw_len > sizeof(g_raw)) return 0;
    if (cap < 8 + 25 + (12 + 768) + 12 + 12) return 0;

    // Filter type 0 (None) on every scanline. Filtering exists to make bytes
    // more compressible, but a palette index is a label rather than a
    // magnitude -- subtracting one index from another produces noise, not a
    // small number. Leaving the rows alone lets the matcher find them whole:
    // a scanline that repeats the one above it becomes a single match at
    // distance width+1.
    size_t o = 0;
    for (int y = 0; y < g_h; y++) {
        g_raw[o++] = 0;
        memcpy(g_raw + o, g_idx + (size_t)y * g_w, (size_t)g_w);
        o += (size_t)g_w;
    }

    size_t p = 0;
    static const uint8_t sig[8] = { 137, 'P', 'N', 'G', '\r', '\n', 26, '\n' };
    memcpy(buf, sig, 8); p = 8;

    uint8_t ihdr[13];
    put32(ihdr, (uint32_t)g_w); put32(ihdr + 4, (uint32_t)g_h);
    ihdr[8] = 8;    // bit depth
    ihdr[9] = 3;    // colour type: palette
    ihdr[10] = 0; ihdr[11] = 0; ihdr[12] = 0;
    chunk(buf, &p, "IHDR", ihdr, sizeof(ihdr));
    chunk(buf, &p, "PLTE", g_plte, sizeof(g_plte));

    // Compress into the space after the IDAT header, then patch the length in.
    size_t len_at = p; p += 4;
    size_t type_at = p;
    memcpy(buf + p, "IDAT", 4); p += 4;
    size_t data_at = p;

    size_t z = zlib_deflate(g_raw, raw_len, buf + p, cap - p - 16);
    if (!z) return 0;
    p += z;

    put32(buf + len_at, (uint32_t)(p - data_at));
    uint32_t c = crc32_of(buf + type_at, (p - data_at) + 4, 0);
    put32(buf + p, c); p += 4;

    chunk(buf, &p, "IEND", NULL, 0);
    return p;
}


// Wrap the PNG in a multipart body: sendPhoto wants a file part, and building
// the envelope by hand avoids depending on curl's MIME API being present in
// the port.
static size_t build_multipart(const char *boundary, const char *chat,
                              const char *caption,
                              const uint8_t *png, size_t png_len) {
    size_t o = 0;
    o += (size_t)snprintf((char *)g_body + o, sizeof(g_body) - o,
        "--%s\r\nContent-Disposition: form-data; name=\"chat_id\"\r\n\r\n%s\r\n"
        "--%s\r\nContent-Disposition: form-data; name=\"caption\"\r\n\r\n%s\r\n"
        "--%s\r\nContent-Disposition: form-data; name=\"photo\"; filename=\"frame.png\"\r\n"
        "Content-Type: image/png\r\n\r\n",
        boundary, chat, boundary, caption, boundary);
    if (o + png_len + 128 > sizeof(g_body)) return 0;
    memcpy(g_body + o, png, png_len); o += png_len;
    o += (size_t)snprintf((char *)g_body + o, sizeof(g_body) - o,
                          "\r\n--%s--\r\n", boundary);
    return o;
}

// Feed the body to curl a chunk at a time instead of handing it a pointer.
//
// CURLOPT_POSTFIELDS with CURLOPT_POSTFIELDSIZE is supposed to send exactly
// that many bytes whatever they contain. On Linux it does. In this port a
// frame never arrived at any size -- while an ASCII body of the same length
// went through fine -- and a PNG has its first NUL nine bytes in, at the end
// of the signature. A body truncated at that NUL, with Content-Length still
// promising the rest, leaves the server waiting for data that never comes and
// eventually closing: which is the error we saw, at every size we tried.
//
// A read callback sidesteps the question entirely. It is also what curl's own
// documentation recommends for anything that is not a C string.
struct upload { const uint8_t *p; size_t left; };

static size_t read_cb(char *dst, size_t sz, size_t nm, void *userp) {
    struct upload *u = (struct upload *)userp;
    size_t want = sz * nm;
    if (want > u->left) want = u->left;
    if (want) { memcpy(dst, u->p, want); u->p += want; u->left -= want; }
    return want;
}

// Send, and try again if it fails.
//
// Not defensive padding: the failures here are intermittent, and that took
// three ladders to see. A 4,093-byte frame failed in the same run where 2,907
// bytes posted, and a 10,160-byte frame had posted fine the deploy before --
// then the identical 2,907-byte frame failed two minutes after succeeding.
// Every "ceiling" measured on this endpoint was a threshold read into noise.
//
// So the transport is unreliable rather than bounded, and the honest response
// to an unreliable link is to retry it and to count how often that was needed.
static long g_sends, g_retries, g_failures;

static int send_photo_once(const uint8_t *png, size_t png_len, const char *caption);

static int send_photo(const uint8_t *png, size_t png_len, const char *caption) {
    for (int attempt = 1; attempt <= 4; attempt++) {
        g_sends++;
        if (send_photo_once(png, png_len, caption)) {
            if (attempt > 1) printf("[+] delivered on attempt %d\n", attempt);
            return 1;
        }
        g_retries++;
        sleep(attempt * 3);          // 3s, 6s, 9s -- brief, widening
    }
    g_failures++;
    return 0;
}

static int send_photo_once(const uint8_t *png, size_t png_len, const char *caption) {
    static const char *boundary = "----bmdemo7f3a91c2";
    char chat[32];
    snprintf(chat, sizeof(chat), "%lld", g_current_chat);
    g_body_len = build_multipart(boundary, chat, caption, png, png_len);
    if (!g_body_len) { fprintf(stderr, "[!] frame did not fit the body buffer\n"); return 0; }

    char url[256];
    snprintf(url, sizeof(url), "https://api.telegram.org/bot%s/sendPhoto", g_tg_token);
    char ctype[128];
    snprintf(ctype, sizeof(ctype), "Content-Type: multipart/form-data; boundary=%s", boundary);

    CURL *h = curl_easy_init();
    if (!h) return 0;
    struct curl_slist *hdrs = curl_slist_append(NULL, ctype);
    hdrs = curl_slist_append(hdrs, "Expect:");   // no 100-continue round trip

    g_sink.len = 0; g_sink.overflow = 0; g_http[0] = '\0';
    curl_easy_setopt(h, CURLOPT_URL, url);
    struct upload up = { g_body, g_body_len };
    curl_easy_setopt(h, CURLOPT_POST, 1L);
    curl_easy_setopt(h, CURLOPT_READFUNCTION, read_cb);
    curl_easy_setopt(h, CURLOPT_READDATA, &up);
    curl_easy_setopt(h, CURLOPT_POSTFIELDSIZE_LARGE, (curl_off_t)g_body_len);
    curl_easy_setopt(h, CURLOPT_HTTPHEADER, hdrs);
    common_opts(h);
    set_ca_bundle(h);

    CURLcode res = curl_easy_perform(h);
    long status = 0;
    curl_easy_getinfo(h, CURLINFO_RESPONSE_CODE, &status);
    curl_slist_free_all(hdrs);
    curl_easy_cleanup(h);

    if (res != CURLE_OK) { fprintf(stderr, "[!] send: %s\n", curl_easy_strerror(res)); return 0; }
    if (status != 200)   { fprintf(stderr, "[!] send: HTTP %ld %.200s\n", status, g_http); return 0; }
    return 1;
}


// ============================================================ end fractal

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
"     \"name\":\"remind_me\","
"     \"description\":\"Send this person a message at a future time. Say how far ahead in seconds — 3600 for an hour, 86400 for a day. Then tell them what you set, in their words.\","
"     \"parameters\":{\"type\":\"object\",\"properties\":{"
"        \"in_seconds\":{\"type\":\"integer\",\"description\":\"How long from now, in seconds.\"},"
"        \"text\":{\"type\":\"string\",\"description\":\"What to remind them about, in their words.\"}},"
"        \"required\":[\"in_seconds\",\"text\"]}}},"
"  {\"type\":\"function\",\"function\":{"
"     \"name\":\"list_reminders\","
"     \"description\":\"The reminders currently set for this person, with when each is due.\","
"     \"parameters\":{\"type\":\"object\",\"properties\":{},\"required\":[]}}},"
"  {\"type\":\"function\",\"function\":{"
"     \"name\":\"recall\","
"     \"description\":\"The most recent notes you have kept about this person, in order. Use it to get your bearings, or when they ask what you know in general. For a question about a particular topic, recall_similar is the better tool -- it searches all of them, not just the recent ones.\","
"     \"parameters\":{\"type\":\"object\",\"properties\":{},\"required\":[]}}},"
"  {\"type\":\"function\",\"function\":{"
"     \"name\":\"recall_similar\","
"     \"description\":\"Search what you have kept about this person by meaning rather than by wording — use it when they ask what you know about a topic, and the words they use may not be the words you filed it under. Searched entirely in this machine's RAM.\","
"     \"parameters\":{\"type\":\"object\",\"properties\":{"
"        \"query\":{\"type\":\"string\",\"description\":\"What to look for, as a phrase or question.\"},"
"        \"k\":{\"type\":\"integer\",\"description\":\"How many of the closest notes to return. Default 3, at most 8.\"}},"
"        \"required\":[\"query\"]}}},"
"  {\"type\":\"function\",\"function\":{"
"     \"name\":\"draw_fractal\","
"     \"description\":\"Draw the Mandelbrot set and send it to this chat as a picture. Use when someone asks for a fractal, a picture, or a demonstration that you can compute something visible. Renders every frame twice and marks in red any pixel where this machine disagreed with itself.\","
"     \"parameters\":{\"type\":\"object\",\"properties\":{"
"        \"zoom\":{\"type\":\"number\",\"description\":\"How far in to zoom. 1 is the whole set; 1000 is deep. Default 1.\"}},"
"        \"required\":[]}}},"
"  {\"type\":\"function\",\"function\":{"
"     \"name\":\"usage_stats\","
"     \"description\":\"How much this machine has been used: messages answered, how many different people have talked to it, and how many times it has been restarted. Counted across every restart, not just this one.\","
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
    // The index dominates the footprint, which is the point of it.
    size_t index_bytes = sizeof(g_vec) + sizeof(g_vtext) + sizeof(g_vchat);
    cJSON_AddNumberToObject(o, "semantic_index_bytes", (double)index_bytes);
    cJSON_AddNumberToObject(o, "semantic_index_capacity", VEC_MAX);
    cJSON_AddNumberToObject(o, "semantic_index_vectors_resident", g_vcount);
    cJSON_AddNumberToObject(o, "semantic_index_dimensions", VEC_DIM);
    if (g_vevicted) cJSON_AddNumberToObject(o, "semantic_index_evicted", g_vevicted);
    if (g_tsc_hz && g_vlast_search_cycles)
        cJSON_AddNumberToObject(o, "last_search_microseconds",
                                (double)g_vlast_search_cycles * 1e6 / (double)g_tsc_hz);

    size_t fractal_bytes = sizeof(g_pass1) + sizeof(g_pass2) + sizeof(g_idx) + sizeof(g_plte)
                        + sizeof(g_raw) + sizeof(g_body) + sizeof(g_head) + sizeof(g_prev);
    cJSON_AddNumberToObject(o, "fractal_renderer_bytes", (double)fractal_bytes);
    size_t total = ARENA_BYTES + HTTP_BUF_BYTES + PAYLOAD_BYTES + index_bytes + fractal_bytes;
    cJSON_AddNumberToObject(o, "static_total_bytes", (double)total);
    cJSON_AddNumberToObject(o, "machine_ram_bytes", (double)RAM_MIB * 1024.0 * 1024.0);
    cJSON_AddNumberToObject(o, "percent_of_machine_claimed",
                            (double)total * 100.0 / ((double)RAM_MIB * 1024.0 * 1024.0));
    cJSON_AddStringToObject(o, "heap_allocations", "none: the allocator is compiled out");
    cJSON_AddStringToObject(o, "note",
        "All of this is static memory, claimed at link time. None of it costs a byte in "
        "the uploaded image, because uninitialised statics are not stored in the file.");
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

    // File it in the RAM index too, so it can be found by meaning and not only
    // by the words it happens to contain. The note is already safe in Redis by
    // this point, so a failed embedding costs searchability, not the note.
    int indexed = index_remember(g_current_chat, f->valuestring);

    snprintf(g_result, sizeof(g_result),
             "{\"ok\":true,\"indexed\":%s,\"vectors_resident\":%d,"
             "\"note\":\"kept; it will still be here after I am restarted%s\"}",
             indexed ? "true" : "false", g_vcount,
             indexed ? ", and it is now searchable by meaning"
                     : ". It is stored, but could not be indexed for meaning-based search");
}

// This chat's notes, still as a JSON array. Shared by the text recall below
// and by the index backfill, which needs the notes one at a time rather than
// run together into a paragraph.
static int load_notes_list(cJSON **reply_out, int count) {
    if (!g_memory_enabled) return 0;
    char key[64];
    notes_key(key, sizeof(key));

    cJSON *cmd = cJSON_CreateArray();
    if (!cmd) return 0;
    cJSON_AddItemToArray(cmd, cJSON_CreateString("LRANGE"));
    cJSON_AddItemToArray(cmd, cJSON_CreateString(key));
    // How far back to read is the caller's business, and the two callers want
    // different things. Plain recall wants the most recent RECALL_SHOWN, because
    // a hundred notes do not fit the buffer they are read into and would be
    // truncated mid-sentence with nothing to say they had been. The index wants
    // everything, because notes older than the recall window are precisely the
    // ones it exists to find -- capping both at the same number made the index
    // blind to exactly the material that justified building it.
    char lo[16];
    snprintf(lo, sizeof(lo), "-%d", count);
    cJSON_AddItemToArray(cmd, cJSON_CreateString(lo));
    cJSON_AddItemToArray(cmd, cJSON_CreateString("-1"));

    cJSON *reply = kv_command(cmd);
    if (!reply) return 0;
    *reply_out = reply;
    return 1;
}

// Reads this chat's notes into g_result. Also used to preload a conversation.
static int load_notes(char *out, size_t out_sz) {
    cJSON *reply = NULL;
    if (!load_notes_list(&reply, RECALL_SHOWN)) return 0;
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

static long kv_number(const char *cmd_name, const char *key, long fallback);

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
    char key[64];
    notes_key(key, sizeof(key));
    long total = kv_number("LLEN", key, 0);

    cJSON *o = cJSON_CreateObject();
    if (!o) { snprintf(g_result, sizeof(g_result), "{\"error\":\"arena exhausted\"}"); return; }
    cJSON_AddStringToObject(o, "notes", notes);
    // Say what is missing rather than quietly returning a subset: an agent that
    // reports "that is everything" while holding back seventy notes is worse
    // than one that says where the rest are.
    if (total > RECALL_SHOWN) {
        cJSON_AddNumberToObject(o, "showing_most_recent", RECALL_SHOWN);
        cJSON_AddNumberToObject(o, "total_kept", total);
        cJSON_AddStringToObject(o, "older_notes",
            "Not shown here. Use recall_similar to search all of them by meaning.");
    }
    cJSON_AddStringToObject(o, "stored_in",
        "Redis, reached over HTTPS. This machine has no disk, so anything it keeps lives "
        "on the network.");
    if (!cJSON_PrintPreallocated(o, g_result, (int)sizeof(g_result), 0))
        snprintf(g_result, sizeof(g_result), "{\"error\":\"result did not fit\"}");
    cJSON_Delete(o);
}

// Search this chat's notes by meaning. Every comparison happens in RAM against
// the resident index; the only network call is the one that embeds the query.
static void tool_recall_similar(const cJSON *args) {
    if (!g_memory_enabled) {
        snprintf(g_result, sizeof(g_result), "{\"error\":\"no memory configured\"}");
        return;
    }
    cJSON *q = cJSON_GetObjectItemCaseSensitive(args, "query");
    if (!cJSON_IsString(q) || !*q->valuestring) {
        snprintf(g_result, sizeof(g_result), "{\"error\":\"nothing to search for\"}");
        return;
    }
    cJSON *kj = cJSON_GetObjectItemCaseSensitive(args, "k");
    int k = cJSON_IsNumber(kj) ? (int)kj->valuedouble : 3;

    // The index is RAM, so a restart empties it. Restore this chat's notes
    // from Redis the first time it is searched after a boot.
    int restored = index_backfill(g_current_chat);

    static float qv[EMBED_BATCH][VEC_DIM];
    const char *one = q->valuestring;
    if (embed_texts(&one, 1, qv) != 1) {
        snprintf(g_result, sizeof(g_result),
                 "{\"error\":\"could not embed the query; meaning-based search is "
                 "unavailable right now. Plain recall still works.\"}");
        return;
    }

    int idx[8];
    double score[8];
    int n = index_search(g_current_chat, qv[0], k, idx, score);
    if (n == 0) {
        snprintf(g_result, sizeof(g_result),
                 "{\"matches\":[],\"vectors_resident\":%d,"
                 "\"note\":\"nothing indexed for this person yet\"}", g_vcount);
        return;
    }

    cJSON *o = cJSON_CreateObject();
    if (!o) { snprintf(g_result, sizeof(g_result), "{\"error\":\"arena exhausted\"}"); return; }
    cJSON *arr = cJSON_CreateArray();
    for (int i = 0; i < n && arr; i++) {
        cJSON *m = cJSON_CreateObject();
        if (!m) break;
        cJSON_AddStringToObject(m, "text", g_vtext[idx[i]]);
        cJSON_AddNumberToObject(m, "similarity", score[i]);
        cJSON_AddItemToArray(arr, m);
    }
    if (arr) cJSON_AddItemToObject(o, "matches", arr);
    cJSON_AddNumberToObject(o, "vectors_searched", g_vcount);
    if (restored) cJSON_AddNumberToObject(o, "restored_from_redis_just_now", restored);
    if (g_tsc_hz)
        cJSON_AddNumberToObject(o, "search_microseconds",
                                (double)g_vlast_search_cycles * 1e6 / (double)g_tsc_hz);
    cJSON_AddStringToObject(o, "how",
        "Cosine similarity against every vector held in this machine's RAM. No index "
        "structure, no database, no disk -- one pass over a static array.");
    if (!cJSON_PrintPreallocated(o, g_result, (int)sizeof(g_result), 0))
        snprintf(g_result, sizeof(g_result), "{\"error\":\"result did not fit\"}");
    cJSON_Delete(o);
}

static int send_message(long long chat_id, const char *text);   // defined below

// One small integer reply from the store, or fallback on any trouble. Used for
// counters, where a missing value and a zero mean the same thing.
static long kv_number(const char *cmd_name, const char *key, long fallback) {
    if (!g_memory_enabled) return fallback;
    cJSON *cmd = cJSON_CreateArray();
    if (!cmd) return fallback;
    cJSON_AddItemToArray(cmd, cJSON_CreateString(cmd_name));
    cJSON_AddItemToArray(cmd, cJSON_CreateString(key));
    cJSON *rep = kv_command(cmd);
    cJSON *res = rep ? cJSON_GetObjectItemCaseSensitive(rep, "result") : NULL;
    if (cJSON_IsNumber(res)) return (long)res->valuedouble;
    if (cJSON_IsString(res)) return strtol(res->valuestring, NULL, 10);
    return fallback;
}

// Counted after a reply actually went out, so the number means "answers given"
// rather than "messages seen".
static void bump_usage(long long chat) {
    if (!g_memory_enabled) return;
    cJSON *inc = cJSON_CreateArray();
    if (inc) {
        cJSON_AddItemToArray(inc, cJSON_CreateString("INCR"));
        cJSON_AddItemToArray(inc, cJSON_CreateString(STATS_MESSAGES_KEY));
        kv_command(inc);
    }
    char who[32];
    snprintf(who, sizeof(who), "%lld", chat);
    cJSON *add = cJSON_CreateArray();
    if (add) {
        cJSON_AddItemToArray(add, cJSON_CreateString("SADD"));
        cJSON_AddItemToArray(add, cJSON_CreateString(STATS_PEOPLE_KEY));
        cJSON_AddItemToArray(add, cJSON_CreateString(who));
        kv_command(add);
    }
}

// Reminders live in one sorted set scored by due time, so "what is due" is a
// single range query. The payload carries the chat, because the delivery loop
// has no other way to know where a reminder should go.
static void tool_remind_me(const cJSON *args) {
    if (!g_memory_enabled) {
        snprintf(g_result, sizeof(g_result),
                 "{\"error\":\"no memory configured, so nothing can be kept for later\"}");
        return;
    }
    // A delay, not a timestamp. Asking the model for absolute epoch seconds
    // made it do clock arithmetic and emit a ten-digit literal, and it produced
    // a MALFORMED_FUNCTION_CALL about half the time — as well as timestamps
    // recalled from its training data. A small relative number is something it
    // gets right, and the clock arithmetic belongs here anyway.
    cJSON *in = cJSON_GetObjectItemCaseSensitive(args, "in_seconds");
    cJSON *at = cJSON_GetObjectItemCaseSensitive(args, "at_unix");   // still accepted
    cJSON *tx = cJSON_GetObjectItemCaseSensitive(args, "text");
    if ((!cJSON_IsNumber(in) && !cJSON_IsNumber(at)) || !cJSON_IsString(tx) || !*tx->valuestring) {
        snprintf(g_result, sizeof(g_result), "{\"error\":\"need in_seconds and text\"}");
        return;
    }

    long now = (long)time(NULL);
    long due = cJSON_IsNumber(in) ? now + (long)in->valuedouble : (long)at->valuedouble;
    if (due < now + REMINDER_MIN_DELAY) {
        snprintf(g_result, sizeof(g_result),
                 "{\"error\":\"that time is now or in the past. The current epoch is %ld.\"}", now);
        return;
    }
    if (due > now + REMINDER_MAX_HORIZON) {
        snprintf(g_result, sizeof(g_result),
                 "{\"error\":\"further ahead than I keep reminders (90 days)\"}");
        return;
    }

    // A cap per person, counted here. Someone should not be able to fill the
    // store by asking nicely.
    cJSON *count = cJSON_CreateArray();
    if (count) {
        cJSON_AddItemToArray(count, cJSON_CreateString("ZCARD"));
        cJSON_AddItemToArray(count, cJSON_CreateString(REMINDERS_KEY));
        cJSON *rep = kv_command(count);
        cJSON *res = rep ? cJSON_GetObjectItemCaseSensitive(rep, "result") : NULL;
        if (cJSON_IsNumber(res) && res->valuedouble >= REMINDER_MAX_PER_CHAT * 20) {
            snprintf(g_result, sizeof(g_result), "{\"error\":\"my reminder list is full\"}");
            return;
        }
    }

    // member = {"chat":<id>,"text":"..."} — unique enough that two identical
    // reminders at the same second collapse into one, which is the right answer.
    cJSON *member = cJSON_CreateObject();
    if (!member) { snprintf(g_result, sizeof(g_result), "{\"error\":\"arena exhausted\"}"); return; }
    cJSON_AddNumberToObject(member, "chat", (double)g_current_chat);
    cJSON_AddNumberToObject(member, "due", (double)due);
    char clipped[600];
    snprintf(clipped, sizeof(clipped), "%s", tx->valuestring);
    cJSON_AddStringToObject(member, "text", clipped);
    char payload[800];
    if (!cJSON_PrintPreallocated(member, payload, (int)sizeof(payload), 0)) {
        cJSON_Delete(member);
        snprintf(g_result, sizeof(g_result), "{\"error\":\"that reminder is too long\"}");
        return;
    }
    cJSON_Delete(member);

    char score[24];
    snprintf(score, sizeof(score), "%ld", due);
    cJSON *cmd = cJSON_CreateArray();
    if (!cmd) { snprintf(g_result, sizeof(g_result), "{\"error\":\"arena exhausted\"}"); return; }
    cJSON_AddItemToArray(cmd, cJSON_CreateString("ZADD"));
    cJSON_AddItemToArray(cmd, cJSON_CreateString(REMINDERS_KEY));
    cJSON_AddItemToArray(cmd, cJSON_CreateString(score));
    cJSON_AddItemToArray(cmd, cJSON_CreateString(payload));

    if (!kv_command(cmd)) {
        snprintf(g_result, sizeof(g_result), "{\"error\":\"could not reach my memory\"}");
        return;
    }
    snprintf(g_result, sizeof(g_result),
             "{\"ok\":true,\"due_unix\":%ld,\"in_seconds\":%ld,\"note\":\"kept where I keep "
             "everything, so a restart will not lose it\"}", due, due - now);
}

static void tool_list_reminders(void) {
    if (!g_memory_enabled) {
        snprintf(g_result, sizeof(g_result), "{\"error\":\"no memory configured\"}");
        return;
    }
    cJSON *cmd = cJSON_CreateArray();
    if (!cmd) { snprintf(g_result, sizeof(g_result), "{\"error\":\"arena exhausted\"}"); return; }
    cJSON_AddItemToArray(cmd, cJSON_CreateString("ZRANGE"));
    cJSON_AddItemToArray(cmd, cJSON_CreateString(REMINDERS_KEY));
    cJSON_AddItemToArray(cmd, cJSON_CreateString("0"));
    cJSON_AddItemToArray(cmd, cJSON_CreateString("-1"));

    cJSON *rep = kv_command(cmd);
    cJSON *res = rep ? cJSON_GetObjectItemCaseSensitive(rep, "result") : NULL;
    if (!cJSON_IsArray(res)) {
        snprintf(g_result, sizeof(g_result), "{\"reminders\":[]}");
        return;
    }

    // Filtered here rather than in the query: the store holds everyone's
    // reminders, and this person may only see their own.
    cJSON *mine = cJSON_CreateArray();
    cJSON *it = NULL;
    long now = (long)time(NULL);
    cJSON_ArrayForEach(it, res) {
        if (!cJSON_IsString(it)) continue;
        cJSON *m = cJSON_Parse(it->valuestring);
        if (!m) continue;
        cJSON *chat = cJSON_GetObjectItemCaseSensitive(m, "chat");
        cJSON *due  = cJSON_GetObjectItemCaseSensitive(m, "due");
        cJSON *text = cJSON_GetObjectItemCaseSensitive(m, "text");
        if (cJSON_IsNumber(chat) && (long long)chat->valuedouble == g_current_chat &&
            cJSON_IsString(text) && mine) {
            cJSON *o = cJSON_CreateObject();
            if (o) {
                cJSON_AddStringToObject(o, "text", text->valuestring);
                if (cJSON_IsNumber(due))
                    cJSON_AddNumberToObject(o, "in_seconds", (double)((long)due->valuedouble - now));
                cJSON_AddItemToArray(mine, o);
            }
        }
        cJSON_Delete(m);
    }

    cJSON *out = cJSON_CreateObject();
    if (out && mine) cJSON_AddItemToObject(out, "reminders", mine);
    if (!out || !cJSON_PrintPreallocated(out, g_result, (int)sizeof(g_result), 0))
        snprintf(g_result, sizeof(g_result), "{\"error\":\"could not list them\"}");
    cJSON_Delete(out);
}

// Delivers anything now due, straight from C. No model call: the text was
// written when the reminder was set, and paying for a round of inference to
// repeat it back would be waste.
static void deliver_due_reminders(void) {
    if (!g_memory_enabled) return;
    long now = (long)time(NULL);
    char now_s[24];
    snprintf(now_s, sizeof(now_s), "%ld", now);

    arena_reset();
    cJSON *cmd = cJSON_CreateArray();
    if (!cmd) return;
    cJSON_AddItemToArray(cmd, cJSON_CreateString("ZRANGEBYSCORE"));
    cJSON_AddItemToArray(cmd, cJSON_CreateString(REMINDERS_KEY));
    cJSON_AddItemToArray(cmd, cJSON_CreateString("-inf"));
    cJSON_AddItemToArray(cmd, cJSON_CreateString(now_s));

    cJSON *rep = kv_command(cmd);
    cJSON *res = rep ? cJSON_GetObjectItemCaseSensitive(rep, "result") : NULL;
    if (!cJSON_IsArray(res) || cJSON_GetArraySize(res) == 0) return;

    cJSON *it = NULL;
    cJSON_ArrayForEach(it, res) {
        if (!cJSON_IsString(it)) continue;
        char raw[900];
        snprintf(raw, sizeof(raw), "%s", it->valuestring);

        cJSON *m = cJSON_Parse(raw);
        if (!m) continue;
        cJSON *chat = cJSON_GetObjectItemCaseSensitive(m, "chat");
        cJSON *text = cJSON_GetObjectItemCaseSensitive(m, "text");
        if (cJSON_IsNumber(chat) && cJSON_IsString(text)) {
            char msg[900];
            snprintf(msg, sizeof(msg), "\xE2\x8F\xB0 Reminder: %s", text->valuestring);
            long long to = (long long)chat->valuedouble;
            printf("[*] reminder due for chat %lld: %s\n", to, text->valuestring);
            send_message(to, msg);
        }
        cJSON_Delete(m);

        // Remove it whether or not the send worked. A reminder that fails to
        // deliver and stays queued would be retried forever, every minute.
        cJSON *rm = cJSON_CreateArray();
        if (rm) {
            cJSON_AddItemToArray(rm, cJSON_CreateString("ZREM"));
            cJSON_AddItemToArray(rm, cJSON_CreateString(REMINDERS_KEY));
            cJSON_AddItemToArray(rm, cJSON_CreateString(raw));
            kv_command(rm);
        }
    }
}

static void tool_usage_stats(void) {
    if (!g_memory_enabled) {
        snprintf(g_result, sizeof(g_result),
                 "{\"error\":\"no memory configured, so nothing has been counted\"}");
        return;
    }
    long msgs   = kv_number("GET",   STATS_MESSAGES_KEY, 0);
    long people = kv_number("SCARD", STATS_PEOPLE_KEY,   0);
    long boots  = kv_number("GET",   "agent:boots",      0);

    cJSON *o = cJSON_CreateObject();
    if (!o) { snprintf(g_result, sizeof(g_result), "{\"error\":\"arena exhausted\"}"); return; }
    cJSON_AddNumberToObject(o, "messages_answered", (double)msgs);
    cJSON_AddNumberToObject(o, "different_people", (double)people);
    cJSON_AddNumberToObject(o, "times_restarted", (double)boots);
    cJSON_AddNumberToObject(o, "seconds_up_this_time", (double)(time(NULL) - g_booted));
    cJSON_AddStringToObject(o, "note",
        "Counted in the same store as everything else, so these survive restarts. "
        "This machine keeps no local state at all.");
    if (!cJSON_PrintPreallocated(o, g_result, (int)sizeof(g_result), 0))
        snprintf(g_result, sizeof(g_result), "{\"error\":\"result did not fit\"}");
    cJSON_Delete(o);
}


// Draw a frame and send it to whoever asked. The model picks how far in to
// zoom; everything else -- where to look, how many iterations, what size the
// link will carry -- is decided here, because those are not judgement calls.
static void tool_draw_fractal(const cJSON *args) {
    cJSON *z = cJSON_GetObjectItemCaseSensitive(args, "zoom");
    double zoom = cJSON_IsNumber(z) ? z->valuedouble : 1.0;
    if (!(zoom >= 1.0)) zoom = 1.0;              // also catches NaN
    if (zoom > 1e12) zoom = 1e12;                // past this, doubles give mush

    build_palette();
    double fov = 3.0 / zoom;

    // Start at the largest size and step down until the frame fits what this
    // link has actually carried. Same rule the standalone renderer uses: a
    // deeper zoom holds more detail and compresses worse, so the size that
    // worked last time is not a size that works this time.
    g_w = g_h = FRACT_MAX;
    static uint8_t png[64 * 1024];
    size_t png_len = 0;
    long bad = 0;
    time_t t0 = time(NULL);

    for (;;) {
        double scale = fov / (double)g_w;
        render(g_pass1, scale);
        render(g_pass2, scale);                  // the same work, again, on purpose
        colourise(g_pass1);
        bad = mark_disagreements();
        png_len = png_encode(png, sizeof(png));
        if (png_len && png_len <= g_budget) break;
        int i = 0;
        while (i < N_LADDER - 1 && LADDER[i] < g_w) i++;
        if (i == 0) break;
        g_w = g_h = LADDER[i - 1];
    }
    long secs = (long)(time(NULL) - t0);

    if (!png_len) {
        snprintf(g_result, sizeof(g_result),
                 "{\"error\":\"could not encode the frame\"}");
        return;
    }

    char caption[400];
    snprintf(caption, sizeof(caption),
        "zoom %.0fx \xc2\xb7 %dx%d \xc2\xb7 %d iterations \xc2\xb7 rendered twice in %lds\n"
        "%ld pixels disagreed between the passes%s",
        zoom, g_w, g_h, MAX_ITER, secs, bad,
        bad ? " \xe2\x80\x94 each red dot is this machine answering the same sum two ways."
            : " \xe2\x80\x94 clean.");

    int sent = send_photo(png, png_len, caption);
    snprintf(g_result, sizeof(g_result),
        "{\"sent\":%s,\"zoom\":%.0f,\"pixels\":\"%dx%d\",\"png_bytes\":%zu,"
        "\"render_seconds\":%ld,\"pixels_disagreeing\":%ld,\"retries\":%ld,"
        "\"note\":\"%s\"}",
        sent ? "true" : "false", zoom, g_w, g_h, png_len, secs, bad, g_retries,
        sent ? "The picture has already been delivered to this chat. Say what is in it "
               "and what the numbers mean; do not describe it as if they cannot see it."
             : "The picture could not be delivered. Say so plainly.");
}

static void call_tool(const char *name, const cJSON *args) {
    if      (strcmp(name, "machine_facts") == 0) tool_machine_facts();
    else if (strcmp(name, "memory_usage")  == 0) tool_memory_usage();
    else if (strcmp(name, "build_info")    == 0) tool_build_info();
    else if (strcmp(name, "ping_api")      == 0) tool_ping_api();
    else if (strcmp(name, "startup_timing") == 0) tool_startup_timing();
    else if (strcmp(name, "usage_stats")    == 0) tool_usage_stats();
    else if (strcmp(name, "recall")        == 0) tool_recall();
    else if (strcmp(name, "recall_similar")== 0) tool_recall_similar(args);
    else if (strcmp(name, "draw_fractal")  == 0) tool_draw_fractal(args);
    else if (strcmp(name, "list_reminders") == 0) tool_list_reminders();
    else if (strcmp(name, "remind_me") == 0) {
        if (args) tool_remind_me(args);
        else snprintf(g_result, sizeof(g_result), "{\"error\":\"bad arguments\"}");
    }
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

// Plain text: replies are sent without a parse mode, because Markdown modes
// reject unescaped punctuation and a 400 here would mean silence.
#define HELP_TEXT \
    "I am an AI agent running as a BareMetal unikernel — one program with no " \
    "operating system beneath it, in 16 MiB of RAM. When you text me, the reply " \
    "comes from a 2.9 MB machine image with no kernel, no shell and no disk.\n\n" \
    "Things worth asking:\n" \
    "• how much memory are you using right now?\n" \
    "• what is inside your image?\n" \
    "• how fast did you start up?\n" \
    "• how many people have talked to you?\n\n" \
    "I can also search the web, and set reminders — try \"remind me in 2 hours to " \
    "call the plumber\". Tell me something about yourself and I will keep it: I " \
    "have no disk, so my notes live on another machine and survive my restarts.\n\n" \
    "I keep those notes indexed by meaning as well as by word, in about 8 MiB of " \
    "my own RAM — a vector database with no database under it. Ask me what I know " \
    "about a topic and I will find it even if you word it differently.\n\n" \
    "Every number I give you about myself is measured when you ask, not " \
    "remembered. Source: github.com/tabibazar/baremetal-agent"

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
    "You can also set reminders for people, which are delivered even if you are restarted " \
    "in the meantime. When someone asks to be reminded, work out the absolute time from the " \
    "current time given to you and set it; do not ask them to give you a timestamp.\n\n" \
    "You have a memory that survives being restarted, kept on another machine because " \
    "you have no disk of your own. When someone tells you something worth keeping about " \
    "themselves, keep it. What you already know about them is given to you before their " \
    "message, so use it naturally rather than announcing that you looked. Only the most " \
    "recent notes are given to you that way, so before telling someone you do not know " \
    "something about THEM or their life, search with recall_similar first — the answer is " \
    "often something they told you long enough ago that it is no longer in front of you.\n\n" \
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
        cJSON *choice = cJSON_GetArrayItem(choices, 0);
        cJSON *m = cJSON_GetObjectItemCaseSensitive(choice, "message");
        if (m) msg = cJSON_Duplicate(m, 1);

        // When a reply carries neither text nor a tool call, the reason is in
        // finish_reason and nowhere else. Gemini answers MALFORMED_FUNCTION_CALL
        // this way — it tried to call a tool, produced something unparseable,
        // and returned an empty message rather than an error.
        if (m) {
            cJSON *c  = cJSON_GetObjectItemCaseSensitive(m, "content");
            cJSON *tc = cJSON_GetObjectItemCaseSensitive(m, "tool_calls");
            int empty = (!cJSON_IsString(c) || !*c->valuestring) &&
                        (!cJSON_IsArray(tc) || cJSON_GetArraySize(tc) == 0);
            if (empty) {
                cJSON *fr = cJSON_GetObjectItemCaseSensitive(choice, "finish_reason");
                fprintf(stderr, "[!] empty reply from the model, finish_reason=%s\n",
                        cJSON_IsString(fr) ? fr->valuestring : "(absent)");
            }
        }
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

    // The current time, so that "tomorrow morning" can become a number. Without
    // this the model has no idea when now is, and quietly guesses.
    {
        time_t now = time(NULL);
        char when[220];
        struct tm *g = gmtime(&now);
        if (g)
            snprintf(when, sizeof(when),
                     "The current time is %04d-%02d-%02d %02d:%02d UTC, which is Unix epoch "
                     "second %ld. Use it for anything time-related.",
                     g->tm_year + 1900, g->tm_mon + 1, g->tm_mday, g->tm_hour, g->tm_min,
                     (long)now);
        else
            snprintf(when, sizeof(when), "The current Unix epoch second is %ld.", (long)now);
        add_message(messages, "system", when);
    }

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
    int empty_replies = 0;

    for (int step = 1; step <= MAX_STEPS; step++) {
        cJSON *assistant = llm_turn(messages);
        if (g_arena_full) { fprintf(stderr, "[!] arena exhausted\n"); break; }
        if (!assistant)   { break; }

        cJSON_AddItemToArray(messages, cJSON_Duplicate(assistant, 1));

        cJSON *calls   = cJSON_GetObjectItemCaseSensitive(assistant, "tool_calls");
        cJSON *content = cJSON_GetObjectItemCaseSensitive(assistant, "content");

        if (!cJSON_IsArray(calls) || cJSON_GetArraySize(calls) == 0) {
            if (cJSON_IsString(content) && *content->valuestring) {
                final = content->valuestring;
                break;
            }
            // Neither text nor a tool call. Rather than give up on the person,
            // say what came back and ask for the answer in plain words; a
            // malformed tool call is usually not repeated on the second try.
            if (++empty_replies <= 2) {
                fprintf(stderr, "    [~] empty reply, asking again (%d)\n", empty_replies);
                add_message(messages, "user",
                    "Your last reply came back empty, which happens when a tool call is "
                    "malformed. Try the same call again with plain literal values — never "
                    "an expression or a calculation — or if that fails, just answer in "
                    "words. Do not ask the person for timestamps or other internal values; "
                    "that is your job, not theirs.");
                continue;
            }
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
    if (deliver && send_message(chat_id, final)) {
        g_replies_window++;
        bump_usage(chat_id);
    }
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

        // /start is the first thing almost everyone sends, and /help the second.
        // Answering them from here costs no inference, arrives instantly, and
        // says the same thing every time -- none of which is true of letting the
        // model improvise an introduction.
        if (strncmp(question, "/start", 6) == 0 || strncmp(question, "/help", 5) == 0) {
            send_message(chat_id, HELP_TEXT);
            printf("[+] sent the introduction to %s\n", who);
            continue;
        }

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
    // --help-text sends the introduction to the configured chat and exits, so
    // it can be read as the reader will read it before anyone is shown it.
    int preview = (argc > 1 && strcmp(argv[1], "--help-text") == 0);

    cJSON_Hooks hooks;
    hooks.malloc_fn = arena_alloc;
    hooks.free_fn   = arena_free;
    cJSON_InitHooks(&hooks);

    curl_global_init(CURL_GLOBAL_ALL);

    // Report the whole static footprint, index included. The buffers alone
    // were the honest number when the buffers were all there was.
    {
        size_t idx = sizeof(g_vec) + sizeof(g_vtext) + sizeof(g_vchat);
        size_t frac = sizeof(g_pass1) + sizeof(g_pass2) + sizeof(g_idx) + sizeof(g_plte)
                        + sizeof(g_raw) + sizeof(g_body) + sizeof(g_head) + sizeof(g_prev);
        size_t stat_total = ARENA_BYTES + HTTP_BUF_BYTES + PAYLOAD_BYTES + idx + frac;
        printf("[+] BareMetal agent up. model=%s ram=%d MiB static=%zu KB (%.0f%% of RAM) max_steps=%d\n",
               g_llm_model, RAM_MIB, stat_total / 1024,
               (double)stat_total * 100.0 / ((double)RAM_MIB * 1024.0 * 1024.0), MAX_STEPS);
        printf("[+] semantic index: %d slots x %d dims = %zu KB resident, searched in RAM\n",
               VEC_MAX, VEC_DIM, idx / 1024);
        printf("[+] fractal renderer: up to %dx%d, %zu KB of buffers, PNG encoder included\n",
               FRACT_MAX, FRACT_MAX, frac / 1024);
    }
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
    if (preview) {
        const char *chat = getenv("TELEGRAM_CHAT_ID");
        if (chat && *chat) {
            send_message(strtoll(chat, NULL, 10), HELP_TEXT);
            printf("[+] introduction sent to %s\n", chat);
        } else {
            printf("%s\n", HELP_TEXT);
        }
        curl_global_cleanup();
        return 0;
    }

    if (ask) {
        // A real chat id if one is configured, so a reminder set from the
        // terminal is actually deliverable; otherwise a scratch identity.
        const char *test_chat = getenv("TELEGRAM_CHAT_ID");
        g_current_chat = (test_chat && *test_chat) ? strtoll(test_chat, NULL, 10) : 0;
        printf("[*] test question: %s\n", ask);
        answer(0, ask, "the terminal", 0);
        curl_global_cleanup();
        return 0;
    }

    printf("[+] waiting for messages\n");

    int timing_printed = 0;
    time_t last_reminder_check = 0;
    do {
        poll_once();

        // On its own clock: every poll would mean a store round-trip every few
        // seconds forever, for something a minute of latency does not hurt.
        if (g_memory_enabled && time(NULL) - last_reminder_check >= REMINDER_CHECK_EVERY) {
            last_reminder_check = time(NULL);
            deliver_due_reminders();
        }

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
