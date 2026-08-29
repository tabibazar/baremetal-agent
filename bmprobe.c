// bmprobe.c -- how large an HTTPS request body can this machine actually send?
//
// The fractal renderer could not post a 787 KB image, and could not post a
// 263 KB one either. The same binary posts both successfully from Linux on the
// same host, so the limit is somewhere in this port's lwIP or mbedTLS rather
// than at the API on the other end. Guessing at it one redeploy at a time
// costs four minutes a guess.
//
// So: one deploy, a ladder of body sizes, and a line of output per rung
// saying whether it left the machine. Everything about each attempt is
// identical apart from the number of bytes, which is the only way the answer
// means anything.
//
// The target is our own key-value service rather than a public endpoint. It
// speaks HTTPS with a real certificate, it is under our control, and it will
// reject the junk we send with a 400 -- which is a perfectly good result here.
// An HTTP status of any kind means the body crossed the wire and a response
// came back; that is the whole question.
//
//   baremetal:  cp bmprobe.c BareMetal-App/ && ./1-build.sh bmprobe.c

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <unistd.h>
#include <curl/curl.h>

#ifndef PROBE_URL_DEFAULT
#define PROBE_URL_DEFAULT "PUT_PROBE_URL_HERE"
#endif

#define CA_BUNDLE_PATH "/etc/ssl/cacert.pem"
__attribute__((weak)) const unsigned char cacert_pem[1];
__attribute__((weak)) const unsigned int  cacert_pem_len;

// The ladder, in KB. Fine-grained where the answer is expected to be, because
// knowing it fails "somewhere under 256" is not worth a deploy.
static const int SIZES_KB[] = {
    1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 80, 96, 128, 160, 192, 256
};
#define N_SIZES ((int)(sizeof(SIZES_KB) / sizeof(SIZES_KB[0])))

static uint8_t g_body[256 * 1024 + 64];
static char    g_resp[8192];
static size_t  g_len;

static size_t write_cb(void *p, size_t sz, size_t nm, void *u) {
    (void)u;
    size_t n = sz * nm;
    if (g_len + n < sizeof(g_resp) - 1) {
        memcpy(g_resp + g_len, p, n);
        g_len += n;
        g_resp[g_len] = '\0';
    }
    return n;
}

// One POST of exactly `bytes` bytes. Returns the HTTP status, or 0 with the
// curl error printed -- the distinction between "rejected" and "never arrived"
// is the entire point.
static long try_size(const char *url, size_t bytes) {
    // Valid JSON of the requested length, so nothing downstream rejects it for
    // a reason unrelated to size.
    memset(g_body, 'a', bytes);
    g_body[0] = '[';
    g_body[1] = '"';
    g_body[bytes - 2] = '"';
    g_body[bytes - 1] = ']';

    CURL *h = curl_easy_init();
    if (!h) return -1;
    struct curl_slist *hdrs = curl_slist_append(NULL, "Content-Type: application/json");
    hdrs = curl_slist_append(hdrs, "Expect:");

    g_len = 0; g_resp[0] = '\0';
    curl_easy_setopt(h, CURLOPT_URL, url);
    curl_easy_setopt(h, CURLOPT_POST, 1L);
    curl_easy_setopt(h, CURLOPT_POSTFIELDS, g_body);
    curl_easy_setopt(h, CURLOPT_POSTFIELDSIZE, (long)bytes);
    curl_easy_setopt(h, CURLOPT_HTTPHEADER, hdrs);
    curl_easy_setopt(h, CURLOPT_WRITEFUNCTION, write_cb);
    curl_easy_setopt(h, CURLOPT_TIMEOUT, 60L);
    curl_easy_setopt(h, CURLOPT_NOSIGNAL, 1L);

    FILE *ca = fopen(CA_BUNDLE_PATH, "r");
    if (ca) { fclose(ca); curl_easy_setopt(h, CURLOPT_CAINFO, CA_BUNDLE_PATH); }
    else if (cacert_pem_len > 0) {
        struct curl_blob blob = { (void *)cacert_pem, cacert_pem_len, CURL_BLOB_NOCOPY };
        curl_easy_setopt(h, CURLOPT_CAINFO_BLOB, &blob);
    }

    time_t t0 = time(NULL);
    CURLcode res = curl_easy_perform(h);
    long status = 0;
    curl_easy_getinfo(h, CURLINFO_RESPONSE_CODE, &status);
    long secs = (long)(time(NULL) - t0);
    curl_slist_free_all(hdrs);
    curl_easy_cleanup(h);

    if (res != CURLE_OK) {
        printf("PROBE %6zu bytes  FAILED  %s (%ds)\n",
               bytes, curl_easy_strerror(res), (int)secs);
        return 0;
    }
    printf("PROBE %6zu bytes  ok      HTTP %ld (%ds)\n", bytes, status, (int)secs);
    return status;
}

static const char *env_or(const char *n, const char *fb) {
    const char *v = getenv(n);
    return (v && *v) ? v : fb;
}

int main(void) {
    setvbuf(stdout, NULL, _IOLBF, 0);
    curl_global_init(CURL_GLOBAL_DEFAULT);
    const char *url = env_or("PROBE_URL", PROBE_URL_DEFAULT);

    printf("[+] bmprobe up. target=%.40s\n", url);

    int last_ok = 0, first_bad = 0;
    for (int i = 0; i < N_SIZES; i++) {
        size_t bytes = (size_t)SIZES_KB[i] * 1024;
        long st = try_size(url, bytes);
        if (st > 0) {
            last_ok = SIZES_KB[i];
        } else if (!first_bad) {
            first_bad = SIZES_KB[i];
            // Do not stop: a single failure could be one bad connection, and
            // the shape of what follows says which it was.
        }
        sleep(2);   // a fresh connection each time, not a congested one
    }

    printf("[+] largest body that left this machine: %d KB\n", last_ok);
    if (first_bad) printf("[+] first size that did not: %d KB\n", first_bad);
    else           printf("[+] nothing on the ladder failed\n");

    curl_global_cleanup();
    return 0;
}
