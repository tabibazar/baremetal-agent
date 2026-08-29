// bmdemo.c -- a fractal renderer that runs as a BareMetal unikernel and posts
// its frames to Telegram.
//
// There is no GPU here, no graphics library, no image library, and no
// operating system. There is arithmetic and a socket. The PNG is encoded by
// hand a few hundred lines below, because linking libpng would have meant
// porting zlib, and a store-only deflate stream is legal PNG that any decoder
// in the world will accept.
//
// THE POINT OF RENDERING IT TWICE
//
// A Mandelbrot escape count is a pure function of the pixel. Render the same
// frame twice and a correct machine returns the same image, byte for byte, by
// construction. This machine does not: measurement on the same host, against a
// Linux control under the same hypervisor, put its rate of disagreeing with
// itself at roughly one result in four thousand.
//
//   https://github.com/tabibazar/unikernel-c/tree/main/docs/nanos-vs-baremetal
//
// So every frame is rendered twice and the two escape-count buffers compared.
// Where they differ, the pixel is painted red. The picture is the measurement:
// each red dot is the machine returning two different answers to the same
// question, and the caption counts them.
//
// It only ever calls sendPhoto. It never polls getUpdates, so it can share a
// bot token with the agent without the two of them stealing each other's
// messages.
//
// BUILD
//   linux:      gcc -O2 -o bmdemo bmdemo.c cjson/cJSON.c -lcurl -lm
//   baremetal:  cp bmdemo.c BareMetal-App/ && ./1-build.sh bmdemo.c

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include <curl/curl.h>

// ---------------------------------------------------------------- config

#ifndef TELEGRAM_TOKEN_DEFAULT
#define TELEGRAM_TOKEN_DEFAULT "PUT_BOT_TOKEN_HERE"
#endif
#ifndef DEMO_CHAT_DEFAULT
#define DEMO_CHAT_DEFAULT      "PUT_CHAT_ID_HERE"
#endif

// 480, not 512, and the 32 pixels are bought rather than lost. A probe walked
// a ladder of HTTPS body sizes from this platform: everything up to 262,144
// bytes leaves the machine, and the 512x512 frame was 263,824 bytes with its
// multipart wrapper -- over the line by 0.6%. At 480x480 a frame is about
// 231 KB, which clears it with 30 KB to spare.
//
//   PROBE 262144 bytes  ok      HTTP 401
//   [+] largest body that left this machine: 256 KB
//
// The limit is the port's, not the API's: the same binary posts a 787 KB frame
// from Linux on the same host without complaint.
#define W            480        // maximum; the working size is found at runtime
#define H            480
static int g_w = 480, g_h = 480;
#define MAX_ITER     600
#define FRAMES       0          // 0 = keep going until stopped
#define FRAME_GAP    120        // seconds between frames. Rendering is on top
                                // of this, and grows with the zoom, so the chat
                                // gets a frame every few minutes rather than a
                                // stream of them.
#define ZOOM_PER_FRAME 1.35     // each frame is this much deeper

// A point on the boundary worth falling into: the "seahorse valley" pinch.
// Chosen because detail keeps arriving as you descend rather than the image
// flattening into one colour.
#define TARGET_RE   (-0.743643887037151)
#define TARGET_IM   ( 0.131825904205330)

#define HTTP_TIMEOUT 120L
#define CA_BUNDLE_PATH "/etc/ssl/cacert.pem"

__attribute__((weak)) const unsigned char cacert_pem[1];
__attribute__((weak)) const unsigned int  cacert_pem_len;

// ---------------------------------------------------------------- memory
//
// All static, all claimed at link time, none of it in the uploaded image:
//
//   escape counts, pass 1   512 KB
//   escape counts, pass 2   512 KB
//   palette indices         256 KB   one byte per pixel, not three
//   encoded PNG             320 KB   stored deflate, so ~= the raw pixels
//   upload body             384 KB   multipart wrapper + the PNG inside it
//   -----------------------------
//   total                  1984 KB   of a 16 MiB machine
//
// The image is a palette PNG rather than truecolour, which is not a size trick
// so much as an honest description: it has at most 256 distinct colours by
// construction -- the escape bands, the interior, and the red. Sending three
// bytes per pixel was sending two bytes of nothing. It matters because a
// 787 KB body failed to leave this machine at all: the same code and the same
// frame post fine from Linux, so the ceiling is somewhere in lwIP or mbedTLS
// pushing a large body, not in the API at the other end.

static uint16_t g_pass1[W * H];
static uint16_t g_pass2[W * H];
static uint8_t  g_idx[W * H];        // palette index per pixel
static uint8_t  g_plte[256 * 3];     // the palette itself
static uint8_t  g_body[384 * 1024];
static size_t   g_body_len;

static char g_resp[16 * 1024];
static struct { size_t len; } g_sink;

static size_t write_cb(void *p, size_t sz, size_t nm, void *u) {
    (void)u;
    size_t n = sz * nm;
    if (g_sink.len + n < sizeof(g_resp) - 1) {
        memcpy(g_resp + g_sink.len, p, n);
        g_sink.len += n;
        g_resp[g_sink.len] = '\0';
    }
    return n;
}

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

// Encode g_idx + g_plte into buf, returning the byte count (0 if it would not fit).
static size_t png_encode(uint8_t *buf, size_t cap) {
    // One index byte per pixel, plus PNG's per-scanline filter byte.
    const size_t raw_len = (size_t)g_h * (1 + (size_t)g_w);
    // signature + IHDR + PLTE + IDAT + IEND + zlib overhead + a 5-byte
    // header per stored block
    const size_t need = 8 + 25 + (12 + 768) + 12 + 12 + raw_len + 6
                      + (raw_len / 65535 + 1) * 5;
    if (need > cap) return 0;

    size_t o = 0;
    static const uint8_t sig[8] = { 137, 'P', 'N', 'G', '\r', '\n', 26, '\n' };
    memcpy(buf, sig, 8); o = 8;

    uint8_t ihdr[13];
    put32(ihdr, (uint32_t)g_w); put32(ihdr + 4, (uint32_t)g_h);
    ihdr[8] = 8;    // bit depth
    ihdr[9] = 3;    // colour type: palette
    ihdr[10] = 0; ihdr[11] = 0; ihdr[12] = 0;
    chunk(buf, &o, "IHDR", ihdr, sizeof(ihdr));
    chunk(buf, &o, "PLTE", g_plte, sizeof(g_plte));

    // The IDAT payload is written in place, so its length is known only after
    // the fact: remember where it starts and patch the header afterwards.
    size_t len_at = o; o += 4;
    size_t type_at = o;
    memcpy(buf + o, "IDAT", 4); o += 4;
    size_t data_at = o;

    buf[o++] = 0x78; buf[o++] = 0x01;          // zlib: deflate, 32K window

    uint32_t a = 1, b = 0;                      // Adler-32, computed as we go
    size_t written = 0;
    int row = 0, col = 0;                       // position in the raw stream
    while (written < raw_len) {
        size_t block = raw_len - written;
        if (block > 65535) block = 65535;
        buf[o++] = (written + block >= raw_len) ? 1 : 0;   // BFINAL, stored
        buf[o++] = (uint8_t)(block & 0xFF);
        buf[o++] = (uint8_t)(block >> 8);
        buf[o++] = (uint8_t)(~block & 0xFF);
        buf[o++] = (uint8_t)((~block >> 8) & 0xFF);

        for (size_t i = 0; i < block; i++) {
            // Filter byte 0 at the start of every scanline, then the pixels.
            uint8_t v;
            if (col == 0) { v = 0; }
            else          { v = g_idx[(size_t)row * g_w + (col - 1)]; }
            buf[o++] = v;
            a = (a + v) % 65521; b = (b + a) % 65521;
            if (++col == 1 + g_w) { col = 0; row++; }
        }
        written += block;
    }
    put32(buf + o, (b << 16) | a); o += 4;      // Adler-32 of the raw stream

    put32(buf + len_at, (uint32_t)(o - data_at));
    uint32_t c = crc32_of(buf + type_at, (o - data_at) + 4, 0);
    put32(buf + o, c); o += 4;

    chunk(buf, &o, "IEND", NULL, 0);
    return o;
}

// ---------------------------------------------------------------- telegram

static const char *g_token, *g_chat;

// Wrap the PNG in a multipart body: sendPhoto wants a file part, and building
// the envelope by hand avoids depending on curl's MIME API being present in
// the port.
static size_t build_multipart(const char *boundary, const char *caption,
                              const uint8_t *png, size_t png_len) {
    size_t o = 0;
    o += (size_t)snprintf((char *)g_body + o, sizeof(g_body) - o,
        "--%s\r\nContent-Disposition: form-data; name=\"chat_id\"\r\n\r\n%s\r\n"
        "--%s\r\nContent-Disposition: form-data; name=\"caption\"\r\n\r\n%s\r\n"
        "--%s\r\nContent-Disposition: form-data; name=\"photo\"; filename=\"frame.png\"\r\n"
        "Content-Type: image/png\r\n\r\n",
        boundary, g_chat, boundary, caption, boundary);
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

static int send_photo(const uint8_t *png, size_t png_len, const char *caption) {
    static const char *boundary = "----bmdemo7f3a91c2";
    g_body_len = build_multipart(boundary, caption, png, png_len);
    if (!g_body_len) { fprintf(stderr, "[!] frame did not fit the body buffer\n"); return 0; }

    char url[256];
    snprintf(url, sizeof(url), "https://api.telegram.org/bot%s/sendPhoto", g_token);
    char ctype[128];
    snprintf(ctype, sizeof(ctype), "Content-Type: multipart/form-data; boundary=%s", boundary);

    CURL *h = curl_easy_init();
    if (!h) return 0;
    struct curl_slist *hdrs = curl_slist_append(NULL, ctype);
    hdrs = curl_slist_append(hdrs, "Expect:");   // no 100-continue round trip

    g_sink.len = 0; g_resp[0] = '\0';
    curl_easy_setopt(h, CURLOPT_URL, url);
    struct upload up = { g_body, g_body_len };
    curl_easy_setopt(h, CURLOPT_POST, 1L);
    curl_easy_setopt(h, CURLOPT_READFUNCTION, read_cb);
    curl_easy_setopt(h, CURLOPT_READDATA, &up);
    curl_easy_setopt(h, CURLOPT_POSTFIELDSIZE_LARGE, (curl_off_t)g_body_len);
    curl_easy_setopt(h, CURLOPT_HTTPHEADER, hdrs);
    curl_easy_setopt(h, CURLOPT_WRITEFUNCTION, write_cb);
    curl_easy_setopt(h, CURLOPT_TIMEOUT, HTTP_TIMEOUT);
    curl_easy_setopt(h, CURLOPT_NOSIGNAL, 1L);
    // Same rule as the agent: prefer a bundle on disk if the port put one
    // there, otherwise use the copy linked into the image. On a BareMetal
    // build there is no disk, so it is always the second.
    FILE *ca = fopen(CA_BUNDLE_PATH, "r");
    if (ca) {
        fclose(ca);
        curl_easy_setopt(h, CURLOPT_CAINFO, CA_BUNDLE_PATH);
    } else if (cacert_pem_len > 0) {
        struct curl_blob blob = { (void *)cacert_pem, cacert_pem_len, CURL_BLOB_NOCOPY };
        curl_easy_setopt(h, CURLOPT_CAINFO_BLOB, &blob);
    }

    CURLcode res = curl_easy_perform(h);
    long status = 0;
    curl_easy_getinfo(h, CURLINFO_RESPONSE_CODE, &status);
    curl_slist_free_all(hdrs);
    curl_easy_cleanup(h);

    if (res != CURLE_OK) { fprintf(stderr, "[!] send: %s\n", curl_easy_strerror(res)); return 0; }
    if (status != 200)   { fprintf(stderr, "[!] send: HTTP %ld %.200s\n", status, g_resp); return 0; }
    return 1;
}

// ---------------------------------------------------------------- main

static const char *env_or(const char *n, const char *fb) {
    const char *v = getenv(n);
    return (v && *v) ? v : fb;
}

int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IOLBF, 0);
    curl_global_init(CURL_GLOBAL_DEFAULT);
    build_palette();
    g_token = env_or("TELEGRAM_BOT_TOKEN", TELEGRAM_TOKEN_DEFAULT);
    g_chat  = env_or("DEMO_CHAT_ID",       DEMO_CHAT_DEFAULT);

    (void)argc; (void)argv;
    if (!strchr(g_token, ':')) {
        fprintf(stderr, "[!] no usable Telegram token.\n");
        return 1;
    }

    printf("[+] bmdemo up. %dx%d, max %d iterations, static %zu KB\n", W, H, MAX_ITER,
           (sizeof(g_pass1) + sizeof(g_pass2) + sizeof(g_idx) + sizeof(g_body)
            + 320 * 1024) / 1024);

    // Find the largest frame this machine can actually post, by posting.
    //
    // A ladder of body sizes against Telegram said everything from 16 KB up
    // failed -- but that probe sent deliberately invalid JSON, so the server
    // may have been rejecting it early and closing the connection mid-upload,
    // which looks identical from this end. The only way to separate "the port
    // cannot send this much" from "the server hung up on nonsense" is to send
    // something valid. A real frame is something valid.
    //
    // So: render small, post it, and step up while it keeps working. The
    // largest size that survives becomes the size it runs at. The machine
    // discovers its own limit instead of being told one.
    static const int LADDER[] = { 64, 96, 128, 160, 192, 256, 320, 384, 480 };
    const int N_LADDER = (int)(sizeof(LADDER) / sizeof(LADDER[0]));
    int chosen = 0;

    for (int i = 0; i < N_LADDER; i++) {
        g_w = g_h = LADDER[i];
        double s0 = 3.0 / (double)g_w;
        render(g_pass1, s0);
        colourise(g_pass1);
        static uint8_t probe_png[320 * 1024];
        size_t n = png_encode(probe_png, sizeof(probe_png));
        if (!n) { printf("[!] %dx%d did not fit the buffer\n", g_w, g_w); break; }

        char cap[256];
        snprintf(cap, sizeof(cap),
                 "sizing up: %dx%d, %zu bytes. Finding the largest frame this "
                 "machine can post.", g_w, g_h, n);
        int ok = send_photo(probe_png, n, cap);
        printf("[+] ladder %3dx%-3d png=%6zu bytes  %s\n", g_w, g_h, n,
               ok ? "posted" : "FAILED");
        if (!ok) break;
        chosen = LADDER[i];
        sleep(3);
    }

    if (!chosen) {
        fprintf(stderr, "[!] could not post even the smallest frame\n");
        return 1;
    }
    g_w = g_h = chosen;
    printf("[+] settled on %dx%d\n", chosen, chosen);

    double scale = 3.0 / (double)g_w;
    long frame = 0;

    for (;;) {
        frame++;
        time_t t0 = time(NULL);
        render(g_pass1, scale);
        render(g_pass2, scale);      // the same work, again, on purpose
        long secs = (long)(time(NULL) - t0);

        colourise(g_pass1);
        long bad = mark_disagreements();

        static uint8_t png[320 * 1024];
        size_t png_len = png_encode(png, sizeof(png));
        if (!png_len) { fprintf(stderr, "[!] png did not fit\n"); return 1; }

        double zoom = (3.0 / (double)g_w) / scale;
        char caption[512];
        snprintf(caption, sizeof(caption),
            "frame %ld  \xc2\xb7  zoom %.0fx  \xc2\xb7  %d iterations  \xc2\xb7  rendered twice in %lds\n"
            "%ld pixels disagreed between the two passes%s\n"
            "%dx%d PNG encoded by hand, in a 16 MiB machine with no operating system.",
            frame, zoom, MAX_ITER, secs, bad,
            bad ? " \xe2\x80\x94 each red dot is this machine giving two different answers to the same arithmetic."
                : " \xe2\x80\x94 clean frame.",
            g_w, g_h);

        printf("[+] frame %ld zoom=%.0fx render=%lds disagreements=%ld png=%zu bytes\n",
               frame, zoom, secs, bad, png_len);

        if (!send_photo(png, png_len, caption))
            fprintf(stderr, "[!] frame %ld not delivered\n", frame);

        if (FRAMES && frame >= FRAMES) break;
        scale /= ZOOM_PER_FRAME;
        if (scale < 1e-15) { scale = 3.0 / (double)g_w; printf("[+] precision floor, restarting the zoom\n"); }
        sleep(FRAME_GAP);
    }
    curl_global_cleanup();
    return 0;
}
