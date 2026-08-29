// bm4d.c -- a four-dimensional cross-section, animated, on a BareMetal unikernel.
//
// A tesseract (the 4D cube) is rotated rigidly through 4-space and cut by the
// fixed hyperplane w = 0. The cut is a three-dimensional polyhedron, and as
// the rotation turns the hypercube's square faces through the slice, that
// polyhedron morphs -- cube, to rhombus, to octahedron, and back. Each frame
// is one moment of that morph, projected to 2D and drawn as a wireframe.
//
// This is the same idea as slicing a 3D object to see its 2D cross-sections,
// one dimension up: slicing a 4D object gives a changing 3D one. There is no
// GPU and no graphics library; the geometry is arithmetic and the PNG is
// written by hand, exactly as in the fractal renderer this borrows its
// encoder and transport from.
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
#define FRAMES       0          // 0 = keep going until stopped
#define FRAME_GAP    90         // seconds between frames
#define ANGLE_STEP   0.13       // radians the tesseract turns per frame
#define VIEW_SCALE   0.28       // fraction of the frame the object fills

#define HTTP_TIMEOUT 120L
#define CA_BUNDLE_PATH "/etc/ssl/cacert.pem"

__attribute__((weak)) const unsigned char cacert_pem[1];
__attribute__((weak)) const unsigned int  cacert_pem_len;

// ---------------------------------------------------------------- memory
//
// All static, all claimed at link time, none of it in the uploaded image:
//
//   verification pass       256 KB   the frame drawn a second time, compared
//   (unused pass buffer)    512 KB   inherited from the fractal, left in place
//   palette indices         256 KB   one byte per pixel: background, depth, red
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

// ---------------------------------------------------------------- geometry

// A tesseract: 16 vertices at every (+/-1, +/-1, +/-1, +/-1). A vertex index's
// four low bits are the four signs.
#define NVERT 16
#define NFACE 24
static int  g_face[NFACE][4];      // each square 2-face, as four vertex indices
static uint8_t g_scratch[W * H];   // the verification pass

static void build_faces(void) {
    // A square face fixes two of the four axes at a sign and varies the other
    // two. Six axis pairs, four sign choices for the fixed pair -> 24 squares.
    static const int pair[6][2] = { {0,1},{0,2},{0,3},{1,2},{1,3},{2,3} };
    static const int ring[4][2] = { {0,0},{1,0},{1,1},{0,1} };  // corners in order
    int f = 0;
    for (int a = 0; a < 6; a++) {
        int i = pair[a][0], j = pair[a][1];
        int k = -1, l = -1;
        for (int t = 0; t < 4; t++) if (t != i && t != j) { if (k < 0) k = t; else l = t; }
        for (int sk = 0; sk < 2; sk++)
            for (int sl = 0; sl < 2; sl++) {
                for (int c = 0; c < 4; c++)
                    g_face[f][c] = (sk << k) | (sl << l) | (ring[c][0] << i) | (ring[c][1] << j);
                f++;
            }
    }
}

static void vertex(int idx, double out[4]) {
    out[0] = (idx & 1) ? 1.0 : -1.0;
    out[1] = (idx & 2) ? 1.0 : -1.0;
    out[2] = (idx & 4) ? 1.0 : -1.0;
    out[3] = (idx & 8) ? 1.0 : -1.0;
}

// Rotate a 4D point in the plane spanned by axes (a,b).
static void rot4(double v[4], int a, int b, double ang) {
    double c = cos(ang), s = sin(ang);
    double va = v[a], vb = v[b];
    v[a] = c * va - s * vb;
    v[b] = s * va + c * vb;
}

// A rigid pose of the tesseract at time t: two simultaneous 4D rotations, one
// mixing the w axis in (so the slice actually changes) and one that does not.
static void pose(int idx, double t, double out[4]) {
    vertex(idx, out);
    rot4(out, 0, 3, t);          // x-w plane: sweeps material through w = 0
    rot4(out, 1, 2, t * 0.6);    // y-z plane: rolls it as it goes
}

// Project a sliced 3D point to screen. A fixed isometric view, orthographic,
// so parallel edges stay parallel and the morph reads clearly.
static void project(const double p[3], int *sx, int *sy, double *depth) {
    static const double ca = 0.7071, sa = 0.7071;   // 45 deg about y
    double x =  ca * p[0] + sa * p[2];
    double z = -sa * p[0] + ca * p[2];
    double y =  p[1] * 0.9239 - z * 0.3827;          // then tilt about x
    double d =  p[1] * 0.3827 + z * 0.9239;
    double r = (double)((g_w < g_h) ? g_w : g_h) * VIEW_SCALE;
    *sx = (int)(g_w * 0.5 + x * r);
    *sy = (int)(g_h * 0.5 - y * r);
    *depth = d;
}

// A palette index for a segment at a given depth: 1..250 dark-to-bright, so
// nearer edges of the wireframe read brighter.
static uint8_t depth_shade(double d) {
    double t = (d + 1.6) / 3.2;                       // roughly [0,1]
    if (t < 0) t = 0; if (t > 1) t = 1;
    return (uint8_t)(20 + t * 229);
}

static void build_palette(void) {
    g_plte[0] = 8; g_plte[1] = 10; g_plte[2] = 18;    // 0: background, near-black blue
    for (int k = 1; k < 251; k++) {
        double t = (double)(k - 1) / 249.0;
        g_plte[k * 3 + 0] = (uint8_t)(40 + t * 215);  // warm white, cyan-ish in shadow
        g_plte[k * 3 + 1] = (uint8_t)(70 + t * 185);
        g_plte[k * 3 + 2] = (uint8_t)(120 + t * 135);
    }
    g_plte[255 * 3 + 0] = 255; g_plte[255 * 3 + 1] = 40; g_plte[255 * 3 + 2] = 40;  // disagreement
}

// Bresenham, into a chosen buffer, with a two-pixel nib so thin edges survive
// the JPEG-free downscale a phone does. Writes the max of existing and new
// shade, so crossings stay bright.
static void plot(uint8_t *buf, int x, int y, uint8_t v) {
    if (x < 0 || y < 0 || x >= g_w || y >= g_h) return;
    if (v > buf[y * g_w + x]) buf[y * g_w + x] = v;
}
static void line(uint8_t *buf, int x0, int y0, int x1, int y1, uint8_t v) {
    int dx = abs(x1 - x0), dy = -abs(y1 - y0);
    int sx = x0 < x1 ? 1 : -1, sy = y0 < y1 ? 1 : -1;
    int err = dx + dy;
    for (;;) {
        plot(buf, x0, y0, v); plot(buf, x0 + 1, y0, v); plot(buf, x0, y0 + 1, v);
        if (x0 == x1 && y0 == y1) break;
        int e2 = 2 * err;
        if (e2 >= dy) { err += dy; x0 += sx; }
        if (e2 <= dx) { err += dx; y0 += sy; }
    }
}

// Draw the w = 0 cross-section of the posed tesseract into buf. For each square
// face, the boundary crosses w = 0 in exactly two points when the face
// straddles the slice; those two points are one edge of the cut polyhedron.
static void slice_into(uint8_t *buf, double t) {
    memset(buf, 0, (size_t)g_w * g_h);
    double vp[NVERT][4];
    for (int i = 0; i < NVERT; i++) pose(i, t, vp[i]);

    for (int f = 0; f < NFACE; f++) {
        double pts[4][3]; double dep[4]; int n = 0;
        for (int e = 0; e < 4 && n < 4; e++) {
            double *a = vp[g_face[f][e]];
            double *b = vp[g_face[f][(e + 1) & 3]];
            double wa = a[3], wb = b[3];
            if ((wa <= 0 && wb > 0) || (wa > 0 && wb <= 0)) {
                double u = wa / (wa - wb);             // where the edge hits w = 0
                for (int c = 0; c < 3; c++) pts[n][c] = a[c] + u * (b[c] - a[c]);
                dep[n] = 0; n++;
            }
        }
        if (n == 2) {
            int x0, y0, x1, y1; double d0, d1;
            project(pts[0], &x0, &y0, &d0);
            project(pts[1], &x1, &y1, &d1);
            line(buf, x0, y0, x1, y1, depth_shade((d0 + d1) * 0.5));
        }
    }
}

// Draw the frame twice and mark, in red, any pixel the two passes disagree on.
// A cross-section is a pure function of the pose, so a correct machine draws it
// identically both times.
static long slice_render(double t) {
    slice_into(g_idx, t);
    slice_into(g_scratch, t);
    long bad = 0;
    for (int i = 0; i < g_w * g_h; i++)
        if (g_idx[i] != g_scratch[i]) { g_idx[i] = 255; bad++; }
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
static const int LADDER[] = { 64, 96, 128, 160, 192, 256, 320, 384, 480 };
#define N_LADDER ((int)(sizeof(LADDER) / sizeof(LADDER[0])))
static size_t g_budget;     // largest frame this machine has actually delivered

#define DEF_WINDOW  32768
#define DEF_MIN_MATCH 3
#define DEF_MAX_MATCH 258
#define DEF_HASH_BITS 15
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
static uint8_t g_raw[(W + 1) * H];      // filter byte + indices, per scanline

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
static int send_photo_once(const uint8_t *png, size_t png_len, const char *caption);

static long g_sends, g_retries, g_failures;

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

    build_faces();
    build_palette();
    printf("[+] bm4d up. tesseract cross-section, up to %dx%d, static %zu KB\n", W, H,
           (sizeof(g_scratch) + sizeof(g_pass1) + sizeof(g_idx) + sizeof(g_body)
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
    int chosen = 0;

    for (int i = 0; i < N_LADDER; i++) {
        g_w = g_h = LADDER[i];
        slice_render(0.0);
        static uint8_t probe_png[320 * 1024];
        size_t n = png_encode(probe_png, sizeof(probe_png));
        if (!n) { printf("[!] %dx%d did not fit the buffer\n", g_w, g_w); break; }

        char cap[256];
        snprintf(cap, sizeof(cap),
                 "sizing up: %dx%d, %zu bytes. Finding the largest frame this "
                 "machine can post.", g_w, g_h, n);
        int ok = send_photo(probe_png, n, cap);
        printf("[+] ladder %3dx%-3d png=%6zu bytes  %s\n", g_w, g_h, n,
               ok ? "posted" : "FAILED after 4 attempts");
        if (!ok) break;
        chosen = LADDER[i];
        g_budget = n;           // this many bytes is known to work
        sleep(3);
    }

    if (!chosen) {
        fprintf(stderr, "[!] could not post even the smallest frame\n");
        return 1;
    }
    g_w = g_h = chosen;
    printf("[+] settled on %dx%d\n", chosen, chosen);

    // Track the field of view -- how much of the plane is on screen -- rather
    // than the distance between pixels. They differ the moment the frame size
    // changes: holding units-per-pixel fixed while dropping 320 to 256 keeps
    // the detail and quietly crops a fifth of the picture away, so the
    // sequence jumps instead of zooming. Holding the field of view fixed
    // shows the same region at fewer pixels, which is what dropping
    // resolution should mean.
    double angle = 0.0;
    long frame = 0;

    for (;;) {
        frame++;
        long bad = 0, secs = 0;
        static uint8_t png[320 * 1024];
        size_t png_len = 0;

        // Re-check the size every frame. A wireframe compresses far better than
        // the fractal did -- flat background, thin lines -- but the amount of
        // line moving through the slice still changes frame to frame, so the
        // size that fit last time is not guaranteed to fit this one.
        for (;;) {
            time_t t0 = time(NULL);
            bad = slice_render(angle);
            secs = (long)(time(NULL) - t0);
            png_len = png_encode(png, sizeof(png));
            if (!png_len) { fprintf(stderr, "[!] png did not fit\n"); return 1; }
            if (!g_budget || png_len <= g_budget) break;

            int idx = 0;
            while (idx < N_LADDER - 1 && LADDER[idx] < g_w) idx++;
            if (idx == 0) break;
            g_w = g_h = LADDER[idx - 1];
            printf("[+] frame %ld was %zu bytes, over the %zu that works -- dropping to %dx%d\n",
                   frame, png_len, g_budget, g_w, g_h);
        }

        char caption[512];
        snprintf(caption, sizeof(caption),
            "frame %ld  \xc2\xb7  a tesseract turned %.0f\xc2\xb0 through 4-space\n"
            "sliced at w = 0 \xe2\x86\x92 a 3D cross-section, projected to your screen. drawn twice in %lds.\n"
            "%ld pixels disagreed between the passes%s\n"
            "%dx%d wireframe, PNG encoded by hand, in a 16 MiB machine with no operating system.",
            frame, angle * 57.2958, secs, bad,
            bad ? " \xe2\x80\x94 each red dot is this machine drawing the same edge two ways."
                : " \xe2\x80\x94 clean.",
            g_w, g_h);

        printf("[+] frame %ld angle=%.2f render=%lds disagreements=%ld png=%zu bytes\n",
               frame, angle, secs, bad, png_len);

        if (!send_photo(png, png_len, caption))
            fprintf(stderr, "[!] frame %ld not delivered\n", frame);

        if (FRAMES && frame >= FRAMES) break;
        angle += ANGLE_STEP;
        sleep(FRAME_GAP);
    }
    curl_global_cleanup();
    return 0;
}
