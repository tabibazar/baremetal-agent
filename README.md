# baremetal-agent

An LLM agent you can text, running as a [BareMetal](https://github.com/ReturnInfinity) unikernel.

There is no operating system under this program. No kernel, no init, no shell, no filesystem it depends on — the image *is* the machine. Inside it: an agent loop with tool calling, a TLS 1.3 client, a TCP/IP stack, and enough room left to hold a conversation. All of it in **16 MiB of RAM**, from a **2.88 MB** image.

A live one is running at **[@unikernel_bot](https://t.me/unikernel_bot)** — text it and you are talking to that.

```
you:  how much memory are you using right now, and what is under you?

bot:  I am currently using 10,400 bytes of memory within my arena.
      I am a BareMetal unikernel, meaning I run directly on the
      hardware with no operating system or kernel beneath me.
```

Every number it gives you about itself is measured at the moment you ask — it has no way to state one except by calling a tool that reads it. It can also search the web, and it keeps the difference explicit:

```
you:  what is the latest release version of Firecracker,
      and what are you running on?

bot:  The latest release version of Firecracker is v1.16.1, according
      to its GitHub releases page at https://github.com/firecracker-
      microvm/firecracker/releases.

      I am running on BareMetal-AppPort. I am linked directly against
      the hardware and do not have an operating system, kernel, or
      other typical software layers beneath me.
```

## Why this is interesting

Agent frameworks generally assume a language runtime, a garbage collector, a container image, and hundreds of megabytes of headroom. This one is a single C file that boots as the machine.

| | |
|---|---|
| Image size | 2,883,040 bytes |
| RAM | 16 MiB (the platform's per-instance ceiling) |
| Static footprint | 768 KB — arena 384 KB, HTTP 256 KB, payload 128 KB |
| Arena used for a typical answer | 8–54 KB (the upper end with web results in context) |
| Heap allocations | none — the allocator is compiled out |
| Inside the image | musl, lwIP, mbedTLS, libcurl, cJSON, Mozilla CA bundle |
| Not inside the image | operating system, kernel, init, shell, package manager, language runtime |

## How it works

```
poll Telegram for a message
  → think (Gemini, via the OpenAI-compatible chat/completions API)
  → call tools, observe results, think again          (up to MAX_STEPS)
  → reply into the chat it came from
repeat
```

The conversation array is the agent's whole memory for one exchange, and the arena is reset after each answer — so memory use is bounded by a single question, not by uptime.

### The tools

The first four report on the machine the agent is living inside, which is the point: the subject is the platform, not the model. The last two let it answer questions about the world.

| Tool | Answers |
|---|---|
| `machine_facts` | RAM ceiling, image size, seconds since boot |
| `memory_usage` | live arena use and high-water mark, read from its own allocator |
| `build_info` | what is linked into the image, and what is absent |
| `ping_api` | a real HTTPS round-trip, timed now, TLS handshake included |
| `startup_timing` | how long it took from program start to its first completed HTTPS request |
| `web_search` | the outside world, via [Firecrawl](https://firecrawl.dev) |
| `read_page` | a page's main content as markdown, when a snippet is too thin |
| `remember` | keeps one fact about the person it is talking to |
| `recall` | everything it has kept about them |

The two kinds of knowledge are kept visibly apart: facts about itself are measured
at the moment you ask, while anything from the web is somebody else's claim and is
cited with its URL. Reading is confined to hosts a search result came from, so it
cannot wander onto a domain it invented, and pages are clipped to 5 KB — the whole
conversation is re-sent on every turn, and an untrimmed page would crowd out
everything else in a 16 MiB machine.

Web search is optional. Build without a Firecrawl key and the machine simply cannot
see out, and says so rather than pretending.

### Memory, on a machine with no disk

BareMetal Cloud instances have no writable filesystem. `open(O_CREAT)` fails with `ENOENT` — `disktest.c` in this repo measures it, and the same binary on a local Firecracker run with a real 512 MB `disk.img` writes happily and the value survives a restart, so this is the platform rather than the program.

So its memory lives a network round-trip away, behind the same HTTPS stack everything else uses: Redis with an [Upstash](https://upstash.com)-shaped REST interface. `kv_service.py` here is a self-hosted version of that interface if you would rather not sign up for anything; hosted Upstash works unchanged by swapping the URL and token.

```
you (before):  My name is Reza and I built you.
bot:           It is good to meet you, Reza. I will remember that you built me.

...restarted, redeployed, a different machine entirely...

you (after):   who am I?
bot:           You are Reza, my builder.
```

At boot it proves the path before anyone talks to it, which also happens to count restarts:

```
[+] web search: on   memory: on
[+] memory reachable: this is boot #3, remembered across restarts
```

**The key is built in C, not chosen by the model.** Notes live under `agent:notes:<chat_id>`, taken from the message that arrived — so the model cannot name a key and therefore cannot read or write another person's notes, however it is asked. The list is trimmed to the last 20 entries per person. What it already knows is loaded before the model sees the message, so it recognises you without being told to check.

Without credentials, memory is simply off and the agent says it has none rather than pretending.

#### Self-hosting the store

`kv_service.py` is a small HTTP front for a normal Redis, in the Upstash request shape:

```sh
sudo apt install -y redis-server python3-redis
redis-cli config set appendonly yes && redis-cli config rewrite   # survive restarts
openssl rand -hex 24 > ~/.kv_token
KV_TOKEN=$(cat ~/.kv_token) python3 kv_service.py                 # listens on :8080
```

Put it behind TLS (any reverse proxy) and point `KV_URL` at it. Commands are allowlisted — `FLUSHALL`, `CONFIG`, `KEYS` and `SHUTDOWN` are refused — because the endpoint faces the internet and a leaked token should be able to touch keys, not reconfigure or wipe the server.

### Cold start, measured

On boot it prints, and will tell you if asked:

```
[+] startup: program start -> first HTTPS (TLS included) in 888.4 ms
```

That covers process start, network bring-up, DNS, TCP connect, the TLS handshake and the first HTTP response. It does **not** include the virtual machine booting before the program began — that is not visible from inside, and the tool says so rather than quietly claiming it.

`time()` has one-second resolution, which is useless here, so the interval is measured with the CPU cycle counter. The counter has resolution but no unit, so its frequency is calibrated against a one-second sleep *after* the interval has already been recorded — measure first, learn the scale later, convert at the end. Calibrating up front would put a second of sleep inside the thing being measured. The result is rounded to 0.1 ms because the calibration, not the counter, is the limiting factor.

The figure depends on which endpoint is contacted first. With memory enabled the first request is the boot-time memory check, and the number rose to about 1,700 ms; the 888 ms above was measured against `api.telegram.org`. It is the same measurement of a different first request, not a regression.

For reference, the same source as an ordinary Linux process on a different host measured 876.5 ms. Different machines, so not a controlled comparison — but the two being within about 1% suggests this interval is dominated by DNS and TLS round-trips rather than by whatever is underneath.

On a non-x86 build the cycle counter is unavailable and it falls back to the one-second clock, reporting that it did.

### Zero heap, by construction

cJSON is pointed at a static bump arena via `cJSON_InitHooks`, buffers are fixed arrays, and past a certain line in the file the allocator simply does not exist:

```c
#define malloc   DO_NOT_malloc
#define calloc   DO_NOT_calloc
#define realloc  DO_NOT_realloc
#define free     DO_NOT_free
#define strdup   DO_NOT_strdup
```

Any code added below that fails to compile if it reaches for the heap. libcurl still allocates internally — this claim is about the program, not the process.

## Build and run

The same source builds for Linux, macOS and BareMetal. Start with an ordinary build: it is faster to iterate on, and it proves your tokens work before the unikernel is involved.

### 1 · Get the credentials

**Telegram bot token.** Message [@BotFather](https://t.me/botfather), send `/newbot`, follow the prompts. You get a token shaped like `8123456789:AAH…`.

**Model API key.** A Gemini key from [aistudio.google.com/apikey](https://aistudio.google.com/apikey) (free tier is fine). Any OpenAI-compatible provider works — see [Configuration](#configuration).

**Search key (optional).** A [Firecrawl](https://firecrawl.dev) key enables `web_search` and `read_page`. Without one the agent can still talk about itself.

Keep them in files rather than in your shell history:

```sh
printf '%s' '8123456789:AAH...' > ~/.tg_token   && chmod 600 ~/.tg_token
printf '%s' 'AIza...'           > ~/.gemini_key    && chmod 600 ~/.gemini_key
printf '%s' 'fc-...'            > ~/.firecrawl_key && chmod 600 ~/.firecrawl_key
```

### 2 · Build and run on Linux or macOS

Dependencies are libcurl and libcjson:

```sh
# Debian/Ubuntu
sudo apt install -y build-essential libcurl4-openssl-dev libcjson-dev
# macOS
brew install curl cjson
```

Then:

```sh
make
export TELEGRAM_BOT_TOKEN=$(cat ~/.tg_token)
export GEMINI_API_KEY=$(cat ~/.gemini_key)

./build/bmagent --ask "what are you running on?"   # one question, no Telegram
./build/bmagent --once                             # answer whatever is waiting, then exit
./build/bmagent                                    # poll forever
```

`--ask` is the fastest way to check that your key works and the tools return sane numbers.

> **Run only one instance at a time.** Telegram hands each message to whoever polls first, so a local copy and a deployed copy will steal messages from each other.

### 3 · Build as a BareMetal unikernel

**Prerequisites.** A Linux x86-64 host with `git curl unzip tar gcc nasm make patch jq e2fsprogs`. On Ubuntu:

```sh
sudo apt install -y build-essential nasm jq e2fsprogs unzip patch screen git curl
```

**Set up the toolchain.** This builds musl, lwIP, mbedTLS, curl and more from source, and takes a few minutes. You only do it once:

```sh
git clone https://github.com/ReturnInfinity/BareMetal-App
cd BareMetal-App
./setup.sh
```

**Add cJSON**, which BareMetal-AppPort does not ship:

```sh
git clone --depth 1 https://github.com/DaveGamble/cJSON.git cjson_src
mkdir -p cjson && cp cjson_src/cJSON.c cjson_src/cJSON.h cjson/ && rm -rf cjson_src
```

**Bake in the configuration.** BareMetal exposes no environment, so `getenv` returns nothing there and the compile-time defaults are used instead. Copy the source in and substitute:

```sh
cp /path/to/baremetal-agent/bmagent.c .
sed -i "s|PUT_BOT_TOKEN_HERE|$(cat ~/.tg_token)|"   bmagent.c
sed -i "s|PUT_GEMINI_KEY_HERE|$(cat ~/.gemini_key)|" bmagent.c
sed -i "s|PUT_FIRECRAWL_KEY_HERE|$(cat ~/.firecrawl_key)|" bmagent.c   # optional
```

**Build the image:**

```sh
./1-build.sh bmagent.c cjson/cJSON.c
```

That produces `baremetal.elf` — a complete bootable machine, around 2.88 MB.

**Run it locally** under Firecracker (needs `firecracker` on PATH, and a tap device for networking — `BareMetal-Firecracker/scripts/mkbr0.sh` sets one up, but read it first: on a wired single-NIC host it moves your host IP onto a bridge and will disconnect a remote session):

```sh
./2-run.sh
```

**Or deploy to BareMetal Cloud**, which needs no networking setup at all:

```sh
BM_API_KEY=bmvps_... bash deploy_baremetal.sh
```

Get the key from [baremetal.returninfinity.com/dashboard](https://baremetal.returninfinity.com/dashboard). The script fetches cJSON if needed, bakes your tokens, builds twice — the second pass compiles in the image size measured from the first, so the agent can report its own size honestly — uploads, and starts a 1 vCPU / 16 MiB instance.

Watch the serial console:

```sh
./bm-api.sh instances logs <instance-id>
```

## Configuration

Environment variables where there are any; the compile-time `#define`s otherwise.

| Setting | Default | Notes |
|---|---|---|
| `TELEGRAM_BOT_TOKEN` | — | required |
| `GEMINI_API_KEY` | — | required |
| `FIRECRAWL_API_KEY` | — | optional; enables web search |
| `KV_URL` / `KV_TOKEN` | — | optional; enables memory that survives restarts |
| `LLM_BASE_URL` | Gemini's OpenAI-compatible endpoint | any compatible provider works |
| `LLM_MODEL` | `gemini-2.5-flash` | pinned deliberately, see below |
| `MAX_STEPS` | 6 | tool-calling rounds per message |
| `REPLIES_PER_HOUR` | 40 | spend ceiling, enforced in C |
| `ARENA_BYTES` | 384 KB | shrink for a tighter machine; it prints its high-water mark |
| `RAM_MIB` / `IMAGE_BYTES` | 16 / 0 | facts it cannot discover itself; the deploy script fills them in |

Compile-time values are `-D` overridable: `gcc -DMAX_STEPS=10 ...`

**On the model name:** `gemini-2.5-flash` is pinned rather than using a `-latest` alias. Under load the aliases queue indefinitely instead of returning an error, which is indistinguishable from a hang; the pinned name either answers or returns a clean 503, and the agent retries with backoff.

## Operating notes

**Secrets are compiled into the image.** There is no environment on BareMetal to read them from at runtime. Anyone who can read the image can recover the tokens, so treat a built `baremetal.elf` as a secret. Rotating a token means rebuilding and redeploying — about a minute.

**There is a spend ceiling.** `REPLIES_PER_HOUR` (default 40) is counted in C, not requested in the prompt, because a prompt is a request and this is a limit. Past it, the agent says so and stops answering until the hour rolls over.

**It answers the backlog on first boot.** Telegram redelivers any messages that were never acknowledged, so a fresh instance will work through whatever is queued. Send it something and it catches up.

**Clock resolution is one second.** Uptime and round-trip figures are whole seconds; the agent says so rather than implying more precision than it has.

## Credits

Built on [BareMetal](https://github.com/ReturnInfinity/BareMetal), [BareMetal-App and BareMetal-AppPort](https://github.com/ReturnInfinity) by Return Infinity — musl, lwIP, mbedTLS and curl ported into an image that needs no operating system. The idea of a C program on BareMetal calling an HTTP API and posting to a chat channel comes from [Flâneur the Wanderer](https://github.com/varunmadhok/Flaneur-the-wanderer) by varunmadhok.

New to unikernels? [**The Program Is the Machine**](https://github.com/tabibazar/unikernel-c/blob/main/docs/unikernels-explained.html) explains what exokernels and unikernels are, what a unikernel removes that a container does not, and what all of this measured — this agent is its worked example.

Companion project: [unikernel-c](https://github.com/tabibazar/unikernel-c) — the agent loop this grew out of, a prime-search swarm, and a technical report measuring the platform.

## License

MIT — see [LICENSE](LICENSE).
