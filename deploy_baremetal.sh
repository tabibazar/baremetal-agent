#!/bin/bash
# Build bmagent.c into a BareMetal unikernel and deploy it to BareMetal Cloud.
#
# Run this from a machine that has a working BareMetal-App checkout (see
# README.md). Secrets are read from files and baked into the image, because
# BareMetal exposes no environment for a program to read at runtime.
#
#   BM_API_KEY=bmvps_...  bash deploy_baremetal.sh
#
# Reads:  ~/.tg_token       Telegram bot token
#         ~/.gemini_key     model API key
#         ~/.firecrawl_key  search/scrape key (optional; omit for a machine that
#                           can only talk about itself)
set -eu

: "${BM_API_KEY:?set BM_API_KEY — create one at https://baremetal.returninfinity.com/dashboard}"

APP_DIR="${APP_DIR:-$HOME/BareMetal-App}"
SRC="${SRC:-$HOME/bmagent.c}"
NAME="${NAME:-bmagent}"
RAM_MIB="${RAM_MIB:-16}"

cd "$APP_DIR"

# cJSON is not part of BareMetal-AppPort, so it is compiled alongside the app.
if [ ! -f cjson/cJSON.c ]; then
    echo "=== fetching cJSON ==="
    git clone --quiet --depth 1 https://github.com/DaveGamble/cJSON.git cjson_src
    mkdir -p cjson
    cp cjson_src/cJSON.c cjson_src/cJSON.h cjson/
    rm -rf cjson_src
fi

bake() {   # bake <image_bytes>
    cp "$SRC" bmagent.c
    sed -i "s|PUT_BOT_TOKEN_HERE|$(cat "$HOME/.tg_token")|" bmagent.c
    sed -i "s|PUT_GEMINI_KEY_HERE|$(cat "$HOME/.gemini_key")|" bmagent.c
    if [ -f "$HOME/.firecrawl_key" ]; then
        sed -i "s|PUT_FIRECRAWL_KEY_HERE|$(cat "$HOME/.firecrawl_key")|" bmagent.c
    fi
    # Memory: any Redis with an Upstash-shaped REST interface.
    if [ -f "$HOME/.kv_url" ] && [ -f "$HOME/.kv_token" ]; then
        sed -i "s|PUT_KV_URL_HERE|$(cat "$HOME/.kv_url")|" bmagent.c
        sed -i "s|PUT_KV_TOKEN_HERE|$(cat "$HOME/.kv_token")|" bmagent.c
    fi
    sed -i "s|^#define RAM_MIB         16$|#define RAM_MIB         ${RAM_MIB}|" bmagent.c
    sed -i "s|^#define IMAGE_BYTES     0 |#define IMAGE_BYTES     ${1} |" bmagent.c
}

# Two passes on purpose. The agent reports its own image size, and that number
# is only correct if it was compiled in AFTER the image existed to be measured.
echo "=== pass 1: build to learn the image size ==="
bake 0
./1-build.sh bmagent.c cjson/cJSON.c > /dev/null
SIZE=$(stat -c %s baremetal.elf)
echo "image is $SIZE bytes"

echo "=== pass 2: rebuild with that size compiled in ==="
bake "$SIZE"
./1-build.sh bmagent.c cjson/cJSON.c > /dev/null
FINAL=$(stat -c %s baremetal.elf)
echo "final image: $FINAL bytes"
[ "$FINAL" = "$SIZE" ] || echo "note: size shifted by $((FINAL - SIZE)) bytes between passes"

echo "=== uploading ==="
IMAGE_ID=$(./bm-api.sh images upload "$NAME" baremetal.elf | awk -F': ' '/^id:/{print $2}')
echo "image id: $IMAGE_ID"

echo "=== creating instance (1 vCPU, ${RAM_MIB} MiB) ==="
INSTANCE_ID=$(./bm-api.sh instances create "$NAME" 1 "$RAM_MIB" "$IMAGE_ID" | awk -F': ' '/^id:/{print $2}')
echo "instance id: $INSTANCE_ID"
echo "$INSTANCE_ID" > "$HOME/.bmagent_instance"

echo
echo "It is live. Text the bot; the reply comes from the unikernel."
echo "Serial console:  ./bm-api.sh instances logs $INSTANCE_ID"
