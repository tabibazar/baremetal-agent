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

# Retire older instances of the same name only AFTER the new one exists, so a
# failed deploy leaves the previous one serving rather than nothing at all.
echo "=== retiring previous instances ==="
# Deleting an instance is refused unless it has actually reached STOPPED, and
# stopping is not instant -- so wait for the state rather than sleeping and
# hoping. Two instances left running would both poll Telegram and steal each
# other's messages, so this failing quietly is worse than it failing loudly.
./bm-api.sh instances list 2>/dev/null | awk -v n="$NAME" -v keep="$INSTANCE_ID" \
    '$2 == n && $1 != keep {print $1}' | while read -r old; do
    ./bm-api.sh instances stop "$old" >/dev/null 2>&1 || true
    for _ in 1 2 3 4 5 6 7 8 9 10; do
        state=$(./bm-api.sh instances show "$old" 2>/dev/null | awk -F': ' '/^status:/{print $2}')
        [ "$state" = "STOPPED" ] && break
        sleep 2
    done
    if ./bm-api.sh instances rm "$old" >/dev/null 2>&1; then
        echo "  retired $old"
    else
        echo "  WARNING: could not retire $old (state=$state) — it is still polling" >&2
    fi
done || true

# Only now are the old images unreferenced. The account allows ten, and every
# redeploy adds one, so without this a deploy eventually fails at the upload
# step. Failures here are ignored on purpose: an image still in use should stay,
# and a tidy-up problem must never fail a deploy that already succeeded.
echo "=== pruning unused images named $NAME ==="
./bm-api.sh images list 2>/dev/null | awk -v n="$NAME" -v keep="$IMAGE_ID" \
    '$2 == n && $1 != keep {print $1}' | while read -r old; do
    if ./bm-api.sh images rm "$old" >/dev/null 2>&1; then echo "  removed $old"; fi
done || true

echo
echo "It is live. Text the bot; the reply comes from the unikernel."
echo "Serial console:  ./bm-api.sh instances logs $INSTANCE_ID"
