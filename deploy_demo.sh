#!/bin/bash
# Build bmdemo.c into a BareMetal unikernel and deploy it alongside the agent.
#
# Deliberately separate from deploy_baremetal.sh rather than a flag on it: this
# touches only instances and images named "bmdemo", so a mistake here cannot
# retire the agent. The two share a bot token safely because the demo only ever
# calls sendPhoto and never polls getUpdates.
#
#   BM_API_KEY=bmvps_...  bash deploy_demo.sh
#
# Reads:  ~/.tg_token   Telegram bot token
#         ~/.demo_chat  chat id to post frames to
set -eu

: "${BM_API_KEY:?set BM_API_KEY}"
APP_DIR="${APP_DIR:-$HOME/BareMetal-App}"
SRC="${SRC:-$HOME/bmdemo.c}"
NAME="${NAME:-bmdemo}"
RAM_MIB="${RAM_MIB:-16}"

cd "$APP_DIR"
cp "$SRC" bmdemo.c
sed -i "s|PUT_BOT_TOKEN_HERE|$(cat "$HOME/.tg_token")|" bmdemo.c
sed -i "s|PUT_CHAT_ID_HERE|$(cat "$HOME/.demo_chat")|" bmdemo.c

echo "=== building ==="
./1-build.sh bmdemo.c > /tmp/deploy_demo_build.log 2>&1 || { tail -20 /tmp/deploy_demo_build.log; exit 1; }
SIZE=$(stat -c %s baremetal.elf)
echo "image: $SIZE bytes"

echo "=== uploading ==="
IMAGE_ID=$(./bm-api.sh images upload "$NAME" baremetal.elf | awk -F': ' '/^id:/{print $2}')
echo "image id: $IMAGE_ID"

echo "=== creating instance (1 vCPU, ${RAM_MIB} MiB) ==="
INSTANCE_ID=$(./bm-api.sh instances create "$NAME" 1 "$RAM_MIB" "$IMAGE_ID" | awk -F': ' '/^id:/{print $2}')
echo "instance id: $INSTANCE_ID"
echo "$INSTANCE_ID" > "$HOME/.bmdemo_instance"

# Same order as the agent's deploy, for the same reason: the new one exists
# before the old one goes, so a failure leaves something running.
echo "=== retiring previous $NAME instances ==="
./bm-api.sh instances list 2>/dev/null | awk -v n="$NAME" -v keep="$INSTANCE_ID" \
    '$2 == n && $1 != keep {print $1}' | while read -r old; do
    ./bm-api.sh instances stop "$old" >/dev/null 2>&1 || true
    for _ in 1 2 3 4 5 6 7 8 9 10; do
        state=$(./bm-api.sh instances show "$old" 2>/dev/null | awk -F': ' '/^status:/{print $2}')
        [ "$state" = "STOPPED" ] && break
        sleep 2
    done
    ./bm-api.sh instances rm "$old" >/dev/null 2>&1 && echo "  retired $old" || \
        echo "  WARNING: could not retire $old (state=$state)" >&2
done || true

echo "=== pruning unused images named $NAME ==="
./bm-api.sh images list 2>/dev/null | awk -v n="$NAME" -v keep="$IMAGE_ID" \
    '$2 == n && $1 != keep {print $1}' | while read -r old; do
    ./bm-api.sh images rm "$old" >/dev/null 2>&1 && echo "  removed $old"
done || true

echo
echo "Live. Frames will arrive in the configured chat."
echo "Serial console:  ./bm-api.sh instances logs $INSTANCE_ID"
