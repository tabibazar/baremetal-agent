#!/bin/bash
# Build bm4d.c and deploy it as its own instance. Touches only bm4d-named
# resources, so it cannot disturb the agent. The animation only ever calls
# sendPhoto, so it shares the bot token with the agent without contention.
#
#   BM_API_KEY=bmvps_...  bash deploy_4d.sh
#
# Reads:  ~/.tg_token   Telegram bot token
#         ~/.demo_chat  chat id to post frames to
set -eu
: "${BM_API_KEY:?set BM_API_KEY}"
cd "$HOME/BareMetal-App"
cp "$HOME/bm4d.c" bm4d.c
sed -i "s|PUT_BOT_TOKEN_HERE|$(cat "$HOME/.tg_token")|" bm4d.c
sed -i "s|PUT_CHAT_ID_HERE|$(cat "$HOME/.demo_chat")|" bm4d.c

echo "=== building ==="
./1-build.sh bm4d.c > /tmp/deploy_4d_build.log 2>&1 || { tail -20 /tmp/deploy_4d_build.log; exit 1; }
echo "image: $(stat -c %s baremetal.elf) bytes"

IMAGE_ID=$(./bm-api.sh images upload bm4d baremetal.elf | awk -F': ' '/^id:/{print $2}')
INSTANCE_ID=$(./bm-api.sh instances create bm4d 1 16 "$IMAGE_ID" | awk -F': ' '/^id:/{print $2}')
echo "$INSTANCE_ID" > "$HOME/.bm4d_instance"
echo "instance: $INSTANCE_ID"

# Retire older bm4d instances only after the new one exists.
./bm-api.sh instances list 2>/dev/null | awk -v keep="$INSTANCE_ID" '$2=="bm4d" && $1!=keep {print $1}' | while read -r old; do
    ./bm-api.sh instances stop "$old" >/dev/null 2>&1 || true
    for _ in $(seq 1 10); do
        st=$(./bm-api.sh instances show "$old" 2>/dev/null | awk -F': ' '/^status:/{print $2}')
        [ "$st" = "STOPPED" ] && break; sleep 2
    done
    ./bm-api.sh instances rm "$old" >/dev/null 2>&1 && echo "  retired $old"
done || true

./bm-api.sh images list 2>/dev/null | awk -v keep="$IMAGE_ID" '$2=="bm4d" && $1!=keep {print $1}' | while read -r old; do
    ./bm-api.sh images rm "$old" >/dev/null 2>&1 && echo "  removed image $old"
done || true

echo "Live. Frames arrive in the configured chat."
