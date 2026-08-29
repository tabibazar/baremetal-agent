#!/bin/bash
# One-shot deploy of the body-size probe. Touches only bmprobe-named resources.
set -eu
: "${BM_API_KEY:?set BM_API_KEY}"
cd "$HOME/BareMetal-App"
sed "s|PUT_PROBE_URL_HERE|$(cat "$HOME/.kv_url")|" "$HOME/bmprobe.c" > bmprobe.c
echo "=== building ==="
./1-build.sh bmprobe.c > /tmp/probe_build.log 2>&1 || { tail -20 /tmp/probe_build.log; exit 1; }
echo "image: $(stat -c %s baremetal.elf) bytes"
IMAGE_ID=$(./bm-api.sh images upload bmprobe baremetal.elf | awk -F': ' '/^id:/{print $2}')
INSTANCE_ID=$(./bm-api.sh instances create bmprobe 1 16 "$IMAGE_ID" | awk -F': ' '/^id:/{print $2}')
echo "instance: $INSTANCE_ID"
echo "$INSTANCE_ID" > "$HOME/.bmprobe_instance"
echo "image: $IMAGE_ID" > "$HOME/.bmprobe_image"
echo "Live."
