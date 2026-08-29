#!/bin/sh
# Render the carousel to a PDF LinkedIn will accept as a document post.
#
# Chrome rather than a PDF library because the slides are ordinary CSS and
# @page sets a square 1080x1080 canvas; anything that reflows to A4 ruins the
# layout. --virtual-time-budget waits for the webfont, without which the
# monospace columns fall back and stop lining up.
set -eu
CHROME=${CHROME:-"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"}
cd "$(dirname "$0")"
"$CHROME" --headless --disable-gpu --no-pdf-header-footer \
    --print-to-pdf=vector-index-carousel.pdf \
    --virtual-time-budget=9000 vector-index-carousel.html
echo "wrote vector-index-carousel.pdf"
