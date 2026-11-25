#!/usr/bin/env bash
# Opens the report (live-rebuilt PDF) and the slides (browser, localhost:3030) for editing.
# Ctrl+C stops both.
set -e
cd "$(dirname "$0")"

[ -d slides/node_modules ] || (cd slides && npm install)

trap 'kill 0' EXIT
(cd report && typst watch main.typ --open) &
(cd slides && npm run dev) &
wait
