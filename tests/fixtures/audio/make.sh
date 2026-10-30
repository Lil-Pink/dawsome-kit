#!/bin/sh
# SPDX-License-Identifier: GPL-3.0-or-later
# Makes ramp.flac, the FLAC fixture of test_kitlib and test_build, with the reference encoder
# (flac 1.5.0 made the committed file): 1000 frames, 22050 Hz, mono, 16 bits, a ramp. The
# encoder adds an 8 KiB PADDING block by default, which test_build's strip test needs.
# metaflac --list ramp.flac gives the expected values the tests use.
# usage: sh tests/fixtures/audio/make.sh   (needs python3 and flac on PATH)
set -eu
here=$(cd "$(dirname "$0")" && pwd)
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
python3 -I -c '
import struct, sys, wave
with wave.open(sys.argv[1], "wb") as out:
    out.setnchannels(1)
    out.setsampwidth(2)
    out.setframerate(22050)
    out.writeframes(b"".join(struct.pack("<h", i * 30 - 15000) for i in range(1000)))
' "$work/ramp.wav"
flac --silent --force -o "$here/ramp.flac" "$work/ramp.wav"
