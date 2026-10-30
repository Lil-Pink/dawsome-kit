#!/bin/sh
# SPDX-License-Identifier: GPL-3.0-or-later
# Makes the .7z fixtures from content/ with the 7-Zip program (7-Zip 26.04 made the committed
# ones). The tests compare what sevenzip.py extracts with content/, so the expected bytes come
# from 7-Zip and the plain files, never from the reader under test.
# usage: sh tests/fixtures/7z/make.sh   (from anywhere; needs 7z on PATH)
set -eu
here=$(cd "$(dirname "$0")" && pwd)
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
cp -r "$here/content/." "$work/"
mkdir -p "$work/emptydir"
cd "$work"
members="hello.txt dir emptydir ramp.bin naïve*"
pack() { name=$1; shift; rm -f "$here/$name"; 7z a -t7z -bso0 -bsp0 "$@" "$here/$name" $members; }
pack lzma2.7z -m0=LZMA2
pack lzma.7z -m0=LZMA
pack copy.7z -m0=Copy -mhc=off
pack delta.7z -m0=Delta:4 -m1=LZMA2
pack bcj.7z -m0=BCJ -m1=LZMA2
pack bcj-lzma.7z -m0=BCJ -m1=LZMA
pack nonsolid.7z -m0=LZMA2 -ms=off
pack bzip2.7z -m0=BZip2
pack deflate.7z -m0=Deflate
pack ppmd.7z -m0=PPMd
pack encrypted.7z -m0=LZMA2 -pfixture -mhe=on
