# Electric Guitar

An instrument set of dawsome-kit, written by build.py: `guitar-electric.sfz` plays the sounds in `samples/` on keys 35 to 86 (MIDI numbers, C4 = 60).

## Source

- FreePats FSBS Electric Guitar Clean #1, version 2026-08-07, bridge pickup (SFZ + FLAC)
- Made by: FreePats: a Fender electric guitar sampled direct, processed through an amplifier and effects rack
- From: https://github.com/freepats/electric-guitar-FSBS-clean/releases/download/2026-08-07/EGuitarFSBS-clean-SFZ+FLAC-20260807.7z

## License

CC0 1.0 Universal, a public domain dedication: use, change, sell and share these sounds, in songs or in software, with no credit due.

- License page: https://freepats.zenvoid.org/ElectricGuitar/clean-electric-guitar.html
- It says: "Published under the terms of the Creative Commons CC0 1.0 public domain dedication."
- Note: The archive's SHA-256 is the digest GitHub states for the release asset (freepats/electric-guitar-FSBS-clean, tag 2026-08-07); the repository's LICENSE.txt is the CC0 1.0 legal code.

The changes made here (conversion, trimming, mapping) are dedicated CC0 too. `set.json` holds the same facts with the set's figures, and the kit's `MANIFEST.json` traces every file to its source and SHA-256.

## What build.py did

- FreePats' FSBS Electric Guitar Clean #1: a Fender electric guitar, bridge pickup, recorded direct and processed through an amplifier and effects rack into a clean (undistorted) stereo sound; Dawsome's Saturator insert drives it into an amp sound
- FreePats' own mapping: 19 notes over keys 35-86, two velocity layers (soft 1-92, normal 93-127) up to E4, one above; the first of each note's four random round robins (lorand/hirand are outside the subset)
- loudness follows velocity (the default amp_veltrack=100), as in the source, over equally loud layers; ampeg_release=0.3 for crisper stabs (the source's 1.1)
- 10 samples retuned by 3 to 19 cents to A4 = 440 Hz (measured; the A2 soft layer was 21 cents under its normal layer), resampled with soxr
- each sample (the sources ring for up to 39 s) started 15 ms before its first moment within 20 dB of its peak, which keeps the pluck's own rise, faded in over the first 2 ms (unchanged when it starts sooner), but never more than 1 ms before its first moment within 60 dB of its peak, faded in over that 1 ms (a lead that began on near-silence played as the note starting late), cut where it stays 60 dB under its peak, cut at 5 s at most, fading out over the last 2 s
- dropped width at its SFZ default (no effect)
- each sample is normalized to -1 dBFS before the cut to 16 bits, and its region's volume brings it to one loudness for the whole set (the loudest 400 ms of each sample, EBU R128 momentary loudness): every key and layer plays equally loud and amp_veltrack alone turns velocity into loudness, so soft to loud is continuous across the layers, whose sound still changes with velocity; 1 of 30 samples, too sharp to reach it under -1 dBFS, play up to 2 dB under it
- volume lowered by 0.03 dB so a full-velocity note peaks at or below -1 dBFS (its loudest note reaches -0.97 dBFS when resampled)
