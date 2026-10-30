# Electric Piano

An instrument set of dawsome-kit, written by build.py: `ep.sfz` plays the sounds in `samples/` on keys 27 to 98 (MIDI numbers, C4 = 60).

## Source

- FreePats FM Synthesized Piano #1, version 2019-09-16 (SFZ + FLAC)
- Made by: Roberto, for FreePats, recorded from the Hexter synthesizer (the Yamaha DX7's 'E. Piano 1' patch)
- From: https://github.com/freepats/fm-piano1/releases/download/2019-09-16/FM-Piano1-SFZ+FLAC-20190916.7z

## License

CC0 1.0 Universal, a public domain dedication: use, change, sell and share these sounds, in songs or in software, with no credit due.

- License page: https://freepats.zenvoid.org/ElectricPiano/synthesized-piano.html
- It says: "Published under the terms of the Creative Commons CC0 1.0 public domain dedication."
- Note: The quote is the page's line under 'FM Synthesized Piano #1, Version 2019-09-16', whose SFZ FLAC download is this archive (the page as the kit's reference research saved it on 2026-10-27). The archive itself holds the CC0 1.0 legal code (LICENSE.txt) and a README.txt that says 'Published under the terms of Creative Commons CC0 public domain dedication'.

The changes made here (conversion, trimming, mapping) are dedicated CC0 too. `set.json` holds the same facts with the set's figures, and the kit's `MANIFEST.json` traces every file to its source and SHA-256.

## What build.py did

- FreePats' FM Synthesized Piano #1: the Yamaha DX7's 'E. Piano 1' patch, the FM electric piano of many 1980s songs, recorded from the Hexter synthesizer, a model of the DX7
- FreePats' own mapping, unchanged: 12 notes a tritone apart from F#1 to C7, spread over keys 27-98, three velocity layers (1-69, 70-109, 110-127) recorded at velocities 60, 80 and 100, each sample at full scale; the default amp_veltrack=100 makes velocity the loudness, as in the source; ampeg_release=0.4, the source's
- the 24-bit samples are stored as 16-bit FLAC (triangular dither), otherwise unchanged: each starts from silence, and its tail falls into the dither, about 96 dB under its peak (the source's last 20 ms are 100 to 130 dB down)
- volume lowered by 1.28 dB so a full-velocity note peaks at or below -1 dBFS (its loudest note reaches +0.27 dBFS when resampled)
