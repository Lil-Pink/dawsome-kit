# Upright Knight

An instrument set of dawsome-kit, committed as made: `upright-knight.sfz` plays the sounds in `samples/` on keys 0 to 127 (MIDI numbers, C4 = 60).

## Source

- re-voiced from public-domain (CC0) VCSL Knight recordings with a pedal and variation layer of the kit's own
- Made by: Dawsome, from VCSL's Knight upright recordings (CC0) and a pedal and variation layer of the kit's own
- From: this repository, committed as made
- Recordings: Versilian Community Sample Library (VCSL), at a pinned commit: Chordophones/Zithers/Upright Piano, Knight, Sustains; made by Versilian Studios LLC; https://github.com/sgossner/VCSL/tree/c1ea7bcc3c7309650ab0da9d15c9cd1fbc4a4c7e

## License

CC0 1.0 Universal, a public domain dedication: use, change, sell and share these sounds, in songs or in software, with no credit due.

- License page: https://creativecommons.org/publicdomain/zero/1.0/
- It says: "No Copyright. The person who associated a work with this deed has dedicated the work to the public domain by waiving all of his or her rights to the work worldwide under copyright law"
- Note: a fusion of two CC0 works: every note is one of VCSL's Knight upright recordings (sources.json vcsl, at its pinned commit, whose LICENSE is the CC0 1.0 legal code), the sustain pedal's ring, the per-note variation and the effects over it are the kit's own; the fusion is dedicated CC0 1.0. Each sample's MANIFEST.json entry names the recording it was made from (made_from)
- The recordings' license page: https://github.com/sgossner/VCSL (the CC0 1.0 legal code: https://github.com/sgossner/VCSL/blob/c1ea7bcc3c7309650ab0da9d15c9cd1fbc4a4c7e/LICENSE)

`set.json` holds the same facts with the set's figures, and the kit's `MANIFEST.json` traces every file to its source and SHA-256.

## How it was made

- an upright piano re-voiced from VCSL's Knight recordings (CC0), an upright sampled from the player's seat for VSCO 2 Pro: every note is its recording, with a layer of the kit's own over it; no felt
- the layer: the sustain pedal's sympathetic ring, one resonator for each of the 88 strings but the struck one, at its equal-tempered pitch (A4 = 440 Hz), ringing 1.5 s to 60 dB down, its peak 18 dB under the note's; and a per-note variation, a seeded tilt of up to 0.25 dB either way of the 10 kHz level against the 100 Hz level
- played through effects of its own, baked into the samples: a low shelf (-1 dB at 150 Hz), a high shelf (+1 dB at 6 kHz), a light tanh saturation (drive 0.5) and a small room (0.6 s decay, 8 ms predelay, 12 % wet)
- 45 notes a whole tone apart from A0 to C8, the notes the recording has, each playing itself and the key under it (A0 every key under it, C8 every key over it: keys 0-127); two velocity layers, 1-63 from the recording's soft layer and 64-127 from its loud one, rendered at velocities 48 and 104; one sample per note and layer
- every sample peaks at -1 dBFS (true peak -0.8 dBTP at most); the default amp_veltrack=100 turns velocity into loudness; ampeg_release=0.6 stands in for the damper (the recording's release and pedal-noise samples are left out)
- each note resampled once from the recording's 44.1 kHz to 48 kHz, started at its onset with a 0.25 ms fade-in, high-passed at 15 Hz, and ended where its 10 ms level has fallen 63 dB under its peak for good, faded out over 5 ms (1.7 to 30.9 s); a note whose recording stops while it still rings is faded out over its last 200 ms first
- 48 kHz stereo, rendered at 24 bits and stored as 16-bit FLAC: each sample dithered to 16 bits as y = rint(x * 32767 + a - b), clipped to 16 bits, where x is the render's 24-bit value over 2^23 and a, then b, are a frame-by-channel array of uniform noise on [0, 1) from NumPy's default_rng seeded with the first 8 bytes, big-endian, of the SHA-256 of the file's name without .flac, so the same render always gives the same 16-bit samples; each sample's MANIFEST.json entry names its render and that file's SHA-256 (member, made_from_sha256)
- volume lowered by 0.20 dB so a full-velocity note peaks at or below -1 dBFS (its loudest note reaches -0.80 dBFS when resampled)
