# Dreamy EP

An instrument set of dawsome-kit, committed as made: a physically modelled tine electric piano made for Dawsome. `ep-dreamy.sfz` plays the sounds in `samples/` on keys 0 to 127 (MIDI numbers, C4 = 60).

## Source

- procedural / physically modelled synthesis, curated by ear
- Made by: Dawsome
- From: this repository, committed as made
- Recordings: none; every sound is synthesized by the model

## License

CC0 1.0 Universal, a public domain dedication: use, change, sell and share these sounds, in songs or in software, with no credit due.

- License page: https://creativecommons.org/publicdomain/zero/1.0/
- It says: "No Copyright. The person who associated a work with this deed has dedicated the work to the public domain by waiving all of his or her rights to the work worldwide under copyright law"
- Note: made for Dawsome from no recordings, by a physical model and effects of its own; dedicated CC0 1.0

`set.json` holds the same facts with the set's figures, and the kit's `MANIFEST.json` traces every file to its source and SHA-256.

## How it was made

- a tine electric piano made for Dawsome by a physical model, with no recordings: a struck steel tine whose own overtones sit at a free bar's mode ratios and die fast (the bell of the attack), a tone bar that swells in and rings on, and a magnetic pickup's curve that the tine swings through; harder notes bark because the tine swings further into the pickup's field, not because of a filter
- played through its own effects, baked into the samples: an EQ lift (+2 dB under 250 Hz, +3 dB around 1.2 kHz, +3.5 dB over 3 kHz), a two-voice chorus, a four-stage phaser sweeping 800 Hz to 6 kHz with its two sides opposite, a stereo tremolo at 1.76 Hz with a lighter one at 3.52 Hz over it, and light 1.5:1 compression
- 31 notes on C, D, F, G and A from C1 to C7, each key playing its nearest note (keys 0-127); three velocity layers (1-42, 43-84, 85-127) played at velocities 40, 80 and 112, so harder is brighter as well as louder; one sample per note and layer
- every sample peaks at -1 dBFS (true peak -0.5 dBTP at most); the default amp_veltrack=100 turns velocity into loudness; ampeg_release=0.25
- each note is held until it dies away: its sample starts at the note's onset with a 0.25 ms fade-in, is high-passed at 15 Hz, and ends where its 10 ms level has fallen 63 dB under its peak for good, faded out over 5 ms (4.8 to 29.3 s; the low notes ring longest)
- 48 kHz stereo, rendered at 24 bits and stored as 16-bit FLAC: each sample dithered to 16 bits as y = rint(x * 32767 + a - b), clipped to 16 bits, where x is the render's 24-bit value over 2^23 and a, then b, are a frame-by-channel array of uniform noise on [0, 1) from NumPy's default_rng seeded with the first 8 bytes, big-endian, of the SHA-256 of the file's name without .flac, so the same render always gives the same 16-bit samples; each sample's MANIFEST.json entry names its render and that render's SHA-256 (member, made_from_sha256)
- volume lowered by 0.07 dB so a full-velocity note peaks at or below -1 dBFS (its loudest note reaches -0.93 dBFS when resampled)
