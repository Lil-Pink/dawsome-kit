# Bass Guitar

An instrument set of dawsome-kit, written by build.py: `bass-guitar.sfz` plays the sounds in `samples/` on keys 12 to 72 (MIDI numbers, C4 = 60).

## Source

- Karoryfer Fashionbass, its notes and clean mapping (fingers, neck pickup), at a pinned commit
- Made by: Karoryfer Samples: a Killer KB-Simmony bass guitar played and mapped by D. Smolken
- From: https://github.com/sfzinstruments/karoryfer.fashionbass/tree/0703973b300a6bacb837602be9b588ab64732a72

## License

CC0 1.0 Universal, a public domain dedication: use, change, sell and share these sounds, in songs or in software, with no credit due.

- License page: https://github.com/sfzinstruments/karoryfer.fashionbass/blob/0703973b300a6bacb837602be9b588ab64732a72/LICENSE
- It says: "Creative Commons Legal Code / CC0 1.0 Universal ... 2. Waiver. To the greatest extent permitted by, but not in contravention of, applicable law, Affirmer hereby overtly, fully, permanently, irrevocably and unconditionally waives, abandons, and surrenders all of Affirmer's Copyright and Related Rights and associated claims and causes of action"
- Note: The repository's LICENSE at this commit is the CC0 1.0 legal code, byte for byte this kit's LICENSE (read from the reference research's download of the commit, GitHub's zip of it, SHA-256 14aebbf78d7ba5d03c734a6dfddd24167676d39eb7a52ed0ab9ed5a77a426656). Its readme.txt says 'Killer KB-Simmony bass guitar played and mapped by D. Smolken.', 'All samples recorded direct, using neck pickup only, with no effects and the onboard EQ set flat.' and 'Royalty-free for all commercial and non-commercial use. Copyright 2014 Karoryfer Lecolds.'; under CC0 the copyright line asks nothing.

The changes made here (conversion, trimming, mapping) are dedicated CC0 too. `set.json` holds the same facts with the set's figures, and the kit's `MANIFEST.json` traces every file to its source and SHA-256.

## What build.py did

- Karoryfer's Fashionbass: a Killer KB-Simmony bass guitar played with the fingers, neck pickup, recorded direct with no effects, tuned in fifths (CGDA), its low C string also sampled tuned down to A and F#
- its own clean mapping (fashionbass_clean.sfz), every note: 19 notes from F#0 to A4 over keys 12-72, five velocity layers (pp 1-40, p 41-65, mf 66-85, f 86-110, ff 111-127), three round robins, which take turns (seq_length=3) where the source picks them at random; its noises (keys 84-96) and release samples are left out
- amp_veltrack=80 over equally loud layers, the closest the subset comes to the source's own curves (each layer rising to full at its top velocity): within 1.4 dB rms of them over velocities 20-127 (measured); ampeg_release=0.4 (the source's 0.15 cuts notes short in the DAW, whose release falls in decibels)
- 120 of the 285 samples retuned by 3 to 18 cents to A4 = 440 Hz (measured; those that hold one pitch), resampled with soxr; the 40 that glide (every F#0 and A0 sample and 10 of C1's 15: the low string, slack, starts sharp when plucked hard and falls into tune as it rings) are left as recorded
- each sample faded in from silence over its first 2 ms, or only up to its first moment within 20 dB of its peak when that comes sooner, which keeps the note's own rise, with 2 ms of silence before it, so that every note starts from silence, not on a step (the source's first frames clicked where it starts mid-wave or jumps at once); the samples' own 6.6 s length and fade-out are kept
- Karoryfer names Fashionbass's files by the pitch they sound (measured: b1 at 61.7 Hz), as its mapping places them
- each sample is normalized to -1 dBFS before the cut to 16 bits, and its region's volume brings it to one loudness for the whole set (the loudest 400 ms of each sample, EBU R128 momentary loudness): every key and layer plays equally loud and amp_veltrack alone turns velocity into loudness, so soft to loud is continuous across the layers, whose sound still changes with velocity; 27 of 285 samples, too sharp to reach it under -1 dBFS, play up to 2 dB under it
- volume lowered by 0.01 dB so a full-velocity note peaks at or below -1 dBFS (its loudest note reaches -0.99 dBFS when resampled)
