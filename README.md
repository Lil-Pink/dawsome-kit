# dawsome-kit

The instrument sets for [Dawsome](https://github.com/dawsome), an open-source digital audio
workstation: a piano, three basses, an organ, an electric piano, two pads, strings, an
acoustic and an electric guitar, a marimba, two drum kits and a folder of one-shots and
loops, about 142 MB in all.
Every sound in it is CC0 at its source, so you may use, change, sell and redistribute it, in
songs or in software, with no credit due.

Each set is a folder under `sets/`:

```
sets/piano/
  piano.sfz      the regions, written with only the SFZ opcodes Dawsome reads
  samples/       the sounds, 16-bit FLAC at the source's rate (or the source's own FLAC)
  set.json       name, maker, `licence` with a quote from the source's license page, source,
                 size on disk, size once decoded, the peak of a full-velocity note, notes
                 (and, in the sets added later, the Library group as `category`)
  README.md      the guitars, the bass guitar and the electric piano: the source, maker,
                 license and notes in plain words
```

`sets/one-shots/` is the exception: plain sound files for the sampler, no `.sfz`
(`"kind": "samples"` in its `set.json`). `MANIFEST.json` lists every file under `sets/` with
the archive or repository file it came from, its path there, its `licence`, its `licence_page`
and its SHA-256 (an acoustic drum hit, mixed from several microphones' files, names each of
them with its share of the mix under `mixed_from`), plus the download sets that are too big
for this repository.

## The sets

Decoded size is what a set takes in memory once its samples are decoded to 32-bit float at
their own rate, which is how Dawsome preloads them (a 48 kHz project adds about 9 % for
44.1 kHz files). `set.json` holds the exact figures; this table rounds them.

| Set | Name | Files | On disk | Decoded | Keys | Source | License page |
|-|-|-|-|-|-|-|-|
| `piano` | Upright Piano | 66 | 11.2 MB | 114.7 MB | 21–108 | FreePats Upright Piano KW 2022-02-21 | [FreePats](https://freepats.zenvoid.org/Piano/acoustic-grand-piano.html) |
| `bass-electric` | Electric Bass | 12 | 1.0 MB | 8.4 MB | 26–45 | FreePats Bass Guitar YR, finger, 2019-09-30; notes start from silence | [FreePats](https://freepats.zenvoid.org/ElectricGuitar/clean-electric-bass.html) |
| `bass-synth` | Synth Bass | 9 | 0.9 MB | 9.3 MB | 28–88 | FreePats Synth Bass #2, 2021-04-05 | [FreePats](https://freepats.zenvoid.org/Synthesizer/synth-bass.html) |
| `bass-guitar` | Bass Guitar | 285 | 52.6 MB | 329.6 MB | 12–72 | Karoryfer Fashionbass (fingers, neck pickup; tuned in fifths, its low string down to F#0), 5 velocity layers, 3 round robins; retuned where a note holds one pitch, layers made equally loud, notes start from silence | [LICENSE](https://github.com/sfzinstruments/karoryfer.fashionbass/blob/0703973b300a6bacb837602be9b588ab64732a72/LICENSE) |
| `organ` | Drawbar Organ | 16 | 1.9 MB | 13.0 MB | 33–98 | FreePats Drawbar organ emulation, 2019-07-12; velocity leaves its level alone, as on a drawbar organ, played at the other sets' loudness | [FreePats](https://freepats.zenvoid.org/Organ/electric-organ.html) |
| `ep` | Electric Piano | 36 | 9.0 MB | 87.6 MB | 27–98 | FreePats FM Synthesized Piano #1, 2019-09-16 (the DX7's E. Piano 1, recorded from Hexter) | [FreePats](https://freepats.zenvoid.org/ElectricPiano/synthesized-piano.html) |
| `pad-choir` | Choir Pad | 12 | 6.9 MB | 28.8 MB | 33–108 | FreePats Synth Pad Choir, 2020-05-16 | [FreePats](https://freepats.zenvoid.org/Synthesizer/synth-pad.html) |
| `pad-sweep` | Sweep Pad | 13 | 3.1 MB | 16.8 MB | 33–96 | FreePats Sweep Pad, 2019-08-13 | [FreePats](https://freepats.zenvoid.org/Synthesizer/synth-pad.html) |
| `strings` | Strings | 34 | 15.8 MB | 57.7 MB | 31–93 | VSCO 2 Community Edition, cello and violin section sustains; loops run past their end | [Versilian Studios](https://versilian-studios.com/vsco-community/) |
| `guitar-acoustic` | Acoustic Guitar | 102 | 9.8 MB | 103.6 MB | 36–88 | a real pluck from Karoryfer Shinyguitar v1.002's microphone samples (archtop) driving a modelled steel string and body, 3 velocity layers, 2 round robins | [LICENSE](https://github.com/sfzinstruments/karoryfer.shinyguitar/blob/57243cca85277dbcc120ce17c6178032f93c80f3/LICENSE) (the pluck); the model is the kit's own, CC0 |
| `guitar-electric` | Electric Guitar | 30 | 5.9 MB | 57.6 MB | 35–86 | FreePats FSBS Electric Guitar Clean #1, 2026-08-07, 2 velocity layers; retuned, layers made equally loud | [FreePats](https://freepats.zenvoid.org/ElectricGuitar/clean-electric-guitar.html) |
| `mallets` | Marimba | 30 | 2.6 MB | 29.9 MB | 36–100 | VCSL marimba; tails that end in the recording's hiss fade into the cut | [VCSL](https://github.com/sgossner/VCSL) |
| `drums-acoustic` | Acoustic Drums | 84 | 9.8 MB | 54.5 MB | 36–53 | Karoryfer Big Rusty Drums, sticks: each hit its close and overhead microphones mixed as the library's basic kit sets them, 4 velocity layers, 2 round robins; hits start from silence | [LICENSE](https://github.com/sfzinstruments/karoryfer.big-rusty-drums/blob/f07ce00df34a46b6b08375be56fe116cf15782bc/LICENSE) |
| `drums-electronic` | Electronic Drums | 37 | 0.8 MB | 4.0 MB | 24–60 | Sonic Pi samples: kicks, snares, hi-hats; hits start from silence; `hat_gnu` made mono (its channels canceled) | [LICENSE.md](https://github.com/sonic-pi-net/sonic-pi/blob/df6f17e4625b0451fa2af8accf2e1e641c7ad458/LICENSE.md) |
| `one-shots` | One-shots and Loops | 34 | 10.9 MB | 46.9 MB | — | Sonic Pi samples: ambient, glitch, misc, vinyl, loops | [LICENSE.md](https://github.com/sonic-pi-net/sonic-pi/blob/df6f17e4625b0451fa2af8accf2e1e641c7ad458/LICENSE.md) |

**Download set** (in `MANIFEST.json` under `downloads`, not stored here): `piano-grand`,
Osiris Piano v0.925 (Versilian Studios and Karoryfer Samples, CC0), the upstream release zip
of 459 MB with its URL and SHA-256. Its program switches microphones, sustain-pedal and noise
variants with CC conditions that Dawsome's SFZ subset does not read, and it decodes to 5.5 GB,
so as published it is not playable as intended; it needs a trimmed build or more of SFZ in
the DAW. `MANIFEST.json` lists the opcodes it uses beyond the subset.

## What the DAW can count on

- Only these SFZ headers and opcodes appear: `<control>` with `default_path`, `<global>`,
  `<region>`, `sample`, `lokey`, `hikey`, `pitch_keycenter`, `lovel`, `hivel`, `loop_mode`,
  `loop_start`, `loop_end`, `ampeg_release`, `amp_veltrack`, `volume`, `seq_length`,
  `seq_position`, and comments. `kitlib.SFZ_OPCODES` is the full list allowed, the DAW's
  subset; `check.py --sfz` enforces it.
- Keys are MIDI numbers, with C4 = 60. Versilian's libraries name notes with C3 = 60; the
  `.sfz` files here are already converted (measured: VSCO's violin A3 sounds at 441 Hz).
  Karoryfer names the files of Fashionbass (the bass guitar) and Shinyguitar by the pitch they
  sound (measured: the bass guitar's `b1` sounds at 61.7 Hz, B1 = 35).
- Every looping region gives `loop_mode`, `loop_start` and `loop_end` (inclusive) in the
  `.sfz`; no sample relies on loop points embedded in the file. The strings' loops, made by
  `build.py`, run 10 ms past `loop_end` with the loop's own start, since the DAW's resampling
  on load reads about 2 ms past the frame it plays (ending at `loop_end`, they clicked at
  every jump back). FreePats writes `loop_end` as the frame after the loop, its files holding
  the loop's first frames again from there, so `build.py` writes it less one: the FreePats loops
  run 8 to 16 frames past `loop_end` with the loop's own start (read as written, every wrap
  played the loop's first frame twice, measured with `compare.py` on 2026-10-27).
- The acoustic guitar is made, not only converted (2026-10-26, chosen by ear among the
  guitar-tone candidates): the pluck of each of Shinyguitar's microphone recordings (the
  recording with its string's own tone taken out) plucks a modelled steel string, heard
  through a modelled body; three velocity layers, two
  round robins (`seq_length=2`). The recorded archtop alone was too twangy, like a banjo, and a
  model alone sounded synthesized. Its files are committed as made.
- The electric guitar (2026-10-26, picked by ear from the guitar-tone candidates) and the bass
  guitar (2026-10-27, Fashionbass, picked by ear) are processed past conversion:
  retuned to A4 = 440 Hz by resampling (the bass only where a sample holds one pitch: its slack
  low string starts sharp and glides into tune, which no fixed retune follows), every key and
  layer made equally loud so `amp_veltrack` alone sets loudness; `build.py` says how and why.
- The electric bass's and the bass guitar's notes and both drum kits' hits start on 2 ms of
  silence, their first frames faded in up to the pluck or hit: their sources start mid-wave or
  jump at their first frame, which clicked (up to -26 dBFS; Fashionbass's first frames sit up
  to 42 dB under their peak, Big Rusty's hits reach 20 dB under theirs within 0 to 56 frames);
  `build.py` `ONSET` says how and why.
- The acoustic drums are mixed, not only converted (2026-10-27, Big Rusty Drums,
  picked by ear): each hit is one sample of its close and overhead microphones at the levels and
  pans the library's basic kit (`Programs/02-basic.sfz`) gives them at its controllers'
  defaults, which the subset cannot say, a close mic placed by Dawsome's own pan law for a mono
  zone; so a hit plays as Dawsome would play those microphones layered (measured through
  `dawsome render`: within 84 dB of the peak). `build.py` `BIG_RUSTY_KIT` says how and why.
- A full-velocity note of any instrument set peaks at or below -1 dBFS: each set's volume is
  trimmed against the worst of three resampled measurements of every sample, with its
  region's volume (`set.json` `peak_db`, `build.py` `PLAYBACK`).
- Sample rates: 44.1 kHz, except the electric guitar (48 kHz) and two
  Sonic Pi hi-hats in the electronic drums (`hat_raw` 96 kHz, `hat_sci` 192 kHz, kept at
  their own rate).

## Rebuilding

Python 3.11 or later, and for `build.py` an ffmpeg built with libsoxr. Run from anywhere:

```sh
python3 -I tools/fetch.py          # download every source, check its pinned SHA-256, extract it
python3 -I tools/build.py          # write sets/ and MANIFEST.json from the downloads
python3 -I tools/check.py          # sets/ against MANIFEST.json and the SFZ subset
python3 -I -m unittest discover -s tests -t .    # from the repository folder
```

Downloads go to `downloads/` (ignored by git; it may be a link to a roomy drive), or wherever
`--downloads DIR` says: about 1.2 GB to download (2.0 GB on disk with the extracted copies),
about two fifths of it the optional Osiris Piano.
`fetch.py --check` verifies the downloads against the pins without downloading;
`build.py SET ...` rebuilds only those sets. The same downloads give the same bytes:
conversion runs ffmpeg with its bitexact flags, and `check.py` compares every file with its
recorded hash.

The acoustic guitar is not built here: its samples, `.sfz`, `set.json` and `README.md` are
committed as they were made, and `build.py` keeps them and their manifest entries as they are.

To add or change a source: read its license page, add it to `sources.json` with a quote from
that page, run `fetch.py --pin SOURCE` (which records the SHA-256 of what it downloads), then
describe the set in `build.py`'s `plans()`.

## Measuring the sets

`compare.py` measures every set Dawsome ships (these and the Dawsome repository's
`assets/instruments/` and starter kit) through Dawsome's own offline render, and writes a
listening set normalized to -18 LUFS; with `--refs INDEX` it does the same for other sample
libraries of the same instruments (an SFZ read as Dawsome plays it at default controllers, a
pack of sound files mapped by note name or pitch detection), so ours and theirs go through one
sampler. It needs numpy, scipy and soundfile and a built `dawsome`; its docstring says what
it measures and how. Nothing it reads or writes enters this repository:

```sh
python3 -I tools/compare.py --out DIR [--refs INDEX] [--dawsome BIN]
```

## Provenance policy

- A source is used only when its own license page (the publisher's page or the LICENSE file
  in its repository, never a blog or a listing) dedicates it to the public domain under CC0
  or states public domain. The page is quoted in `sources.json` and in each `set.json`.
- A source whose page is unreachable or unclear is left out. VSCO 2 CE's 2016 `Readme.txt`
  asks that the samples not be sold directly and that Versilian be credited; under CC0 these
  are requests, not conditions, and the publisher's current page says "no rules", so it is
  used (both are recorded in `sources.json`).
- Each archive or file is pinned by SHA-256 on first download and checked on every run.
  Files from GitHub come from a pinned commit; the first downloads were also checked against
  the commit's Git blob hashes.
- Sonic Pi's sounds come from Freesound uploads marked CC0. Each one's Freesound page was read
  (2026-10-22) and its URL is in `MANIFEST.json` (`upstream`). A sound is left out when its
  page is gone or says it comes from someone else's recording, product or sample set: the
  Amen break loops, a loop made from an Ableton Live kit, a keyboard's built-in loop, a loop
  with another user's sound whose page is gone, a sound from a commercial app, a hi-hat from a
  third-party TR-606 sample set, one described as from a Daft Punk song, two scratches of
  unnamed records, two sounds whose pages are gone, and the whole `elec_` group (one
  uploader's compilation whose file names point to an E-mu SP-12, another DAW's content and a
  "virus kit"). `build.py` lists each with its reason (`SONIC_PI_LEFT_OUT`).
- Archives are untrusted data: they are read with Python's own modules (and `sevenzip.py` for
  .7z), nothing in them is run, and member names that are absolute, climb out with `..`, or
  name links are refused before anything is written.
- Karoryfer's libraries (the bass guitar, the acoustic guitar, the acoustic drums): each
  repository holds the CC0 1.0 legal code at the pinned commit (Shinyguitar's, put there by its
  author D. Smolken, replaced a CC BY 4.0 file on 2022-11-04, "Swap to CC0"; Fashionbass's and
  Big Rusty Drums' are byte for byte the legal code in this repository's `LICENSE`). Their readmes keep older "royalty-free,
  copyright Karoryfer Lecolds" lines, which under CC0 ask nothing; `sources.json` records
  both.
- The acoustic guitar is a fusion of two CC0 works: the pluck from Shinyguitar's recordings,
  the string and body from a model made for the kit, whose sounds are dedicated CC0 too. Its
  samples' manifest entries say `made_by: made` (committed as made) and name the recording each
  sample was made from (`made_from`); `check.py` fails when a recording named there is not a
  file `sources.json` pins for a CC0 source.
- CC0 is only as good as the person who applied it; the manifest exists so that a single file
  can be traced to its source and removed if a claim against it ever turns out to be right.

## Licenses

- The sounds, the `.sfz` and `set.json` files and `MANIFEST.json`: CC0 1.0 Universal
  (`LICENSE`). The sources dedicated their work CC0; the changes made here (conversion,
  trimming, loops, mapping, and the acoustic guitar's modelled string and body) are dedicated
  CC0 too.
- The scripts and tests (`*.py`, `tests/`): GNU General Public License version 3 or
  later (`LICENSE-SCRIPTS`; each file carries `SPDX-License-Identifier: GPL-3.0-or-later`).
