#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Builds the kit's sets from the downloaded sources, then MANIFEST.json.

usage: python3 -I tools/build.py [--downloads DIR] [SET ...]
       --downloads DIR: where fetch.py put the sources (default: downloads/ in the repository)
       SET: only these sets (default: all); MANIFEST.json keeps the other sets' entries

Each set becomes sets/<name>/: <name>.sfz, whose regions use only the SFZ opcodes Dawsome
reads (kitlib.SFZ_OPCODES), its samples in samples/, and set.json (name, maker, `licence` and a
quote from the source's license page, source, size on disk and size once decoded), with
README.md beside it for the sets whose plan asks for one. A set of
kind "samples" (the one-shots) has no .sfz. MANIFEST.json then lists every file under sets/
with its source, the archive or repository file it came from, its path there, its `licence`
and its SHA-256, and the download sets: sources too big for the repository, which the DAW
fetches itself by URL and hash. The acoustic guitar is the one set not built here: its files
are committed as they were made (COMMITTED), and build.py keeps them and their manifest
entries, each sample's naming the recordings it was made from.

Samples keep their source names, as .flac. A source FLAC of 16 bits or fewer that needs no
change is copied byte for byte. Every other sample goes through ffmpeg into 16-bit FLAC at the
source's rate and channel count (24-bit sources with triangular dither); where a set needs
it, build.py mixes a hit's microphones into it (Zone.mics; its manifest entry names every file),
trims it or makes a loop in it first (see Shape). ffmpeg runs with its bitexact flags, so the
same sources give the same bytes.

Needs ffmpeg on PATH; build.py itself uses only the standard library. Exit 0 when built, 2
when a step cannot run.
"""

import argparse
import math
import operator
import posixpath
import re
import shutil
import subprocess
import sys
from array import array
from dataclasses import dataclass, field
from pathlib import Path

# -I keeps the script's own folder off sys.path (so a planted module in the working folder
# is never imported); this adds back exactly the repository's folder for kitlib.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import kitlib  # noqa: E402
from kitlib import Problem  # noqa: E402

# Into 16-bit FLAC at the input's rate and channels. bitexact keeps ffmpeg's version out of
# the file; triangular dither makes the cut from 24 bits noise rather than distortion (a 16-bit
# input needs no conversion, so it passes through unchanged).
DITHER = "aresample=osf=s16:dither_method=triangular"
# Into 16 bits by rounding alone, for a 16-bit source changed only at its start (Shape.onset): its
# other samples round to themselves, so they stay bit for bit as recorded, where dither would put
# its noise on every one of them.
ROUND = "aresample=osf=s16"
FLAC_OUTPUT = ("-map_metadata", "-1", "-fflags", "+bitexact", "-flags:a", "+bitexact",
               "-c:a", "flac", "-compression_level", "8")
FLAC_PADDING = 1
PEAK_DB = kitlib.PEAK_DB
# Playback the peak measurement stands in for: oversampled 8 times (the waveform between the
# samples) and resampled to a 48 kHz project (the DAW resamples on load). Bright sounds such
# as hi-hats have so much near the top of the band that each reconstruction filter gives a
# different peak (Sonic Pi's hat_snap: +1.72 to +2.20 dB), so the worst of three counts.
PLAYBACK = ("aresample=resampler=soxr:precision=28:osf=flt:osr={eight}",
            "aresample=resampler=soxr:precision=28:osf=flt:osr=48000",
            "aresample=osf=flt:osr={eight}:filter_size=64")
# Opcodes the DAW lacks whose values here are SFZ's defaults, so dropping them changes nothing.
NO_EFFECT = {"width": "100"}
# The order opcodes are written in, after the sample.
REGION_ORDER = (
    "lokey", "hikey", "pitch_keycenter", "lovel", "hivel", "seq_length", "seq_position",
    "loop_mode", "loop_start", "loop_end", "offset", "end", "transpose", "tune", "volume",
    "pan", "amp_veltrack", "ampeg_attack", "ampeg_release",
)
# Opcodes that place a region; never moved into <global> even when every region shares one.
PLACEMENT = {"lokey", "hikey", "pitch_keycenter", "lovel", "hivel", "loop_start", "loop_end",
             "seq_position"}
# Versilian (VCSL, VSCO 2 CE) names notes with C3 as MIDI 60, an octave under SFZ's C4 = 60:
# measured 2026-10-22, VSCO's VlnEns_susVib_A3 sounds at 441 Hz (MIDI 69) and VCSL's marimba
# C4 at 525 Hz (MIDI 72), and every other note checked agrees.
VERSILIAN_OCTAVE = 12
NAME_CHARACTERS = re.compile(r"[A-Za-z0-9_#+.-]+")
# normalize="loudness": the one loudness every region plays at is the set's quietest sample's
# plus this much. A sample's loudness once its peak sits at PEAK_DB is set by its crest (a sharp
# pluck is quieter than a round one of the same peak), so aiming at the quietest alone made the
# guitar candidates 2.5 to 5 dB quieter than today's sets; the few samples that cannot reach the
# aim without peaking above PEAK_DB play at their peak's limit instead, at most this much under.
LOUDNESS_SLACK_DB = 2.0
# How far build.py's loops run past their end, with the loop's own start, which is what playback
# reads next: five times the reach of the DAW's resampling on load (see ONSET), which reads past
# the point it plays whatever the file holds there. A file that stopped at its loop's end gave it
# silence past the seam, which bent the last 2 ms before every jump back (the strings, 2026-10-26:
# their B1 loud layer clicked at -21 dBFS every 2 s; against the loop unrolled into a plain file,
# all 34 of their loops bent, by -27 to -63 dBFS, and none with this tail).
LOOP_TAIL_SECONDS = 0.01


@dataclass(frozen=True)
class Shape:
    """What build.py does to a sample between decoding and encoding it.

    floor_db: cut the sample where it stays below this level (dB under its peak), with a
        short fade over the near-silence after it: removes the long silent tails that would
        otherwise be decoded and held in memory.
    hiss_fade: with floor_db, for a sample whose tail is the recording's hiss: from where the
        decay sinks into the hiss to the cut, a gain falling linearly in decibels that brings
        the hiss down to floor_db by the cut, so the tail fades into it instead of stopping at
        the hiss's level (see hiss_fade_start for how the hiss is found). Nothing when the hiss
        is already under floor_db.
    max_seconds: cut the sample at most this long, fading out over `fade_seconds` before the
        cut (for cymbals that ring far longer than a note needs).
    loop: (earliest, latest, length, crossfade) in seconds: make a loop of `length` starting
        between `earliest` and `latest` where the level holds steadiest (a recorded swell or
        bow change inside a loop would pulse at every repeat), cut the sample
        LOOP_TAIL_SECONDS past the loop's end with the loop's own start there, and blend the
        loop's last `crossfade` seconds into the sound just before its start, so the jump back
        is seamless. The blend keeps the level steady whatever the
        two stretches have in common (see crossfade_gains).
    start_db: cut what comes before the first frame within this level (dB under the peak),
        keeping `start_seconds` before it, faded in from silence: removes the pre-note noise
        some sources keep (Karoryfer's basses hold about 25 ms of finger noise before the
        pluck), which would otherwise make every note sound late on the piano roll. A sample
        whose note starts within `start_seconds` is left as it is. Applied before floor_db and
        max_seconds, so a cap counts from the new start.
    start_fade_seconds: fade in over only this much of the lead (None: all of it). A lead longer
        than its fade keeps the pluck's own rise, the finger or pick meeting the string, at its
        recorded level instead of fading it (the guitar candidates, 2026-10-26).
    silence: (level_db, seconds), with start_db: keep at most `seconds` before the sample's first
        frame within `level_db` of its peak, faded in over those `seconds`, so that a start cut's
        lead never begins on a stretch of near-silence (which plays as the note starting late);
        a lead that holds sound from its first frame is cut as start_db says.
    tone: a Tone run over the sample after the cuts above (not with a loop: a filter's memory
        would not carry over the loop's seam).
    onset: (lead, fade, level_db) in seconds and dB: start the sample from silence, for sources cut
        so close to their note that they start on a step from silence, which clicks (see ONSET).
        The recording fades in over its first `fade`, or only up to its first moment within
        `level_db` of its peak when that comes sooner, so the note's rise from there on stays as
        recorded, and `lead` of silence goes before it, so that the DAW's resampling on load
        starts on silence instead of in the note's first frames. Not with start_db (which keeps a
        lead of the recording's own) or a loop.
    """

    floor_db: float | None = None
    hiss_fade: bool = False
    max_seconds: float | None = None
    fade_seconds: float = 0.05
    loop: tuple | None = None
    start_db: float | None = None
    start_seconds: float = 0.005
    start_fade_seconds: float | None = None
    silence: tuple | None = None
    tone: "Tone | None" = None
    onset: tuple | None = None

    def describe(self):
        if self.loop:
            earliest, latest, length, crossfade = self.loop
            return (f"a {length:g} s loop made by build.py, starting between {earliest:g} s and"
                    f" {latest:g} s where the level holds steadiest, with a {crossfade:g} s"
                    " level-keeping crossfade; the sample runs"
                    f" {LOOP_TAIL_SECONDS * 1000:g} ms past the loop's end with the loop's own start,"
                    " which is what the DAW's resampling reads across the seam (a file ending at"
                    " the loop's end bent the sound before every jump back: a click)")
        parts = []
        if self.onset is not None:
            lead, fade, level_db = self.onset
            parts.append(f"faded in from silence over its first {fade * 1000:g} ms, or only up to"
                         f" its first moment within {-level_db:g} dB of its peak when that comes"
                         " sooner, which keeps the note's own rise, with"
                         f" {lead * 1000:g} ms of silence before it, so that every note starts from"
                         " silence, not on a step (the source's first frames clicked where it"
                         " starts mid-wave or jumps at once)")
        if self.start_db is not None and self.start_fade_seconds is None:
            parts.append(f"started {self.start_seconds * 1000:g} ms before its first moment"
                         f" within {-self.start_db:g} dB of its peak, faded in over those"
                         f" {self.start_seconds * 1000:g} ms (unchanged when it starts sooner)")
        elif self.start_db is not None:
            parts.append(f"started {self.start_seconds * 1000:g} ms before its first moment"
                         f" within {-self.start_db:g} dB of its peak, which keeps the pluck's own"
                         f" rise, faded in over the first {self.start_fade_seconds * 1000:g} ms"
                         " (unchanged when it starts sooner)")
        if self.silence is not None:
            level_db, seconds = self.silence
            parts.append(f"but never more than {seconds * 1000:g} ms before its first moment"
                         f" within {-level_db:g} dB of its peak, faded in over that"
                         f" {seconds * 1000:g} ms (a lead that began on near-silence played as the"
                         " note starting late)")
        if self.floor_db is not None:
            parts.append(f"cut where it stays {-self.floor_db:g} dB under its peak")
        if self.hiss_fade:
            parts.append("its tail, where the decay has sunk into the recording's hiss (the"
                         f" first 50 ms within {HISS_MARGIN_DB:g} dB of the hiss's level), faded"
                         " linearly in decibels down to the hiss"
                         f" {-self.floor_db:g} dB under the peak at the cut, so it fades into the"
                         " cut instead of stopping at the hiss's level")
        if self.max_seconds is not None:
            parts.append(f"cut at {self.max_seconds:g} s at most, fading out over the last"
                         f" {self.fade_seconds:g} s")
        if self.tone is not None:
            parts.append(f"then {self.tone.about}")
        return ", ".join(parts)


@dataclass(frozen=True)
class Tone:
    """An ffmpeg filter chain run over a sample after its Shape's cuts, in 32-bit float at the
    sample's own rate and channels, and what it does in plain words for the set's notes. ffmpeg's
    filters are deterministic, so the same sources still give the same bytes. Added for the
    guitar candidates (2026-10-26): body EQ, an amp and a compressor, baked in because the DAW's
    SFZ subset has no filter or effect opcodes."""

    chain: str
    about: str


@dataclass
class Zone:
    """One region of a set: its source file (a path inside the source's archive or
    repository), its opcodes (subset only, values as written), and how its sample is made."""

    member: str
    opcodes: dict
    lowpass: float | None = None  # Hz: the source's static 2-pole low-pass, baked in
    shape: Shape | None = None
    upstream: str | None = None  # the page the sound itself came from, when the source has one
    # Cents the sample is raised by, baked in by resampling: a measured tuning correction (the
    # subset's tune opcode is not among those the guitar sets may use, so the pitch is fixed in
    # the sample itself). Negative lowers it.
    retune: int = 0
    # The one channel of a stereo source the sample keeps, as a mono file (INVERTED_STEREO); None
    # keeps them all.
    channel: int | None = None
    # A multi-microphone recording's files, mixed into this one sample (see mixed); the first is
    # `member`, which names the sample. Empty: `member` alone.
    mics: tuple = ()

    @property
    def sample(self):
        return stem(self.member) + ".flac"


@dataclass(frozen=True)
class Mic:
    """One microphone's file of a hit recorded with several at once, and how it goes into the
    mix: its gain (amplitude) and, for a mono file, its place left to right, -1 to 1, by
    Dawsome's own pan law (pan_gains); a stereo file keeps its own image."""

    member: str
    gain: float
    pan: float = 0.0


def stem(path):
    """A file name without its folder and extension: stem("a/b/C#4.wav") == "C#4"."""
    return posixpath.splitext(posixpath.basename(path))[0]


@dataclass
class Plan:
    name: str  # folder under sets/
    title: str
    source: str  # sources.json id
    zones: list
    notes: list = field(default_factory=list)  # what build.py changed from the source, and why
    kind: str = "instrument"  # or "samples": plain files, no .sfz
    derived_from: str | None = None  # the source .sfz the regions come from
    # Quiet sources lose detail in 16 bits, so each sample can be normalized to PEAK_DB, its
    # region's volume putting it back at its recorded level against the loudest sample of the
    # whole set ("set": the recorded layers are the dynamics, with amp_veltrack=0) or of its
    # velocity layer ("layer": layers change the timbre and amp_veltrack the loudness, as in
    # the FreePats piano, whose every sample peaks at full scale). "loudness": every region plays
    # at one loudness (see momentary_loudness and LOUDNESS_SLACK_DB), and amp_veltrack alone
    # makes velocity the loudness, so soft to loud is continuous across the layers and even
    # across the keys.
    normalize: str | None = None
    # The Library group set.json names (Dawsome's Kind: Drums, Bass, Keys, Pads, Strings,
    # Mallets, Synth, Other). The first sets name none and the DAW groups them by folder name;
    # sets added later say it here so the DAW needs no new entry for them.
    category: str | None = None
    # Write README.md beside set.json: the source, maker and license in plain words for a person
    # browsing the folder (asked for with the guitar sets, 2026-10-25).
    readme: bool = False


def extracted(downloads, source):
    return downloads / source / "extracted"


def require_ffmpeg():
    path = shutil.which("ffmpeg")
    if path is None:
        raise Problem("ffmpeg is not on PATH; build.py needs it to convert samples")
    if "--enable-libsoxr" not in run(path, "-hide_banner", "-buildconf").decode():
        raise Problem("this ffmpeg is built without libsoxr, which build.py's peak"
                      " measurement uses (see PLAYBACK)")
    return path


def run(*args, data=None):
    """The standard output of `args`; Problem with its error output when it fails."""
    done = subprocess.run(args, input=data, capture_output=True)
    if done.returncode != 0:
        raise Problem(f"{args[0]} failed on {args[-1]}:\n{done.stderr.decode(errors='replace')}")
    return done.stdout


# ---------------------------------------------------------------------------------------------
# Samples


def audible_end(samples, channels, floor_db):
    """The frame after the last one louder than `floor_db` under the peak (0 when silent)."""
    peak = max(max(samples, default=0.0), -min(samples, default=0.0))
    if peak == 0:
        return 0
    threshold = peak * 10 ** (floor_db / 20)
    block = 256 * channels
    end = len(samples)
    while end > 0:
        start = max(0, end - block)
        part = samples[start:end]
        if max(part) > threshold or -min(part) > threshold:
            last = next(i for i in range(len(part) - 1, -1, -1) if abs(part[i]) > threshold)
            return (start + last) // channels + 1
        end = start
    return 0


def crossfade_gains(steps, correlation):
    """(fading out, fading in) gains for each step of a crossfade between two stretches whose
    correlation is `correlation`. Linear ramps divided by the level their sum would have keep
    the power steady: for identical stretches (1) that is a plain linear fade, for unrelated
    ones (0) an equal-power fade; anti-correlated stretches are treated as unrelated."""
    correlation = min(max(correlation, 0.0), 1.0)
    gains = []
    for step in range(steps):
        fade_in = (step + 0.5) / steps
        fade_out = 1 - fade_in
        level = math.sqrt(fade_out ** 2 + fade_in ** 2 + 2 * fade_out * fade_in * correlation)
        gains.append((fade_out / level, fade_in / level))
    return gains


def correlation(first, second):
    """The normalized correlation of two equally long sample sequences (0 when one is silent)."""
    energy = math.sqrt(sum(map(operator.mul, first, first)) * sum(map(operator.mul, second,
                                                                       second)))
    return sum(map(operator.mul, first, second)) / energy if energy else 0.0


def steadiest_start(samples, channels, rate, loop):
    """The loop start, in frames on a 50 ms grid between the loop's earliest and latest
    start, whose span from the crossfade's lead-in to the loop's end has the smallest range
    of level (50 ms blocks, in dB)."""
    earliest, latest, length, crossfade = loop
    block = round(0.05 * rate)
    width = block * channels
    levels = []
    for start in range(0, len(samples) - width + 1, width):
        part = samples[start:start + width]
        levels.append(10 * math.log10(sum(map(operator.mul, part, part)) / width + 1e-20))
    span = round((crossfade + length) / 0.05)
    lead = round(crossfade / 0.05)
    # The lead-in must start at or after the sample's start, and the loop end by its end.
    first = max(round(earliest / 0.05), lead)
    last = min(round(latest / 0.05), len(levels) - span + lead)
    if last < first:
        raise Problem(f"a loop of {loop} does not fit {len(samples) / channels / rate:.2f} s")
    best = min(range(first, last + 1), key=lambda index: (
        max(levels[index - lead:index - lead + span]) - min(levels[index - lead:index - lead + span]),
        index))
    return best * block


def loop_end_repeats_start(samples, channels, start, end):
    """Whether the frame at a loop's `end`, read as SFZ reads it (the loop's last frame), is the
    loop's first frame again, so a voice plays that frame twice at every wrap: the frames the file
    holds from `end` on match the loop's first frames more closely than those from `end + 1` on do
    (up to 8 frames, as many as follow `end`; when none does, the frame itself equals the first).
    A loop written as FreePats writes it (see from_source_sfz) does; one whose file runs on past
    its last frame with its start, as build.py's own do, does not."""
    frames = len(samples) // channels
    count = min(8, frames - end - 1)
    if count < 1:
        return samples[end * channels:(end + 1) * channels] == samples[start * channels:
                                                                       (start + 1) * channels]

    def distance(at):
        return sum(abs(samples[at * channels + index] - samples[start * channels + index])
                   for index in range(count * channels))
    return distance(end) < distance(end + 1)


def first_loud_frame(samples, channels, level_db):
    """The first frame in which any channel is louder than `level_db` under the peak."""
    peak = max(max(samples, default=0.0), -min(samples, default=0.0))
    threshold = peak * 10 ** (level_db / 20)
    index = next((i for i, value in enumerate(samples) if abs(value) > threshold), 0)
    return index // channels


def trimmed_start(samples, channels, rate, shape):
    """The samples from `shape.start_seconds` before the note's first loud frame on, faded in
    over that lead, or from `shape.silence`'s lead before its first frame of sound when that is
    later; the samples unchanged when the note starts sooner."""
    lead = round(shape.start_seconds * rate)
    fade = lead if shape.start_fade_seconds is None else round(shape.start_fade_seconds * rate)
    start = first_loud_frame(samples, channels, shape.start_db) - lead
    if shape.silence is not None:
        # Searched from the cut on: a blip before it says nothing of what the lead starts on.
        level_db, seconds = shape.silence
        quiet = round(seconds * rate)
        cut = max(start, 0)
        sound = cut + first_loud_frame(samples[cut * channels:], channels, level_db) - quiet
        if sound > start:
            start, fade = sound, quiet
    if start <= 0:
        return samples
    out = samples[start * channels:]
    for step in range(fade):
        gain = step / fade
        for channel in range(channels):
            out[step * channels + channel] *= gain
    return out


def from_silence(samples, channels, rate, onset):
    """The samples faded in over their first `fade` seconds, or up to their first frame within
    `level_db` of the peak when that comes sooner, after `lead` seconds of silence (Shape.onset)."""
    lead, fade_seconds, level_db = onset
    fade = min(round(fade_seconds * rate), first_loud_frame(samples, channels, level_db))
    if fade == 0 and any(samples[:channels]):
        raise Problem(f"the sample starts within {-level_db:g} dB of its peak: a fade-in would"
                      " fade the note itself")
    out = array("f", samples)
    for step in range(fade):
        gain = step / fade
        for channel in range(channels):
            out[step * channels + channel] *= gain
    return array("f", bytes(4 * channels * round(lead * rate))) + out


def shaped(samples, channels, rate, shape):
    """(samples, loop) after `shape`: the new interleaved samples, and (loop_start, loop_end)
    in frames with loop_end inclusive, as SFZ counts it, or None."""
    if shape.loop and shape.tone:
        raise Problem("a tone's filters would not carry over the loop's seam; give one or the"
                      " other")
    if shape.hiss_fade and shape.floor_db is None:
        raise Problem("a hiss fade fades a tail into its cut; give it with a floor_db")
    if shape.silence is not None and shape.start_db is None:
        raise Problem("a silence lead trims a start cut's lead; give it with a start_db")
    if shape.onset is not None:
        if shape.loop or shape.start_db is not None:
            raise Problem("an onset starts the sample from silence; give it without a loop or a"
                          " start_db")
        samples = from_silence(samples, channels, rate, shape.onset)
    if shape.start_db is not None:
        if shape.loop:
            raise Problem("a start cut would move the loop; give a loop or a start_db, not both")
        samples = trimmed_start(samples, channels, rate, shape)
    frames = len(samples) // channels
    if shape.loop:
        start = steadiest_start(samples, channels, rate, shape.loop)
        length, crossfade = (round(seconds * rate) for seconds in shape.loop[2:])
        end = start + length
        if not 0 < crossfade <= min(start, length) or end > frames:
            raise Problem(f"a loop of {shape.loop} does not fit {frames / rate:.2f} s")
        out = samples[:end * channels]
        tail = samples[(end - crossfade) * channels:end * channels]
        lead_in = samples[(start - crossfade) * channels:start * channels]
        for step, (keep, take) in enumerate(crossfade_gains(crossfade,
                                                            correlation(tail, lead_in))):
            frame = (end - crossfade + step) * channels
            donor = (start - crossfade + step) * channels
            for channel in range(channels):
                out[frame + channel] = (samples[frame + channel] * keep
                                        + samples[donor + channel] * take)
        # Past the loop's end, what playback reads next: the loop's own start (LOOP_TAIL_SECONDS;
        # a loop on steadiest_start's 50 ms grid is at least 12.5 ms long, so this stays in it).
        out += out[start * channels:(start + round(LOOP_TAIL_SECONDS * rate)) * channels]
        return out, (start, end - 1)
    if shape.floor_db is None and shape.max_seconds is None:
        return samples, None
    fade = round(shape.fade_seconds * rate)
    cut = frames
    if shape.floor_db is not None:
        cut = min(cut, audible_end(samples, channels, shape.floor_db) + fade)
    if shape.max_seconds is not None:
        cut = min(cut, round(shape.max_seconds * rate))
    out = samples[:cut * channels]
    if shape.hiss_fade and (found := hiss_fade_start(out, channels, rate)) is not None:
        start, hiss_db = found
        depth = shape.floor_db - hiss_db
        if depth < 0:
            for frame in range(start, cut):
                gain = 10 ** (depth * (frame - start) / (cut - start) / 20)
                for channel in range(channels):
                    out[frame * channels + channel] *= gain
    fade = min(fade, cut)
    for step in range(fade):
        gain = (fade - 1 - step) / fade
        frame = (cut - fade + step) * channels
        for channel in range(channels):
            out[frame + channel] *= gain
    return out, None


# hiss_fade_start: compare.py's tail floor, so that what the build fades is what the comparison
# reports as the tail's floor (its tail_measures), and the margin that says the decay has reached it.
HISS_WINDOW_SECONDS, HISS_HOP_SECONDS, HISS_TAIL_DB, HISS_MARGIN_DB = 0.05, 0.025, -40, 3


def hiss_fade_start(samples, channels, rate):
    """(frame, level in dB under the peak) of a sample's hiss: its level is the 10th percentile
    of the 50 ms RMS windows (25 ms apart, of the channels' mean) after the sound first falls 40
    dB under its loudest window, the last 100 ms left out, against the mean's peak; the frame is
    the start of the first of those windows within HISS_MARGIN_DB of it, where the decay has sunk
    into the hiss. None when the sound never falls 40 dB there."""
    frames = len(samples) // channels
    mono = [sum(samples[frame * channels:(frame + 1) * channels]) / channels
            for frame in range(frames)]
    peak = max(map(abs, mono), default=0.0)
    window, hop = round(HISS_WINDOW_SECONDS * rate), round(HISS_HOP_SECONDS * rate)
    squares = [0.0]
    for value in mono:
        squares.append(squares[-1] + value * value)
    starts = range(0, max(1, frames - window), hop)
    levels = [math.sqrt(max(squares[min(start + window, frames)] - squares[start], 0.0) / window)
              for start in starts]
    if peak == 0 or not levels:
        return None
    quiet = max(levels) * 10 ** (HISS_TAIL_DB / 20)
    first = next((index for index, level in enumerate(levels) if level < quiet), None)
    if first is None:
        return None
    usable = [index for index in range(first, len(levels))
              if starts[index] + window <= frames - round(0.1 * rate) and levels[index] > 1e-7]
    if not usable:
        return None
    ranked = sorted(levels[index] for index in usable)
    position = 0.1 * (len(ranked) - 1)
    low = int(position)
    floor = ranked[low] + (ranked[min(low + 1, len(ranked) - 1)] - ranked[low]) * (position - low)
    near = floor * 10 ** (HISS_MARGIN_DB / 20)
    start = next(starts[index] for index in usable if levels[index] < near)
    return start, 20 * math.log10(floor / peak)


def decode(ffmpeg, path, chain=None):
    """The file's samples as interleaved 32-bit floats, through the ffmpeg filter `chain` when
    one is given."""
    samples = array("f")
    samples.frombytes(run(ffmpeg, "-v", "error", "-nostdin", "-i", str(path), "-map", "0:a:0",
                          *(("-af", chain) if chain else ()), "-f", "f32le", "-c:a", "pcm_f32le",
                          "-"))
    if sys.byteorder == "big":
        samples.byteswap()
    return samples


def filtered(ffmpeg, samples, channels, rate, chain):
    """Interleaved 32-bit float `samples` through the ffmpeg filter `chain`, as 32-bit floats."""
    if sys.byteorder == "big":
        samples = array("f", samples)
        samples.byteswap()
    out = array("f")
    out.frombytes(run(ffmpeg, "-v", "error", "-nostdin", "-f", "f32le", "-ar", str(rate), "-ac",
                      str(channels), "-i", "pipe:0", "-af", chain, "-f", "f32le", "-c:a",
                      "pcm_f32le", "-", data=samples.tobytes()))
    if sys.byteorder == "big":
        out.byteswap()
    return out


def retuning(rate, cents):
    """The ffmpeg chain that raises a sound at `rate` by `cents` (lowers it when negative) and
    keeps its rate: relabeled faster, then resampled back with soxr. The relabeled rate is a
    whole number, so the shift is within 0.02 cents of the one asked for. None for 0."""
    if cents == 0:
        return None
    return (f"asetrate={round(rate * 2 ** (cents / 1200))},"
            f"aresample={rate}:resampler=soxr:precision=28:osf=flt")


def momentary_loudness(ffmpeg, path):
    """The loudness of the file's loudest 400 ms, in LUFS: EBU R128 momentary loudness (ffmpeg's
    ebur128, K-weighted, so a bright attack counts as louder than a dark one of the same level),
    the highest of its 100 ms steps. Half a second of silence is added so a short sound's last
    windows are measured too."""
    out = run(ffmpeg, "-v", "error", "-nostdin", "-i", str(path), "-af",
              "apad=pad_dur=0.5,ebur128=metadata=1,"
              "ametadata=mode=print:key=lavfi.r128.M:file=-", "-f", "null", "-")
    values = [float(value) for value in re.findall(rb"lavfi\.r128\.M=(\S+)", out)]
    if not values or max(values) < -70:
        raise Problem(f"{path}: no loudness measured (silent?)")
    return max(values)


def playback_peak_db(ffmpeg, path):
    """The highest level the sample reaches when played back, in dBFS: the worst of PLAYBACK,
    computed in floating point so a peak above full scale shows."""
    rate = kitlib.audio_info(path).rate
    peak = 0.0
    for chain in PLAYBACK:
        samples = array("f")
        samples.frombytes(run(ffmpeg, "-v", "error", "-nostdin", "-i", str(path), "-af",
                              "aformat=sample_fmts=flt," + chain.format(eight=8 * rate),
                              "-f", "f32le", "-"))
        if sys.byteorder == "big":
            samples.byteswap()
        peak = max(peak, max(samples, default=0.0), -min(samples, default=0.0))
    return 20 * math.log10(peak) if peak else -math.inf


def strip_padding(path):
    """Removes the FLAC file's PADDING blocks (ffmpeg leaves 8 KiB in every file)."""
    data = path.read_bytes()
    position, kept = 4, []
    while True:
        header = data[position]
        length = int.from_bytes(data[position + 1:position + 4], "big")
        block = data[position:position + 4 + length]
        if header & 0x7F != FLAC_PADDING:
            kept.append(bytearray(block))
        position += 4 + length
        if header & 0x80:
            break
    for block in kept:
        block[0] &= 0x7F
    kept[-1][0] |= 0x80
    path.write_bytes(b"fLaC" + b"".join(kept) + data[position:])


def pan_gains(pan):
    """(left, right) gains of a mono sound placed at `pan` (-1 left to 1 right): Dawsome's law for
    a mono zone's pan (engine dsp/laws.rs channel_pan_gains), constant power and 1 in both sides
    at the center, so a centered mic plays at its gain as the library's own mix gives it."""
    side = lambda towards: math.sqrt(2) * math.sin((1 + towards) * math.pi / 4)  # noqa: E731
    return side(-pan), side(pan)


def mixed(ffmpeg, root, mics):
    """(samples, channels) of the files of `mics` (paths under `root`) mixed into one sound,
    interleaved 32-bit floats: each at its gain, a mono file placed at its pan (pan_gains), a
    stereo one as recorded. Mono when every file is mono and centered. The files must be one take:
    the same rate and length, as a multi-microphone recording's are."""
    infos = [kitlib.audio_info(root / mic.member) for mic in mics]
    if len({(info.rate, info.frames) for info in infos}) != 1:
        raise Problem(f"{mics[0].member}: its microphones' files differ in rate or length")
    if any(info.channels > 2 for info in infos):
        raise Problem(f"{mics[0].member}: a microphone's file has more than two channels")
    channels = 2 if any(info.channels == 2 or mic.pan for mic, info in zip(mics, infos)) else 1
    sides = [array("f", bytes(4 * infos[0].frames)) for _ in range(channels)]
    for mic, info in zip(mics, infos):
        samples = decode(ffmpeg, root / mic.member)
        if info.channels == 2:
            parts, gains = (samples[0::2], samples[1::2]), (mic.gain, mic.gain)
        elif channels == 2:
            parts, gains = (samples, samples), tuple(mic.gain * g for g in pan_gains(mic.pan))
        else:
            parts, gains = (samples,), (mic.gain,)
        for index, (part, gain) in enumerate(zip(parts, gains)):
            sides[index] = array("f", map(operator.add, sides[index], map(gain.__mul__, part)))
    if channels == 1:
        return sides[0], 1
    out = array("f", bytes(8 * infos[0].frames))
    out[0::2], out[1::2] = sides
    return out, 2


def make_sample(ffmpeg, source, target, zone, normalize, root=None):
    """Writes `target` from the source file `source` as the zone asks; returns (how it was
    made, the loop it got or None, its peak in dBFS before normalizing or None). `root`: the
    folder a zone's mics are found in."""
    info = kitlib.audio_info(source)
    channels = info.channels
    if zone.mics:
        if zone.mics[0].member != zone.member or zone.retune or zone.channel is not None:
            raise Problem(f"{source}: a mix starts with the zone's own file and is not retuned or"
                          " split")
    if zone.channel is not None:
        if not 0 <= zone.channel < info.channels:
            raise Problem(f"{source} has no channel {zone.channel} to keep")
        channels = 1
    # Keeping one channel copies its samples as they are, so a 16-bit one still rounds to itself.
    only_onset = (zone.shape is not None and zone.shape == Shape(onset=zone.shape.onset)
                  and not (zone.lowpass or normalize or zone.retune or zone.mics))
    filters = ((f"lowpass=f={zone.lowpass:g}:p=2," if zone.lowpass else "")
               + (ROUND if only_onset and info.bits <= 16 else DITHER))
    peak_db = None
    if zone.shape or normalize or zone.retune or zone.channel is not None or zone.mics:
        if zone.lowpass and normalize:
            raise Problem(f"{source}: normalizing after a baked low-pass is not supported")
        if zone.mics:
            samples, channels = mixed(ffmpeg, root, zone.mics)
        else:
            samples = decode(ffmpeg, source, retuning(info.rate, zone.retune))
        loop = None
        if zone.channel is not None:
            samples = samples[zone.channel::info.channels]
        if zone.shape:
            samples, loop = shaped(samples, channels, info.rate, zone.shape)
            if zone.shape.tone:
                samples = filtered(ffmpeg, samples, channels, info.rate, zone.shape.tone.chain)
        if normalize:
            peak = max(max(samples, default=0.0), -min(samples, default=0.0))
            if peak == 0:
                raise Problem(f"{source} is silent")
            peak_db = 20 * math.log10(peak)
            filters = f"volume={PEAK_DB - peak_db:.4f}dB:precision=float," + filters
        if sys.byteorder == "big":
            samples.byteswap()
        run(ffmpeg, "-v", "error", "-nostdin", "-f", "f32le", "-ar", str(info.rate), "-ac",
            str(channels), "-i", "pipe:0", "-af", filters, *FLAC_OUTPUT, str(target),
            data=samples.tobytes())
        made = "processed"
    elif source.suffix.lower() == ".flac" and info.bits <= 16 and not zone.lowpass:
        shutil.copyfile(source, target)
        return "copied", None, None
    else:
        loop = None
        run(ffmpeg, "-v", "error", "-nostdin", "-i", str(source), "-map", "0:a:0", "-af",
            filters, *FLAC_OUTPUT, str(target))
        made = "converted"
    strip_padding(target)
    out = kitlib.audio_info(target)
    if (out.rate, out.channels) != (info.rate, channels) or (
            not zone.shape and not zone.retune and out.frames != info.frames):
        raise Problem(f"{target}: ffmpeg changed the rate, channels or length of {source}")
    return made, loop, peak_db


# ---------------------------------------------------------------------------------------------
# Plans: what each set holds


def subset_region(region, where):
    """(sample, opcodes, low-pass Hz) of a source SFZ region, its opcodes trimmed to the
    subset. A static 2-pole low-pass is kept as a frequency to bake into the sample; opcodes at
    their SFZ default are dropped; any other opcode is an error, so a new source gets a
    decision instead of a silent change in sound."""
    opcodes = dict(region)
    sample = opcodes.pop("sample")
    lowpass = None
    if "fil_type" in opcodes or "cutoff" in opcodes:
        if opcodes.pop("fil_type", "lpf_2p") != "lpf_2p" or "cutoff" not in opcodes:
            raise Problem(f"{where}: only a fixed lpf_2p filter can be baked into a sample")
        lowpass = float(opcodes.pop("cutoff"))
    for name, default in NO_EFFECT.items():
        if name in opcodes and opcodes.pop(name) != default:
            raise Problem(f"{where}: {name} is not {default}, so it cannot be dropped")
    if "key" in opcodes:
        key = opcodes.pop("key")
        for name in ("lokey", "hikey", "pitch_keycenter"):
            opcodes.setdefault(name, key)
    for name in ("lokey", "hikey", "pitch_keycenter"):
        if name in opcodes:
            opcodes[name] = str(kitlib.note_number(opcodes[name]))
    unknown = sorted(set(opcodes) - kitlib.SFZ_OPCODES)
    if unknown:
        raise Problem(f"{where}: {', '.join(unknown)} outside the subset; decide how build.py"
                      " should convert them")
    return sample, opcodes, lowpass


# FreePats writes a loop's end as the frame after the loop, where SFZ and the DAW read loop_end as
# its last frame. Measured 2026-10-27 on all 75 loops of the piano, organ and pads: the 8 or 16
# frames each file holds from loop_end to its end are the loop's first frames again (the organ's and
# the sweep pad's bit for bit, the piano's within 2 and the choir pad's within 98 sixteen-bit steps,
# where one frame to the next moves 12 to 6346 there). Read as written, every wrap played the loop's
# first frame twice: compare.py heard the wraps at 9.7 to 26 times the click inside the loop. So
# those sets' loop_end is written less one (exclusive_loop_end), and build_set refuses any loop
# whose end is its start again (loop_end_repeats_start).
def from_source_sfz(downloads, name, title, source, notes=(), pick=None,
                    exclusive_loop_end=False, sfz=None):
    """A set made from a source's own .sfz: its regions, trimmed to the subset. `pick`, when
    given, sees each source region first and returns it (changed or not) to keep it, or None
    to leave it out. `exclusive_loop_end`: the source writes loop_end as the frame after the loop
    (FreePats does), so each is written here less one, the loop's last frame. `sfz`: the .sfz's
    path in the source (a repository's, at its root), else the one .sfz one folder down (an
    archive's)."""
    root = extracted(downloads, source)
    if sfz is None:
        found = sorted(root.glob("*/*.sfz"))
        if len(found) != 1:
            raise Problem(f"{root}: expected one .sfz one folder down, found {len(found)}")
        sfz_path = found[0]
    else:
        sfz_path = root / sfz
        if not sfz_path.is_file():
            raise Problem(f"{sfz_path} is missing; run fetch.py {source}")
    sfz = kitlib.read_sfz(sfz_path)
    folder = sfz_path.parent.relative_to(root).as_posix()
    zones, filtered, dropped, ended = [], 0, set(), 0
    for index, region in enumerate(sfz.regions):
        where = f"{sfz_path.name}, region {index + 1}"
        if pick is not None and (region := pick(region)) is None:
            continue
        dropped |= set(region) & set(NO_EFFECT)
        sample, opcodes, lowpass = subset_region(region, where)
        member = posixpath.normpath(posixpath.join(folder, sfz.control.get("default_path", ""),
                                                   sample))
        if member.startswith("../"):
            raise Problem(f"{where}: the sample {sample!r} lies outside the source")
        if exclusive_loop_end and "loop_end" in opcodes:
            opcodes["loop_end"] = str(int(opcodes["loop_end"]) - 1)
            ended += 1
        zones.append(Zone(member, opcodes, lowpass))
        filtered += lowpass is not None
    notes = list(notes)
    if ended:
        notes.append(f"loop_end is the source's less one in all {ended} loops: the source writes"
                     " the frame after the loop, and its files hold the loop's first frames again"
                     " from there (measured), where SFZ and the DAW name the loop's last frame;"
                     " read as written, every wrap played the loop's first frame twice")
    if filtered:
        notes.append(f"{filtered} regions had a fixed low-pass (fil_type=lpf_2p, cutoff); it is"
                     " applied to their samples (ffmpeg, 2-pole) since the DAW reads no filter"
                     " opcodes")
    if dropped:
        notes.append(f"dropped {', '.join(sorted(dropped))} at its SFZ default (no effect)")
    return Plan(name, title, source, zones, notes, derived_from=sfz_path.relative_to(root)
                .as_posix())


def spread(roots, lowest, highest):
    """(lokey, hikey) for each root key so that each key plays the nearest root, the lowest
    root reaching down to `lowest` and the highest up to `highest`."""
    edges = [lowest] + [(a + b) // 2 + 1 for a, b in zip(roots, roots[1:])] + [highest + 1]
    return [(low, high - 1) for low, high in zip(edges, edges[1:])]


def velocity_splits(count):
    """(lovel, hivel) of `count` equal velocity layers over 1..127 (for 4: 1-31, 32-63,
    64-95, 96-127, the split Virtuosity Drums uses)."""
    edges = [127 * index // count for index in range(count + 1)]
    return [(low + 1, high) for low, high in zip(edges, edges[1:])]


def versilian_key(name):
    """MIDI key of a Versilian note name such as F#3 (C3 = 60)."""
    return kitlib.note_number(name) + VERSILIAN_OCTAVE


def pitched_zones(pattern, notes, layers, lowest, highest, shape, extra):
    """Zones for samples named by note and layer: each note spread over the keys nearest it,
    each layer over an equal share of the velocities."""
    roots = [versilian_key(note) for note in notes]
    zones = []
    for (low, high), note, root in zip(spread(roots, lowest, highest), notes, roots):
        for layer, (lovel, hivel) in zip(layers, velocity_splits(len(layers))):
            opcodes = {"lokey": str(low), "hikey": str(high), "pitch_keycenter": str(root),
                       "lovel": str(lovel), "hivel": str(hivel), **extra}
            zones.append(Zone(pattern.format(note=note, layer=layer), opcodes, shape=shape))
    return zones


STRINGS_LOOP = Shape(loop=(1.5, 4.0, 2.0, 1.0))


def strings_plan(downloads):
    # Section sustains: cellos up to F#2, violins from G2 (where the violin samples start).
    cellos = pitched_zones(
        "Strings/Cello Section/susvib/susvib_{note}_v{layer}_1.wav",
        ("C1", "E1", "G1", "B1", "D2", "F2"), (1, 3), 31, 54, STRINGS_LOOP, {})
    violins = pitched_zones(
        "Strings/Violin Section/susVib/VlnEns_susVib_{note}_v{layer}.wav",
        ("G2", "A2", "B2", "D3", "F#3", "A3", "C4", "E4", "G4", "B4", "D5"), (1, 2), 55, 93,
        STRINGS_LOOP, {})
    zones = cellos + violins
    for zone in zones:
        zone.opcodes.update(loop_mode="loop_continuous", ampeg_release="0.8")
    return Plan("strings", "Strings", "vsco2-ce", zones, [
        "cello section sustains (vibrato) for keys 31-54, violin section sustains for 55-93;"
        " two velocity layers each (soft 1-63, loud 64-127)",
        f"held notes loop: {STRINGS_LOOP.describe()}",
        "VSCO names notes with C3 = MIDI 60; keys here are MIDI numbers",
    ], normalize="layer")


# VCSL's marimba samples whose tail is the recording's hiss (VCSL names, C3 = 60): measured
# 2026-10-27 with compare.py's tail measure on the shipped files, their tail floor's spectral
# flatness reads 0.44 to 0.83 (broadband, hiss) where 19 of the other 20 read 0.00 to 0.03 (the
# note's own decay, cut by VCSL while it still rings; C6 soft ends before falling 40 dB). Their
# hiss sits 46 to 59 dB under the peak, over the cut's -60, so each tail stopped at the hiss's
# level with the cut's 50 ms fade (the C2 loud sample, which C3 plays: 0.8 s of hiss at -57 dB,
# then the stop). C6 loud's hiss is at -66 dB, under the cut already: the fade leaves it alone.
# With the fade, each sample's -60 dB point (its first 50 ms window under -60 dB of the peak, or its
# end) comes at most 2.6 % earlier (C2 loud, F1 loud, F5 soft); its length is unchanged.
MARIMBA_HISS = {("F1", "soft"), ("F1", "med"), ("F1", "loud"), ("C2", "soft"), ("C2", "med"),
                ("C2", "loud"), ("F5", "soft"), ("F5", "med"), ("C6", "med"), ("C6", "loud")}


def mallets_plan(downloads):
    shape = Shape(floor_db=-60)
    hiss = Shape(floor_db=-60, hiss_fade=True)
    member = "Idiophones/Struck Idiophones/Marimba/Marimba_hit_Outrigger_{note}_{layer}_01.wav"
    zones = pitched_zones(
        member, ("F1", "C2", "G2", "B2", "F3", "C4", "G4", "B4", "F5", "C6"),
        ("soft", "med", "loud"), 36, 100, shape, {"loop_mode": "one_shot"})
    for zone in zones:
        if any(zone.member == member.format(note=note, layer=layer)
               for note, layer in MARIMBA_HISS):
            zone.shape = hiss
    return Plan("mallets", "Marimba", "vcsl", zones, [
        "VCSL's marimba, three velocity layers (soft, medium, loud); notes play to their end",
        f"each sample {shape.describe()}",
        f"the {len(MARIMBA_HISS)} samples whose tail is the recording's hiss (F1 and C2 in VCSL's"
        " names, every layer; F5 soft and medium; C6 medium and loud; measured), besides:"
        f" {hiss.describe().split(', ', 1)[1]} (none where the hiss is under that already)",
        "VCSL names notes with C3 = MIDI 60; keys here are MIDI numbers",
    ], normalize="layer")


# The electric guitar: FreePats' own two velocity layers, the first of four round robins (one keeps
# the set near the size of the strings). No guitar set loops, as no source does. The acoustic
# guitar is made, not sampled (see guitar_acoustic_plan); the bass guitar is Karoryfer's
# Fashionbass (see bass_guitar_plan).
#
# Karoryfer's guitar files are named by the pitch they sound, C4 = 60 (measured 2026-10-25:
# Shinyguitar's e2 at 82.4 Hz, MIDI 40; every one of the 68 used within 15 cents), which is how
# each of the acoustic guitar's notes was matched to its recording.
#
# The electric guitar as chosen by ear on 2026-10-26 (from the guitar-tone candidates: the
# electric's B, with Black And Blue Basses' C, the bass guitar until 2026-10-27), after hearing the
# first build and asking for them to sound "a bit more guitar like". Measured then, and fixed here:
# out of tune, layers stepping unevenly in loudness, the pluck's rise faded away, notes cut short.
#
# The pluck kept whole: 15 ms before the first moment within 20 dB of the peak covers the rise
# from 30 dB under it in 95 % of the electric's and that bass's samples (the first build's 5 ms,
# faded in over all of it, squeezed that rise from a median 4.0 ms to 0.7 ms), and only the
# first 2 ms fade in. Notes start 10 ms later than with 5 ms.
WHOLE_PLUCK = {"start_db": -20, "start_seconds": 0.015, "start_fade_seconds": 0.002}
# Measured tuning, 2026-10-26: cents each sample sounds off its pitch_keycenter (A4 = 440 Hz),
# from harmonics 1-3 in a 2 s Hann-windowed FFT starting 0.3 s in (zero-padded 8 times,
# parabolic interpolation), the level-weighted mean of h1 and h2/2; the 0.05-0.6 s window agrees
# within 3 cents except a hard pluck's first moments, which run sharp as a string does. Values
# are the correction (Zone.retune), for every sample off by 3 cents or more: the electric's A2
# soft layer was 18.7 cents flat against its normal layer's 2.5 sharp. After: every sample within
# 3 cents.
FSBS_CENTS = {
    "C2_s1_01": +4, "C2_s1_soft_01": +4, "E2_s1_soft_01": +7, "A2_s2_soft_01": +19,
    "C3_s2_01": +5, "D3_s3_soft_01": +5, "G3_s4_01": +9, "B3_s5_soft_01": -4, "C#4_s5_01": +4,
    "D5_s6_01": -3,
}
# Velocity: Karoryfer's own mappings raise each layer's gain linearly to full at its top velocity
# (amp_velcurve_40=1 and so on), which the subset cannot say; amp_veltrack=80 over equally loud
# layers (normalize="loudness") is the closest the subset comes. Measured for Fashionbass
# (2026-10-27, each layer's median loudest 400 ms with the source's own curve on it, against
# Dawsome's 1 - t(1 - (v/127)^2)): within 1.4 dB rms over velocities 20-127 (4.4 at most), the
# best of t from 60 to 100 in tens (60: 2.4, 70: 1.9, 90: 1.6, 100: 4.3 dB rms). The first builds
# played each layer at one level: steps between layers measured up to 7.6 dB on the Black And
# Blue bass, its notes 8.3 dB apart across the keys; the electric's normal layer played 3 dB
# under its soft one.
KARORYFER_VELTRACK = "80"
# Release: the DAW's release falls linearly in decibels to -90 dB over ampeg_release, so a bass
# source's 0.25 s is 40 dB down 0.11 s after a note ends (a walking line's 50 ms gaps fell to
# -17.7 dB; -12.8 with 0.4, measured on Black And Blue); Fashionbass's clean mapping says 0.15.
# The electric's 0.3 (the source's 1.1) keeps stabs crisp.
BASS_RELEASE, FSBS_RELEASE = "0.4", "0.3"
# Dawsome resamples a sample to the project's rate when it loads it (44.1 kHz to 48 kHz here) with a
# windowed-sinc filter that reads about 2 ms of the source either side of each point (measured
# 2026-10-26 through the installed CLI: an impulse moves the output from 2.1 ms before it to 2.1 ms
# after, at any transposition), and plays a note from its first frame.
#
# ONSET is for sources cut on the note itself, with no lead of their own to fade (the tester,
# 2026-10-26, every key struck alone from silence through the installed CLI). The FreePats electric
# bass starts mid-wave on 12 of its 20 keys, a step from silence of up to 0.049 (-26 dBFS): E and
# F# start 25 to 28 dB under their peak and reach it within 4 ms, and 6 of its 12 files have no
# zero crossing before the pluck to start on. Sonic Pi's hats zan and zild start mid-wave too
# (0.051, 0.029), and its kick ada and hat metal jump within a frame of their start, which the
# resampling's reach turned into a first frame of 0.014 and 0.018. So: the electric guitar's 2 ms
# fade-in, but stopping where the note comes within 20 dB of its peak, so that it touches only what
# comes before the pluck or hit (2 ms of 8 of the bass's 12 files, 6 frames of its A and D, 1 to
# 88 frames of the drums), and 2 ms of silence before it, the resampling's reach; notes start 2 ms
# later.
ONSET = Shape(onset=(0.002, 0.002, -20))
# The electric's 15 ms lead began on near-silence in 3 of its 30 samples, soft ones: F2's and G3's
# first moment within 60 dB of the peak came 2.1 and 2.5 ms after the cut (E3's 0.02 ms), a note
# started that much late for nothing heard (measured 2026-10-27). So the lead keeps at most 1 ms
# before that moment. The other leads start on sound, the pick on the string (-22 to -46 dB rms,
# tonal at the note's harmonics, 5 to 21 ms before the pluck's -10 dB point), which stays: these
# were chosen by ear with it (2026-10-26); its first 2 ms fade in, so a file can read under -60 dB
# for up to 1.6 ms there with sound under the fade.
FSBS_SHAPE = Shape(floor_db=-60, max_seconds=5.0, fade_seconds=2.0, **WHOLE_PLUCK,
                   silence=(-60, 0.001))


# The acoustic guitar, chosen by ear on 2026-10-26 from three ways of fusing Karoryfer
# Shinyguitar's recorded archtop with a modelled steel string: the archtop's own pluck drives the
# modelled string and body. Its samples were made outside this repository and are committed as
# made, with the set's .sfz, set.json and README.md: build.py does not build the set, it keeps
# the folder and its MANIFEST.json entries as they are. Each sample's entry names the recordings
# it was made from (made_from), files sources.json pins for the download named here, which
# check.py holds it to.
COMMITTED = {"guitar-acoustic": "karoryfer-shinyguitar"}


# The bass guitar as chosen by ear on 2026-10-27, from the kit-reference listening set,
# in place of Black And Blue Basses' darkblack: Karoryfer's Fashionbass, a Killer KB-Simmony played
# with the fingers, neck pickup, recorded direct, tuned in fifths (CGDA) with its low C string also
# sampled tuned down to A and F#, so it reaches F#0, under a five-string's B0. Its own clean
# mapping (fashionbass_clean.sfz, the one without release samples), every note of it: 19 notes
# from F#0 to A4 over keys 12-72, five velocity layers, three round robins. Its files are named by
# the pitch they sound, as its mapping places them (measured: every sample that holds one pitch
# is within 18 cents of its key; b1, the note C2 plays, at 61.7 Hz, B1).
FASHIONBASS_SFZ = "fashionbass_clean.sfz"
# Measured tuning, 2026-10-27, on the source: cents each sample sounds off its pitch_keycenter
# (A4 = 440 Hz) in four 1 s windows from 1, 2, 3 and 4 s in, each measured as FSBS_CENTS says.
# 40 samples glide, their windows each lower than the one before, by more than 6 cents in all
# (every F#0 and A0 sample, 10 of C1's 15): the low string, tuned down or at C1, starts sharp when
# plucked hard and falls into tune as it rings, as a slack string does (gb0 forte: +102 cents at
# 0.1 s, +14 at 2 s, +1 at 5 s); no fixed retune can follow that, so they stay as recorded. The other 245 hold one pitch
# (the median of their windows), and the 120 of them 3 cents or more off it are corrected by it
# (Zone.retune): F1 and Ab1 7 to 18 cents flat, D1's soft layers 6 to 9 sharp and its louder ones 7
# to 14 flat, D3 forte 10 flat, A4 6 or 7 flat, the rest 3 to 6 either way.
FASHIONBASS_CENTS = {
    "c1_p_rr1": -4, "c1_p_rr2": -3, "d1_pp_rr1": -9, "d1_pp_rr2": -7, "d1_pp_rr3": -6,
    "d1_p_rr1": -6, "d1_p_rr2": -8, "d1_p_rr3": -8, "d1_mf_rr3": +11, "d1_f_rr1": +12,
    "d1_f_rr2": +10, "d1_f_rr3": +11, "d1_ff_rr1": +14, "d1_ff_rr2": +11, "d1_ff_rr3": +7,
    "f1_pp_rr1": +14, "f1_pp_rr2": +17, "f1_pp_rr3": +18, "f1_p_rr1": +11, "f1_p_rr2": +11,
    "f1_p_rr3": +11, "f1_mf_rr1": +11, "f1_mf_rr2": +7, "f1_mf_rr3": +7, "f1_f_rr1": +11,
    "f1_f_rr2": +10, "f1_f_rr3": +9, "f1_ff_rr1": +15, "f1_ff_rr2": +15, "f1_ff_rr3": +9,
    "ab1_pp_rr1": +18, "ab1_pp_rr2": +15, "ab1_pp_rr3": +14, "ab1_p_rr1": +13, "ab1_p_rr2": +13,
    "ab1_p_rr3": +13, "ab1_mf_rr1": +12, "ab1_mf_rr2": +11, "ab1_mf_rr3": +11, "ab1_f_rr1": +12,
    "ab1_f_rr2": +11, "ab1_f_rr3": +11, "ab1_ff_rr1": +12, "ab1_ff_rr2": +11, "ab1_ff_rr3": +10,
    "d2_p_rr1": -3, "d2_p_rr2": -4, "d2_p_rr3": -5, "d2_mf_rr1": -4, "d2_mf_rr2": -4,
    "d2_mf_rr3": -4, "d2_f_rr2": -4, "d2_f_rr3": -5, "d2_ff_rr3": -4, "f2_f_rr1": -4,
    "f2_f_rr2": -4, "f2_f_rr3": -4, "d3_mf_rr1": +4, "d3_mf_rr2": +4, "d3_mf_rr3": +4,
    "d3_f_rr1": +5, "d3_f_rr2": +5, "d3_f_rr3": +5, "d3_ff_rr1": +9, "d3_ff_rr2": +10,
    "d3_ff_rr3": +10, "f3_mf_rr1": -4, "f3_mf_rr2": -4, "f3_mf_rr3": -4, "f3_f_rr1": -4,
    "f3_f_rr2": -3, "f3_f_rr3": -4, "f3_ff_rr1": -3, "f3_ff_rr2": -3, "f3_ff_rr3": -3,
    "ab3_pp_rr1": +4, "ab3_pp_rr2": +4, "ab3_pp_rr3": +4, "ab3_p_rr2": +4, "ab3_p_rr3": +4,
    "ab3_mf_rr1": +3, "ab3_mf_rr2": +4, "ab3_mf_rr3": +4, "ab3_f_rr1": +3, "ab3_f_rr3": +3,
    "ab3_ff_rr1": +4, "ab3_ff_rr2": +5, "ab3_ff_rr3": +5, "b3_p_rr2": +3, "b3_p_rr3": +4,
    "b3_mf_rr1": +4, "b3_mf_rr2": +4, "b3_mf_rr3": +4, "b3_f_rr1": +4, "b3_f_rr2": +4,
    "b3_f_rr3": +5, "b3_ff_rr1": +4, "b3_ff_rr2": +4, "b3_ff_rr3": +4, "f4_pp_rr1": +4,
    "f4_ff_rr1": +4, "f4_ff_rr2": +5, "ab4_f_rr2": +3, "ab4_ff_rr1": +3, "ab4_ff_rr3": +3,
    "a4_pp_rr1": +7, "a4_pp_rr2": +7, "a4_pp_rr3": +6, "a4_p_rr1": +7, "a4_p_rr2": +7,
    "a4_p_rr3": +7, "a4_mf_rr1": +7, "a4_mf_rr2": +7, "a4_mf_rr3": +6, "a4_f_rr1": +7,
    "a4_f_rr2": +6, "a4_f_rr3": +6, "a4_ff_rr1": +6, "a4_ff_rr2": +6, "a4_ff_rr3": +7,
}


def fashionbass_region(region):
    """A Fashionbass note region in the subset, or None for its noises (noises/, keys 84-96): its
    random round robin (lorand/hirand in thirds) as the turn it takes in three (seq_length=3), its
    layer's curve (amp_velcurve_N=1 at the layer's top) left to amp_veltrack, its Windows sample
    path written with slashes."""
    sample = region["sample"].replace("\\", "/")
    if sample.startswith("noises/"):
        return None
    turn = round(float(region.get("lorand", "0")) * 3) + 1
    kept = {name: value for name, value in region.items()
            if name not in ("lorand", "hirand") and not name.startswith("amp_velcurve_")}
    kept.update(sample=sample, seq_length="3", seq_position=str(turn),
                amp_veltrack=KARORYFER_VELTRACK, ampeg_release=BASS_RELEASE)
    return kept


def bass_guitar_plan(downloads):
    plan = from_source_sfz(downloads, "bass-guitar", "Bass Guitar", "karoryfer-fashionbass", [
        "Karoryfer's Fashionbass: a Killer KB-Simmony bass guitar played with the fingers, neck"
        " pickup, recorded direct with no effects, tuned in fifths (CGDA), its low C string also"
        " sampled tuned down to A and F#",
        "its own clean mapping (fashionbass_clean.sfz), every note: 19 notes from F#0 to A4 over"
        " keys 12-72, five velocity layers (pp 1-40, p 41-65, mf 66-85, f 86-110, ff 111-127),"
        " three round robins, which take turns (seq_length=3) where the source picks them at"
        " random; its noises (keys 84-96) and release samples are left out",
        f"amp_veltrack={KARORYFER_VELTRACK} over equally loud layers, the closest the subset comes"
        " to the source's own curves (each layer rising to full at its top velocity): within"
        f" 1.4 dB rms of them over velocities 20-127 (measured); ampeg_release={BASS_RELEASE} (the"
        " source's 0.15 cuts notes short in the DAW, whose release falls in decibels)",
        f"each sample {ONSET.describe()}; the samples' own 6.6 s length and fade-out are kept",
        "Karoryfer names Fashionbass's files by the pitch they sound (measured: b1 at 61.7 Hz),"
        " as its mapping places them",
    ], pick=fashionbass_region, sfz=FASHIONBASS_SFZ)
    for zone in plan.zones:
        zone.shape = ONSET
        zone.retune = FASHIONBASS_CENTS.get(stem(zone.member), 0)
    plan.notes.insert(3, f"{sum(1 for zone in plan.zones if zone.retune)} of the"
                      f" {len(plan.zones)} samples retuned by 3 to 18 cents to A4 = 440 Hz"
                      " (measured; those that hold one pitch), resampled with soxr; the 40 that"
                      " glide (every F#0 and A0 sample and 10 of C1's 15: the low string, slack,"
                      " starts sharp when plucked hard and falls into tune as it rings) are left as"
                      " recorded")
    plan.normalize, plan.category, plan.readme = "loudness", "Bass", True
    return plan


def bass_electric_plan(downloads):
    plan = from_source_sfz(downloads, "bass-electric", "Electric Bass", "freepats-finger-bass-yr", [
        "the finger-played variation of FreePats' Bass Guitar YR",
        f"each sample {ONSET.describe()}",
    ])
    for zone in plan.zones:
        zone.shape = ONSET
    return plan


# A drawbar organ plays every note at one level however hard its key is struck, so velocity leaves
# the organ's loudness alone. Its source names no amp_veltrack, so it played at SFZ's default 100,
# which swung a note 19.1 dB from velocity 40 to 120 (compare.py, 2026-10-27; Dawsome's gain is
# 1 - t(1 - (b/127)^2)). A channel's velocity weight in the DAW still adds a swing on top for
# whoever wants one.
ORGAN_VELTRACK = "0"
# Every velocity at the hardest's level made the organ the loudest set the DAW ships: C3 at velocity
# 100 played -9.2 LUFS with only the peak trim (-1.02 dB), where the 18 other pitched sets' median
# is -18.6 (compare.py, 2026-10-27, each held 4 s through dawsome render; -18.71 and -18.52 either
# side of it): a held organ chord swamped any other channel added at the same fader. This volume
# puts its C3 at that median (-18.6 LUFS); its peak is then 10.4 dB under the -1 dBFS limit.
ORGAN_VOLUME = "-10.40"


def organ_plan(downloads):
    plan = from_source_sfz(downloads, "organ", "Drawbar Organ", "freepats-drawbar-organ", [
        "the WAV samples are stored as FLAC, unchanged in sound",
        f"amp_veltrack={ORGAN_VELTRACK}: a drawbar organ is not velocity sensitive, so every"
        " velocity plays at the level of the hardest (the source's default 100 swung a note 19 dB"
        " from velocity 40 to 120); a channel's velocity weight in the DAW still adds a swing",
        f"volume={ORGAN_VOLUME}: at every velocity the organ's C3 plays as loud as the median of"
        " the other pitched sets at velocity 100 (-18.6 LUFS, measured with compare.py); at full"
        " level it was the loudest set by 9 dB",
    ], exclusive_loop_end=True)
    for zone in plan.zones:
        zone.opcodes["amp_veltrack"] = ORGAN_VELTRACK
        zone.opcodes["volume"] = ORGAN_VOLUME
    return plan


# The electric piano as chosen by ear on 2026-10-27, from the kit-reference listening
# set: FreePats' FM Piano 1, which replaces Dawsome's synthesised ep-c4 (removed on the Dawsome
# side). Converted as FreePats made it, nothing trimmed or levelled, measured on the source then:
# 35 of its 36 samples peak at -0.05 to -0.02 dBFS (C3 v100 at -2.37), so the layers, recorded at
# velocities 60, 80 and 100, are equally loud and the source's default amp_veltrack=100 makes
# velocity the loudness; every sample starts from silence (its first frame is 0, its first moment
# within 20 dB of its peak 0.2 to 0.8 ms in), and every one has fallen 60 dB under its peak 0.4 to
# 3.6 s before its end (its last 20 ms 100 to 130 dB down), so there is no lead or tail to cut.
def ep_plan(downloads):
    plan = from_source_sfz(downloads, "ep", "Electric Piano", "freepats-fm-piano-1", [
        "FreePats' FM Synthesized Piano #1: the Yamaha DX7's 'E. Piano 1' patch, the FM electric"
        " piano of many 1980s songs, recorded from the Hexter synthesizer, a model of the DX7",
        "FreePats' own mapping, unchanged: 12 notes a tritone apart from F#1 to C7, spread over"
        " keys 27-98, three velocity layers (1-69, 70-109, 110-127) recorded at velocities 60,"
        " 80 and 100, each sample at full scale; the default amp_veltrack=100 makes velocity"
        " the loudness, as in the source; ampeg_release=0.4, the source's",
        "the 24-bit samples are stored as 16-bit FLAC (triangular dither), otherwise unchanged:"
        " each starts from silence, and its tail falls into the dither, about 96 dB under its"
        " peak (the source's last 20 ms are 100 to 130 dB down)",
    ])
    plan.category, plan.readme = "Keys", True
    return plan


def first_round_robin(region):
    """A FreePats region that is the first of its random round robins (lorand 0 or none),
    without the lorand/hirand the subset lacks; None for the others."""
    if float(region.get("lorand", "0")) != 0:
        return None
    return {name: value for name, value in region.items() if name not in ("lorand", "hirand")}


def guitar_electric_plan(downloads):
    plan = from_source_sfz(downloads, "guitar-electric", "Electric Guitar",
                           "freepats-eguitar-fsbs-clean", [
        "FreePats' FSBS Electric Guitar Clean #1: a Fender electric guitar, bridge pickup,"
        " recorded direct and processed through an amplifier and effects rack into a clean"
        " (undistorted) stereo sound; Dawsome's Saturator insert drives it into an amp sound",
        "FreePats' own mapping: 19 notes over keys 35-86, two velocity layers (soft 1-92,"
        " normal 93-127) up to E4, one above; the first of each note's four random round"
        " robins (lorand/hirand are outside the subset)",
        "loudness follows velocity (the default amp_veltrack=100), as in the source, over"
        f" equally loud layers; ampeg_release={FSBS_RELEASE} for crisper stabs (the source's"
        " 1.1)",
        f"each sample (the sources ring for up to 39 s) {FSBS_SHAPE.describe()}",
    ], pick=first_round_robin)
    for zone in plan.zones:
        zone.shape = FSBS_SHAPE
        zone.retune = FSBS_CENTS.get(stem(zone.member), 0)
        zone.opcodes["ampeg_release"] = FSBS_RELEASE
    plan.notes.insert(3, f"{sum(1 for zone in plan.zones if zone.retune)} samples retuned by 3"
                      " to 19 cents to A4 = 440 Hz (measured; the A2 soft layer was 21 cents"
                      " under its normal layer), resampled with soxr")
    # "loudness": the sources all peak at full scale, so their soft layer played louder than
    # their normal one (a sharper pluck is quieter at the same peak); equally loud, velocity
    # alone sets the level.
    plan.normalize, plan.category, plan.readme = "loudness", "Guitar", True
    return plan


# The acoustic drums as chosen by ear on 2026-10-27, from the kit-reference listening
# set, in place of Virtuosity Drums: Karoryfer's Big Rusty Drums, a big vintage kit, played with
# sticks (its brushes, mallets, stirs and noises left out), on the keys the set had so that a
# pattern hits the same drums: 36 kick, 37 cross-stick, 38 snare, 40 rimshot, 41 low tom, 42
# closed hi-hat, 44 pedal hi-hat, 46 open hi-hat, 48 high tom, 49 crash, 51 ride, 53 ride bell.
# Big Rusty's own keymap has the same, but for its toms: 41 and 48 play the 18" and the 14", the
# low and the high tom of its basic kit (Programs/02-basic.sfz), which its keymap puts on 43 and 47.
#
# The mix is the library's own, at its basic program's defaults, baked into one sample per hit
# (Zone.mics; the subset has no controllers to mix with): each microphone at the share its fader
# gives it there (amplitude_ccN=share at set_ccN, fader), a close mic placed where its pan
# control's default puts it (pan_ccN=100 through the program's curve 9, pan_at) by Dawsome's own
# law for a mono zone (pan_gains); the overheads' pan curves are centered on those defaults, so the
# overheads keep their recorded image. At those defaults the kick plays its inside mic alone (its
# overhead and snare-mic faders, CC71 and CC72, default to 0, which silences them); the snare its
# bottom, top and overhead mics at 100, 40 and 60 of 127; a tom its close mic at 100 of 127, panned
# (the 18" to +0.38, the 14" to -0.68), and the overheads at 70; the hi-hat both mics at 70 % of
# 100/127, centered; the crash and the ride their close mic at 40 of 127 (panned -0.52 and +0.53) and
# overheads at 100, times 70 % (crash) or 60 % (ride and bell).
def fader(share, cc):
    """A microphone's gain at its fader's default: amplitude_ccN=`share` (%) at set_ccN=`cc`."""
    return share / 100 * cc / 127


def pan_at(cc):
    """A close mic's pan at its pan control's default `cc`, through Big Rusty's curve 9 (-1 at 0,
    0 at 63, 1 at 127, straight between)."""
    return (cc - 63) / 63 if cc <= 63 else (cc - 63) / 64


# Every hit starts from silence, as the electric drums' do (ONSET): Big Rusty's close mics start
# within a frame or two of the hit (their first moment within 20 dB of the peak comes 0.0 to 0.2 ms
# in), on a step. Tails as Virtuosity's had them: drums and the closed and pedal hi-hats cut 60 dB
# down, the open hi-hat, the crash and the ride capped (their recordings ring 10 to 18 s).
DRUM = Shape(floor_db=-60, onset=ONSET.onset)
OPEN_HAT = Shape(max_seconds=3.0, fade_seconds=1.0, onset=ONSET.onset)
CYMBAL = Shape(max_seconds=5.0, fade_seconds=2.0, onset=ONSET.onset)
RIDE = Shape(max_seconds=4.0, fade_seconds=1.5, onset=ONSET.onset)
SNARE_MICS = (("btm", fader(100, 100), 0.0), ("top", fader(100, 40), 0.0),
              ("oh", fader(100, 60), 0.0))
HAT_MICS = (("cl", fader(70, 100), 0.0), ("oh", fader(70, 100), 0.0))
RIDE_MICS = (("cl", fader(60, 40), pan_at(97)), ("oh", fader(60, 100), 0.0))
# (key, the hit's files in Samples/ with {mic} for the microphone's folder, its mics as (folder,
# gain, pan), velocity layers recorded, round robins kept, shape). Four layers of each, spread
# evenly up to the loudest, over equal velocity shares (each layer's own curve, amp_velcurve, is
# outside the subset); two of the four round robins (the ride's three) where a hit repeats fast,
# one for the open hi-hat, the crash and the bell, whose long tails cost the most memory.
BIG_RUSTY_KIT = (
    (36, "kick_24/kick/{mic}/k", (("kick", fader(100, 100), 0.0),), 14, 2, DRUM),
    (37, "snare_14/sidestick/{mic}/sn_ss", SNARE_MICS, 4, 2, DRUM),
    (38, "snare_14/center/{mic}/sn_center", SNARE_MICS, 10, 2, DRUM),
    (40, "snare_14/rimshot/{mic}/sn_rims", SNARE_MICS, 6, 2, DRUM),
    (41, "tom_18/center/{mic}/t18", (("cl", fader(100, 100), pan_at(87)),
                                     ("oh", fader(100, 70), 0.0)), 8, 2, DRUM),
    (42, "hihat_14/tc/{mic}/ht_tc", HAT_MICS, 8, 2, DRUM),
    (44, "hihat_14/chik/{mic}/ht_chik", HAT_MICS, 5, 2, DRUM),
    (46, "hihat_14/open/{mic}/ht_open", HAT_MICS, 6, 1, OPEN_HAT),
    (48, "tom_14/center/{mic}/t14", (("cl", fader(100, 100), pan_at(20)),
                                     ("oh", fader(100, 70), 0.0)), 6, 2, DRUM),
    (49, "crash_17/cr/{mic}/cr", (("cl", fader(70, 40), pan_at(30)),
                                  ("oh", fader(70, 100), 0.0)), 5, 1, CYMBAL),
    (51, "ride_22/rd/{mic}/rd", RIDE_MICS, 10, 2, RIDE),
    (53, "ride_22/bl/{mic}/rd_bl", RIDE_MICS, 4, 1, RIDE),
)
# A take cut from the recording a moment into its hit, left out for the next ones: (files, layer)
# -> the round robins used there. The sidestick's loudest layer's first take starts on its hit (its
# bottom mic's first frame at -28.6 dBFS, the mix's 16 dB under its peak), which no fade-in can
# start from silence without fading the hit; every other take starts 30 to 61 dB under its peak.
LATE_TAKES = {("snare_14/sidestick/{mic}/sn_ss", 4): (2, 3)}
DRUM_KEYS = ("36 kick, 37 cross-stick, 38 snare, 40 rimshot, 41 low tom (18\"), 42 closed hi-hat,"
             " 44 pedal hi-hat, 46 open hi-hat, 48 high tom (14\"), 49 crash, 51 ride, 53 ride bell")


def tail(shape):
    """What `shape` does to a hit's tail, in words (its onset said once for all)."""
    return Shape(floor_db=shape.floor_db, max_seconds=shape.max_seconds,
                 fade_seconds=shape.fade_seconds).describe()


def drums_acoustic_plan(downloads):
    zones = []
    for key, files, mics, recorded, robins, shape in BIG_RUSTY_KIT:
        kept = min(4, recorded)
        layers = [math.ceil((index + 1) * recorded / kept) for index in range(kept)]
        for layer, (lovel, hivel) in zip(layers, velocity_splits(kept)):
            takes = LATE_TAKES.get((files, layer), range(1, robins + 1))
            for position, robin in enumerate(takes, 1):
                opcodes = {"lokey": str(key), "hikey": str(key), "pitch_keycenter": str(key),
                           "lovel": str(lovel), "hivel": str(hivel)}
                if robins > 1:
                    opcodes.update(seq_length=str(robins), seq_position=str(position))
                opcodes.update(loop_mode="one_shot", amp_veltrack="0")
                take = tuple(Mic(f"Samples/{files.format(mic=folder)}_vl{layer}_rr{robin}.flac",
                                 gain, pan) for folder, gain, pan in mics)
                zones.append(Zone(take[0].member, opcodes, shape=shape, mics=take))
    return Plan("drums-acoustic", "Acoustic Drums", "karoryfer-big-rusty-drums", zones, [
        "Karoryfer's Big Rusty Drums played with sticks: its 24\" kick, 14\" snare, 18\" and 14\""
        " toms, 14\" hi-hat, 17\" crash and 22\" ride; keys as the set had them, the General MIDI"
        f" drum map's: {DRUM_KEYS}",
        "each hit is the library's own mix of its microphones at its basic program's defaults"
        " (Programs/02-basic.sfz), baked into one sample: the kick its inside mic alone; the snare"
        " its bottom, top and overhead mics at 100, 40 and 60 of 127; a tom its close mic at 100"
        " of 127 (the 18\" panned to +0.38, the 14\" to -0.68) and the overheads at 70; the"
        " hi-hat both at 70 % of 100/127; the crash its close mic at 70 % of 40/127 (panned"
        " -0.52) and overheads at 70 % of 100/127, the ride and bell the same at 60 % (close mic"
        " +0.53); a close mic is placed by Dawsome's pan law for a mono zone, the overheads keep"
        " their own image",
        "four of the recorded velocity layers, spread evenly up to the loudest, over equal"
        " velocity shares; two round robins (seq_length=2), the first two (the sidestick's"
        " loudest layer its second and third: its first was cut a moment into the hit), one for"
        " the open hi-hat, the crash and the bell",
        "amp_veltrack=0: loudness follows the recorded layers (the source's per-layer"
        " amp_velcurve, outside the subset, is dropped)",
        f"every hit {ONSET.describe()}; then drums and the closed and pedal hi-hats"
        f" {tail(DRUM)}; open hi-hat: {tail(OPEN_HAT)}; crash: {tail(CYMBAL)}; ride and bell:"
        f" {tail(RIDE)}",
        "the open hi-hat is not choked by the closed or pedal hi-hat (the source's group and"
        " off_by are outside the subset)",
    ], normalize="set")


# Sonic Pi samples left out, by name: each one's Freesound page was read on 2026-10-22, and
# these are unreachable or say the sound comes from someone else's recording, product or
# sample set, which a CC0 mark from the uploader cannot clear.
SONIC_PI_LEFT_OUT = {
    "loop_amen": "the Amen break (The Winstons, 'Amen, Brother'), says its page",
    "loop_amen_full": "the Amen break, says its page",
    "loop_compus": "made with an Ableton Live factory kit (LD Compuspank), says its page",
    "loop_safari": "'recorded in my keyboard': a keyboard's built-in content",
    "loop_breakbeat": "includes another user's sub bass, whose page is gone (404)",
    "bd_jazz": "'Sound from Figure app' (a commercial app's sound)",
    "hat_bdu": "from the fdiskc.com TR-606 sample set, says its page",
    "hat_noiz": "'That HiHat that was in that Tron song Daft Punk made'",
    "vinyl_backspin": "a scratch of an unnamed record on a turntable",
    "vinyl_scratch": "a scratch of an unnamed record",
    "misc_crow": "its Freesound page is gone (404)",
    "misc_cineboom": "its Freesound page is gone (404)",
    "elec_*": "one uploader's compilation whose file names point to other products (E-mu"
              " SP-12 sounds, another DAW's content, a 'virus kit'); all 25 left out",
}
SONIC_PI_SAMPLES = "etc/samples/"
SONIC_PI_README = "etc/samples/README.md"
# The first drum key: C1, with C4 = 60 (Dawsome's typing keyboard and SFZ agree on that).
FIRST_DRUM_KEY = 24
# Sonic Pi files whose two channels are one sound with one channel upside down, so that they cancel
# when a mono speaker or a mono bus sums them: name -> the channel kept, as a mono file (13 of the
# set's 19 hi-hats are mono files). hat_gnu, measured 2026-10-27: its right channel is its left
# inverted and 0.87 dB quieter (right = -0.905 x left, what is left over 61 dB under it), an L/R
# correlation of -1.00, the two summed 26 dB under its left; its left, the louder, is kept as
# recorded. No other file in the kit's sets correlates at -0.9 or under (the lowest, -0.73).
INVERTED_STEREO = {"hat_gnu": 0}


def sonic_pi_pages(downloads):
    """name (underscores removed) -> the Freesound page Sonic Pi's README gives for it."""
    text = (extracted(downloads, "sonic-pi") / SONIC_PI_README).read_text(encoding="utf-8")
    pages = {}
    for name, url in re.findall(r"\*\s+`:?(\w+)`\s*-\s*(\S+)", text):
        url = re.sub(r"^http://", "https://", url).replace("://www.freesound", "://freesound")
        pages[name.replace("_", "").lower()] = url
    return pages


def sonic_pi_zones(downloads, sources, prefixes):
    """Zones for the Sonic Pi files listed in sources.json whose names start with one of
    `prefixes`, in that prefix order, each with its Freesound page."""
    pages = sonic_pi_pages(downloads)
    zones = []
    for prefix in prefixes:
        for entry in sources["sonic-pi"]["files"]:
            name = stem(entry["path"])
            if not (entry["path"].startswith(SONIC_PI_SAMPLES) and name.startswith(prefix)):
                continue
            if name in SONIC_PI_LEFT_OUT or name.startswith("elec_"):
                raise Problem(f"sources.json lists {name}, which build.py leaves out")
            page = pages.get(name.replace("_", "").lower())
            if page is None:
                raise Problem(f"{name}: no Freesound page in {SONIC_PI_README}")
            zones.append(Zone(entry["path"], {}, upstream=page))
    return zones


def drums_electronic_plan(downloads, sources):
    zones = sonic_pi_zones(downloads, sources, ("bd_", "sn_", "hat_"))
    for key, zone in enumerate(zones, FIRST_DRUM_KEY):
        zone.opcodes.update(lokey=str(key), hikey=str(key), pitch_keycenter=str(key),
                            loop_mode="one_shot")
        zone.shape = ONSET
        zone.channel = INVERTED_STEREO.get(stem(zone.member))
    return Plan("drums-electronic", "Electronic Drums", "sonic-pi", zones, [
        f"one sound per key from key {FIRST_DRUM_KEY} (C1): kicks (bd_), then snares (sn_),"
        " then hi-hats (hat_), each group in name order; each plays at its own pitch to its end",
        f"samples are Sonic Pi's FLAC files, each {ONSET.describe()}; rounded back to 16 bits"
        " without dither, so every sample after the fade is as recorded",
        *(f"{name} is its {('left', 'right')[channel]} channel alone, a mono file: its other"
          " channel is the same sound upside down (L/R correlation -1.00), so the two canceled"
          " when summed to mono (26 dB down)" for name, channel in INVERTED_STEREO.items()),
        "left out after reading each sound's Freesound page: "
        + "; ".join(f"{name}: {why}" for name, why in SONIC_PI_LEFT_OUT.items()
                    if name.startswith(("bd_", "sn_", "hat_", "elec_"))),
    ])


def one_shots_plan(downloads, sources):
    zones = sonic_pi_zones(downloads, sources, ("ambi_", "glitch_", "misc_", "vinyl_", "loop_"))
    return Plan("one-shots", "One-shots and Loops", "sonic-pi", zones, [
        "plain sound files for the sampler, no .sfz: ambient sounds (ambi_), glitches"
        " (glitch_), misc_, vinyl_ and loops (loop_), Sonic Pi's FLAC files unchanged",
        "left out after reading each sound's Freesound page: "
        + "; ".join(f"{name}: {why}" for name, why in SONIC_PI_LEFT_OUT.items()
                    if name.startswith(("ambi_", "glitch_", "misc_", "vinyl_", "loop_"))),
    ], kind="samples")


def plans(downloads, sources):
    """Every set, in the order the README lists them."""
    return [
        from_source_sfz(downloads, "piano", "Upright Piano", "freepats-upright-piano-kw", [
            "FreePats' full Upright Piano KW: every key, both velocity layers (1-80, 81-127);"
            " the bass notes loop, the treble plays out"], exclusive_loop_end=True),
        bass_electric_plan(downloads),
        from_source_sfz(downloads, "bass-synth", "Synth Bass", "freepats-synth-bass-2",
                        ["FreePats' Synth Bass #2, a DX7 'BASS 1' style sound"]),
        bass_guitar_plan(downloads),
        organ_plan(downloads),
        ep_plan(downloads),
        from_source_sfz(downloads, "pad-choir", "Choir Pad", "freepats-synth-pad-choir",
                        exclusive_loop_end=True),
        from_source_sfz(downloads, "pad-sweep", "Sweep Pad", "freepats-sweep-pad",
                        exclusive_loop_end=True),
        strings_plan(downloads),
        guitar_electric_plan(downloads),
        mallets_plan(downloads),
        drums_acoustic_plan(downloads),
        drums_electronic_plan(downloads, sources),
        one_shots_plan(downloads, sources),
    ]


# Sources the DAW downloads itself (too big for the repository): what the manifest says.
DOWNLOAD_SETS = {
    "osiris-piano": {
        "set": "piano-grand",
        "name": "Grand Piano (Osiris)",
        "sfz": "Osiris Piano/Programs/01-natural.sfz",
        "notes": [
            "optional: the upstream release archive, unchanged; fetch, check the SHA-256,"
            " extract (it is a zip)",
            "not playable as intended with the DAW's SFZ subset: the program picks microphones,"
            " sustain-pedal and noise variants with CC conditions (locc/hicc), so under the"
            " subset every variant and microphone would sound at once; it needs a trimmed"
            " build (one variant, one microphone) or CC conditions in the DAW",
        ],
    },
}


# ---------------------------------------------------------------------------------------------
# Writing a set


def format_sfz(plan, source, zones):
    """The .sfz text: a <control> naming the samples folder, opcodes every region shares in
    <global>, then one <region> line each."""
    opcode_sets = [zone.opcodes for zone in zones]
    shared = {name: value for name, value in opcode_sets[0].items()
              if name not in PLACEMENT and all(ops.get(name) == value for ops in opcode_sets)}
    for zone in zones:
        extra = set(zone.opcodes) - set(REGION_ORDER)
        if extra:
            raise Problem(f"{plan.name}: build.py does not write {', '.join(sorted(extra))}")
    lines = [
        f"// {plan.title}: part of dawsome-kit, written by build.py.",
        f"// From {source['title']}; {kitlib.LICENCE}, see set.json.",
        "// Uses only the SFZ opcodes Dawsome reads; keys are MIDI numbers (C4 = 60).",
        "",
        "<control>",
        "default_path=samples/",
        "",
        "<global>",
        *(f"{name}={value}" for name, value in sorted(shared.items(),
                                                      key=lambda item: REGION_ORDER.index(item[0]))),
        "",
    ]
    for zone in zones:
        rest = [f"{name}={zone.opcodes[name]}" for name in REGION_ORDER
                if name in zone.opcodes and name not in shared]
        lines.append(" ".join([f"<region> sample={zone.sample}", *rest]))
    return "\n".join(lines) + "\n"


def format_readme(plan, source, description):
    """README.md for a set folder: what it is, where it comes from, its license and what
    build.py did, from the same facts set.json holds."""
    low, high = description["keys"]
    lines = [
        f"# {plan.title}",
        "",
        f"An instrument set of dawsome-kit, written by build.py: `{description['sfz']}` plays"
        f" the sounds in `samples/` on keys {low} to {high} (MIDI numbers, C4 = 60).",
        "",
        "## Source",
        "",
        f"- {source['title']}",
        f"- Made by: {source['maker']}",
        f"- From: {source_url(source)}",
        "",
        "## License",
        "",
        "CC0 1.0 Universal, a public domain dedication: use, change, sell and share these sounds,"
        " in songs or in software, with no credit due.",
        "",
        f"- License page: {source['licence_page']}",
        f"- It says: \"{source['licence_quote']}\"",
        *([f"- Also: {source['licence_file']}"] if source.get("licence_file") else []),
        *([f"- Note: {source['licence_note']}"] if source.get("licence_note") else []),
        "",
        "The changes made here (conversion, trimming, mapping) are dedicated CC0 too. `set.json`"
        " holds the same facts with the set's figures, and the kit's `MANIFEST.json` traces"
        " every file to its source and SHA-256.",
        "",
        "## What build.py did",
        "",
        *(f"- {note}" for note in plan.notes),
    ]
    return "\n".join(lines) + "\n"


def origin(sources, source_id, member):
    """(the URL a source file came from, its pinned SHA-256): the archive for an archive
    source, the file itself for a repository source."""
    source = sources[source_id]
    if "archive" in source:
        return source["archive"]["url"], source["archive"]["sha256"]
    for entry in source["files"]:
        if entry["path"] == member:
            return kitlib.raw_url(source["repository"], source["commit"], member), entry["sha256"]
    raise Problem(f"{member} is not among the files sources.json lists for {source_id}")


def source_url(source):
    if "archive" in source:
        return source["archive"]["url"]
    return f"https://github.com/{source['repository']}/tree/{source['commit']}"


def build_set(ffmpeg, downloads, sources, plan):
    """Builds sets/<plan.name> and returns its manifest entries."""
    source = sources[plan.source]
    root = extracted(downloads, plan.source)
    staging = kitlib.SETS / f".{plan.name}.partial"
    if staging.exists():
        shutil.rmtree(staging)
    (staging / "samples").mkdir(parents=True)
    made, infos, peaks, entries = {}, {}, {}, []
    for zone in plan.zones:
        if not NAME_CHARACTERS.fullmatch(zone.sample):
            raise Problem(f"{plan.name}: the sample name {zone.sample!r} needs renaming")
        key = (zone.member, zone.lowpass, zone.shape, zone.retune, zone.channel, zone.mics)
        if zone.sample in made:
            if made[zone.sample][0] != key:
                raise Problem(f"{plan.name}: two different samples would be {zone.sample}")
            loop = made[zone.sample][1]
        else:
            path = root / zone.member
            for member in [zone.member] + [mic.member for mic in zone.mics]:
                if not (root / member).is_file():
                    raise Problem(f"{root / member} is missing; add {member!r} to sources.json's"
                                  f" {plan.source} and run fetch.py --pin {plan.source}")
            target = staging / "samples" / zone.sample
            how, loop, peaks[zone.sample] = make_sample(ffmpeg, path, target, zone,
                                                        plan.normalize, root)
            made[zone.sample] = (key, loop)
            infos[zone.sample] = kitlib.audio_info(target)
            url, sha256 = origin(sources, plan.source, zone.member)
            traced = {"origin": url, "origin_sha256": sha256}
            if zone.mics:
                # Every microphone's file the sample is mixed from, with its share of the mix.
                traced["mixed_from"] = [
                    {"member": mic.member, "origin": mic_url, "origin_sha256": mic_sha256,
                     "gain_db": round(20 * math.log10(mic.gain), 2), "pan": round(mic.pan, 4)}
                    for mic in zone.mics
                    for mic_url, mic_sha256 in [origin(sources, plan.source, mic.member)]]
            entries.append({
                "path": f"samples/{zone.sample}", "source": plan.source, **traced,
                "member": zone.member, "made_by": how,
                "licence_page": source["licence_page"],
                **({"upstream": zone.upstream} if zone.upstream else {}),
            })
        if loop:
            zone.opcodes.update(loop_start=str(loop[0]), loop_end=str(loop[1]))
    for zone in plan.zones:
        if "loop_end" in zone.opcodes and loop_end_repeats_start(
                decode(ffmpeg, staging / "samples" / zone.sample), infos[zone.sample].channels,
                int(zone.opcodes.get("loop_start", 0)), int(zone.opcodes["loop_end"])):
            raise Problem(f"{plan.name}: {zone.sample}'s frame at loop_end="
                          f"{zone.opcodes['loop_end']} is its loop's first frame again, which would"
                          " play twice at every wrap; its source writes the frame after the loop"
                          " (see from_source_sfz)")
    if plan.normalize == "loudness":
        loudness = {name: momentary_loudness(ffmpeg, staging / "samples" / name) for name in infos}
        aim = min(loudness.values()) + LOUDNESS_SLACK_DB
        for zone in plan.zones:
            volume = float(zone.opcodes.get("volume", 0)) + min(0.0, aim - loudness[zone.sample])
            zone.opcodes["volume"] = f"{volume:.2f}"
        under = sorted(name for name in loudness if loudness[name] < aim)
        plan.notes.append(
            f"each sample is normalized to {PEAK_DB:g} dBFS before the cut to 16 bits, and its"
            " region's volume brings it to one loudness for the whole set (the loudest 400 ms"
            " of each sample, EBU R128 momentary loudness): every key and layer plays equally"
            " loud and amp_veltrack alone turns velocity into loudness, so soft to loud is"
            " continuous across the layers, whose sound still changes with velocity;"
            f" {len(under)} of {len(loudness)} samples, too sharp to reach it under"
            f" {PEAK_DB:g} dBFS, play up to {LOUDNESS_SLACK_DB:g} dB under it")
    elif plan.normalize:
        def group(zone):
            return (zone.opcodes.get("lovel"), zone.opcodes.get("hivel")) \
                if plan.normalize == "layer" else None
        loudest = {}
        for zone in plan.zones:
            loudest[group(zone)] = max(loudest.get(group(zone), -math.inf), peaks[zone.sample])
        for zone in plan.zones:
            volume = (float(zone.opcodes.get("volume", 0)) + peaks[zone.sample]
                      - loudest[group(zone)])
            zone.opcodes["volume"] = f"{volume:.2f}"
        against = ("the loudest sample of its velocity layer" if plan.normalize == "layer"
                   else "the loudest sample of the set")
        plan.notes.append(
            f"each sample is normalized to {PEAK_DB:g} dBFS before the cut to 16 bits, and its"
            f" region's volume puts it back at its recorded level against {against}")
    peak_db = None
    if plan.kind == "instrument":
        peak_db = trim_to_peak(ffmpeg, plan, staging / "samples")
    files = sorted(infos)
    size = sum((staging / "samples" / name).stat().st_size for name in files)
    description = {
        "name": plan.title,
        "kind": plan.kind,
        **({"category": plan.category} if plan.category else {}),
        "maker": source["maker"],
        "licence": kitlib.LICENCE,
        "licence_page": source["licence_page"],
        "licence_quote": source["licence_quote"],
        "source": source["title"],
        "source_url": source_url(source),
    }
    if plan.kind == "instrument":
        sfz_name = f"{plan.name}.sfz"
        (staging / sfz_name).write_text(format_sfz(plan, source, plan.zones), encoding="utf-8")
        size += (staging / sfz_name).stat().st_size
        keys = [int(zone.opcodes[name]) for zone in plan.zones for name in ("lokey", "hikey")]
        description.update(sfz=sfz_name, zones=len(plan.zones), keys=[min(keys), max(keys)],
                           loops=any("loop_start" in zone.opcodes for zone in plan.zones),
                           peak_db=round(peak_db, 2))
        entries.append({
            "path": sfz_name, "source": plan.source, "made_by": "written",
            "licence_page": source["licence_page"] if plan.derived_from else kitlib.CC0_DEED,
            **({"origin": source_url(source), "member": plan.derived_from}
               if plan.derived_from else {}),
        })
    if plan.readme:
        (staging / "README.md").write_text(format_readme(plan, source, description),
                                           encoding="utf-8")
        size += (staging / "README.md").stat().st_size
        entries.append({"path": "README.md", "source": "dawsome-kit", "made_by": "written",
                        "licence_page": kitlib.CC0_DEED})
    description.update(
        files=len(files),
        size=size,
        decoded_size=sum(info.decoded_size for info in infos.values()),
        sample_rates=sorted({info.rate for info in infos.values()}),
        channels=sorted({info.channels for info in infos.values()}),
        notes=plan.notes,
    )
    kitlib.write_json(staging / "set.json", description)
    entries.append({"path": "set.json", "source": "dawsome-kit", "made_by": "written",
                    "licence_page": kitlib.CC0_DEED})
    target = kitlib.SETS / plan.name
    if target.exists():
        shutil.rmtree(target)
    staging.rename(target)
    for entry in entries:
        path = target / entry["path"]
        entry.update(path=kitlib.relative(path), size=path.stat().st_size,
                     sha256=kitlib.sha256_file(path), licence=kitlib.LICENCE)
    print(f"build: {plan.name}: {len(files)} samples, {size / 1e6:.1f} MB on disk,"
          f" {description['decoded_size'] / 1e6:.1f} MB decoded")
    return entries


def trim_to_peak(ffmpeg, plan, samples):
    """Lowers every region's volume by one amount, if needed, so that no full-velocity note
    peaks above PEAK_DB; returns the loudest such note's peak in dBFS after the trim."""
    measured = {name: playback_peak_db(ffmpeg, samples / name)
                for name in {zone.sample for zone in plan.zones}}
    loudest = max(measured[zone.sample] + float(zone.opcodes.get("volume", 0))
                  for zone in plan.zones)
    trim = math.floor(min(0.0, PEAK_DB - loudest) * 100) / 100
    if trim < 0:
        for zone in plan.zones:
            zone.opcodes["volume"] = f"{float(zone.opcodes.get('volume', 0)) + trim:.2f}"
        plan.notes.append(f"volume lowered by {-trim:.2f} dB so a full-velocity note peaks at"
                          f" or below {PEAK_DB:g} dBFS (its loudest note reaches"
                          f" {loudest:+.2f} dBFS when resampled)")
    return loudest + trim


def download_sets(downloads, sources):
    """The manifest's entries for the sources the DAW downloads itself."""
    entries = []
    for source_id, about in DOWNLOAD_SETS.items():
        source = sources[source_id]
        root = extracted(downloads, source_id)
        sfz = kitlib.read_sfz(root / about["sfz"])
        samples = {(root / about["sfz"]).parent / sfz.sample_path(region)
                   for region in sfz.regions}
        missing = [path for path in samples if not path.is_file()]
        if missing:
            raise Problem(f"{about['sfz']} names {len(missing)} missing samples")
        outside = sorted(sfz.used - kitlib.SFZ_OPCODES - kitlib.SFZ_DIRECTIVES
                         - {f"<{header}>" for header in kitlib.SFZ_HEADERS})
        entries.append({
            "set": about["set"], "name": about["name"], "source": source_id,
            "url": source["archive"]["url"], "sha256": source["archive"]["sha256"],
            "size": source["archive"]["size"], "licence": kitlib.LICENCE,
            "licence_page": source["licence_page"], "licence_quote": source["licence_quote"],
            "maker": source["maker"], "sfz": about["sfz"], "regions": len(sfz.regions),
            "samples": len(samples),
            "decoded_size": sum(kitlib.audio_info(path).decoded_size for path in samples),
            "outside_subset": outside, "notes": about["notes"],
        })
    return entries


def unused_files(sources, built):
    """Repository files sources.json fetches that no set uses."""
    used = {(plan.source, zone.member) for plan in built for zone in plan.zones}
    used |= {(plan.source, mic.member) for plan in built for zone in plan.zones
             for mic in zone.mics}
    # A repository source's .sfz, which a plan reads its regions from.
    used |= {(plan.source, plan.derived_from) for plan in built if plan.derived_from}
    # A committed set's download: its samples' made_from names recordings in it, which check.py
    # holds to the files sources.json pins, so every pinned file of it stays fetched.
    for name in COMMITTED.values():
        used |= {(name, entry["path"]) for entry in sources[name].get("files", [])}
    used.add(("sonic-pi", SONIC_PI_README))
    return [f"{name}: {entry['path']}" for name, source in sources.items()
            for entry in source.get("files", []) if (name, entry["path"]) not in used]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--downloads", type=Path, default=kitlib.DOWNLOADS)
    parser.add_argument("set", nargs="*")
    args = parser.parse_args()
    try:
        ffmpeg = require_ffmpeg()
        sources = kitlib.read_json(kitlib.SOURCES)["sources"]
        every = plans(args.downloads, sources)
        names = [plan.name for plan in every]
        unknown = sorted(set(args.set) - set(names) - set(COMMITTED))
        if unknown:
            raise Problem(f"no such set: {', '.join(unknown)}"
                          f" (sets: {', '.join(names + list(COMMITTED))})")
        for name in COMMITTED:
            if not args.set or name in args.set:
                print(f"build: {name}: committed as made; kept as it is")
        chosen = [plan for plan in every if not args.set or plan.name in args.set]
        kitlib.SETS.mkdir(exist_ok=True)
        files = []
        if kitlib.MANIFEST.exists():
            # A partial build keeps every other set's entries; a full one only the committed sets'.
            kept = tuple(f"sets/{name}/" for name in COMMITTED)
            rebuilt = tuple(f"sets/{plan.name}/" for plan in chosen)
            files = [entry for entry in kitlib.read_json(kitlib.MANIFEST)["files"]
                     if not entry["path"].startswith(rebuilt)
                     and (args.set or entry["path"].startswith(kept))]
        for plan in chosen:
            files += build_set(ffmpeg, args.downloads, sources, plan)
        kitlib.write_json(kitlib.MANIFEST, {
            "about": "Every file under sets/ with where it came from, its `licence` and its"
                     " SHA-256 (check.py --manifest verifies them), and the download sets the"
                     " DAW fetches itself. Written by build.py.",
            "files": sorted(files, key=lambda entry: entry["path"]),
            "downloads": download_sets(args.downloads, sources),
        })
        for line in unused_files(sources, every):
            print(f"build: fetched but unused: {line}", file=sys.stderr)
    except Problem as error:
        print(f"build: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
