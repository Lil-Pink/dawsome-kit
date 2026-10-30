#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Measures every set Dawsome ships, and reference libraries of the same instruments, through
one sampler (Dawsome's own `dawsome render`), and writes a listening set: no ears, only
numbers and files to listen to.

usage: python3 -I tools/compare.py [--out DIR] [--dawsome BIN] [--dawsome-repo DIR]
                            [--refs INDEX] [--pair CLASS=SET].. [--only NAME]..

What it does, for our sets (always) and for the references (with --refs):
- inventory, from the .sfz and the files: key range, velocity layers, round robins, loops,
  release samples, sample rates and bits, files and size;
- measures, from a render of one note at the set's class's key (CLASS_NOTES: basses at C2,
  the 808 at C1, piano, leads, plucks and electric pianos at C4, the rest at C3; our sets by
  the class they are named for, `set_note`) at velocity 100 (and 40 and 120 for the dynamics),
  each held 4 s:
  integrated loudness (ITU-R BS.1770-4), sample peak, spectral centroid and 85 % rolloff at
  the onset and 500 ms later, attack (10-90 %), decay to -20 dB, stereo width, pitch error;
  from the file the note plays: the noise floor of its tail, how loud its last 20 ms are, and
  for a looping zone the seam (level and spectral jump, click, how well it matches);
- the listening set: one phrase per pitched set, one 2-bar pattern per drum hit and a beat
  per kit, each rendered and normalized to -18 LUFS (32-bit float WAV, so nothing clips).

Our sets: dawsome-kit's sets/, the electronic sets in the Dawsome repository's
assets/instruments/ and its starter kit (assets/sounds/kit.json). --refs takes the index the
reference download wrote: a JSON list (or an object holding one) of {class, library, folder,
format, `licence`, tag}; an SFZ library is read as Dawsome would play it at default controllers
(`flatten_sfz` says what that leaves out), a pitched pack of sound files becomes a generated
SFZ (one region per file, keyed by its note name or by pitch detection), a drum pack plays as
the starter kit does, one sampler channel per file. Generated files go under --out, never
beside the references.

The references are untrusted data: run this with python -I, nothing in them is run, and an
SFZ's #include or sample path that leaves its library's folder is refused.

Needs numpy, scipy and soundfile (in a venv outside the repository), and a
built `dawsome` (CARGO_TARGET_DIR/release/dawsome by default). Renders are offline: no audio
or MIDI device is opened, and they run with HOME and the XDG folders in a temporary folder
under --out, which is checked to be empty afterwards.
"""

import argparse
import json
import math
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

# -I keeps the script's folder off sys.path; this adds back exactly the repository's folder.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import kitlib  # noqa: E402

try:
    import numpy as np
    import soundfile as sf
    from scipy.signal import lfilter
except ImportError as missing:  # pragma: no cover - depends on the interpreter
    np = sf = lfilter = None
    IMPORT_ERROR = missing
else:
    IMPORT_ERROR = None

ROOT = kitlib.ROOT
RATE = 48000
TEMPO_MBPM = 120000
PPQ = 96
BAR = 4 * PPQ                       # ticks; 2 s at 120 BPM
SECONDS_PER_TICK = 60.0 / (TEMPO_MBPM / 1000.0) / PPQ
TARGET_LUFS = -18.0
MATCH_VELOCITY = 100                # the velocity byte every row is measured at
DYNAMICS = (40, 120)                # the soft and hard probes
PHRASE_SPAN = 12                    # the phrase plays root .. root + 12
DEFAULT_ROOT = 48                   # C3, with C4 = 60: a class's note unless CLASS_NOTES says
# The note each class is measured and played at (C4 = 60), where its instruments live. At C3 for
# all, the basses sat at or past the top of their range (the electric bass's keys end at 45) and
# the leads, plucks and electric piano an octave under the root they are made at. Basses at C2,
# the 808 at C1, the piano at C4, leads, plucks and electric pianos at C4 (the shipped lead-c4,
# pluck-c4 and ep-c4 are rooted there), the pads at C3 (pad-warm-c3, pad-bright-c3); any other
# class at C3. Our sets take the note of the class they are named for (set_note).
CLASS_NOTES = {"piano": 60, "bass-electric": 36, "bass-guitar-5": 36, "bass-synth": 36,
               "bass-analog": 36, "bass-808": 24, "lead": 60, "pluck": 60, "ep": 60,
               "pad-warm": 48, "pad-bright": 48}
HOLD_BARS = 2                       # the probe note is held 4 s
PROBE_SECTION_BARS = 4              # held 4 s, then 4 s for its release
DRUM_SECTION_BARS = 2
MAX_PACK_HITS = 64                  # drum hits rendered per pack
MAX_SFZ_BYTES = 16 << 20            # Dawsome's own text limit per set
DEFAULT_CC = {7: 100, 10: 64, 11: 127}   # SFZ players' initial controller values; others 0
AUDIO_EXTENSIONS = {".wav", ".flac", ".aif", ".aiff", ".ogg"}


def velocity(byte):
    """The project velocity whose byte Dawsome reads back as `byte`: round(v * 128) = byte,
    exactly, since byte / 128 is a short binary fraction."""
    return byte / 128.0


# ---------------------------------------------------------------- the phrase and the patterns

MOTIF = (0, 4, 7, 12)
RUN = (0, 2, 4, 5, 7, 9, 11, 12, 11, 9, 7, 5, 4, 2, 0)
HELD_START, HELD_LENGTH = 2 * BAR, 6 * BAR          # 12 s from bar 3
PHRASE_BARS = 10                                      # 20 s rendered


def phrase_notes(root):
    """(tick, length, key, velocity byte) of the listening phrase: the motif as eighths at
    velocity 40, 80 and 120, a run of 32nds up and down the octave, then the root held 12 s.

    Why 12 s: at each class's note, the loops of ours and of the references first wrap up to
    9.6 s into the held note (2026-10-27: SampleRadar 00s Synths' pads and lead; our pad-sweep
    8.1, the piano at C4 8.0, pad-choir 6.8), so a shorter hold would end before the loop it is
    there to expose (at 9 s, 5 of the 29 looping sets never wrapped); 12 s holds every first
    wrap with 2.4 s to spare."""
    notes = []
    for index, byte in enumerate((40, 80, 120)):
        for step, offset in enumerate(MOTIF):
            notes.append((index * 192 + step * 48, 44, root + offset, byte))
    for step, offset in enumerate(RUN):
        notes.append((576 + step * 12, 11, root + offset, 80))
    notes.append((HELD_START, HELD_LENGTH, root, 80))
    return notes


def probe_notes(key, drum=False):
    """The measured notes: velocity 100, 40 and 120, each starting a section of its own
    (4 bars for a pitched set, held 2 bars; 2 bars for a drum, a step hit)."""
    bars = DRUM_SECTION_BARS if drum else PROBE_SECTION_BARS
    length = 0 if drum else HOLD_BARS * BAR
    return [(i * bars * BAR, length, key, byte)
            for i, byte in enumerate((MATCH_VELOCITY,) + DYNAMICS)]


HIT_PATTERN = ((0, 40), (96, 80), (192, 120), (288, 80),
               (384, 80), (408, 80), (432, 80), (456, 80), (480, 120))
HIT_BARS = 3


def beat_notes():
    """(tick, role, velocity byte) of the kit beat: four bars of a rock groove (eighth hats,
    an open hat closing bars 2 and 4, a tom fill on bar 4's last beat), a crash on bar 1 and
    a last crash with a kick on bar 5, left to ring."""
    notes = []
    for bar in range(4):
        base = bar * BAR
        fill = bar == 3
        for eighth in range(8):
            tick = base + eighth * 48
            if fill and tick >= base + 288:
                continue
            if bar == 0 and eighth == 0:
                notes.append((tick, "crash", 120))
                continue
            if bar in (1, 3) and eighth == 7:
                notes.append((tick, "hat_open", 90))
            else:
                notes.append((tick, "hat_closed", 90 if eighth % 2 == 0 else 60))
        notes.append((base, "kick", 110))
        notes.append((base + 192, "kick", 110))
        if bar in (1, 3):
            notes.append((base + 240, "kick", 100))
        notes.append((base + 96, "snare", 110))
        if fill:
            for step, role in enumerate(("tom_hi", "tom_hi", "tom_mid", "tom_lo")):
                notes.append((base + 288 + step * 24, role, 100))
        else:
            notes.append((base + 288, "snare", 110))
    notes.append((4 * BAR, "crash", 120))
    notes.append((4 * BAR, "kick", 110))
    return sorted(notes)


BEAT_BARS = 6


# ------------------------------------------------------------------------- loudness (BS.1770)

def k_weighting(rate):
    """The two BS.1770 K-weighting biquads at `rate`, ((b, a), (b, a)): the high shelf and the
    RLB high-pass, designed from the analogue parameters libebur128 fits to the standard's
    48 kHz coefficients (which they reproduce; tests/test_compare.py holds them)."""
    f0, gain, q = 1681.974450955533, 3.999843853973347, 0.7071752369554196
    k = math.tan(math.pi * f0 / rate)
    vh = 10 ** (gain / 20)
    vb = vh ** 0.4996667741545416
    a0 = 1 + k / q + k * k
    shelf = ([(vh + vb * k / q + k * k) / a0, 2 * (k * k - vh) / a0, (vh - vb * k / q + k * k) / a0],
             [1.0, 2 * (k * k - 1) / a0, (1 - k / q + k * k) / a0])
    f0, q = 38.13547087602444, 0.5003270373238773
    k = math.tan(math.pi * f0 / rate)
    a0 = 1 + k / q + k * k
    highpass = ([1.0, -2.0, 1.0], [1.0, 2 * (k * k - 1) / a0, (1 - k / q + k * k) / a0])
    return shelf, highpass


def integrated_loudness(x, rate):
    """Integrated loudness in LUFS of `x` (frames x channels, left and right weighted 1):
    K-weighting, 400 ms blocks every 100 ms, the -70 LUFS absolute and -10 LU relative gates
    (ITU-R BS.1770-4). None when no block passes the gates (silence, or under 400 ms)."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    shelf, highpass = k_weighting(rate)
    y = lfilter(*highpass, lfilter(*shelf, x, axis=0), axis=0)
    block, step = int(round(0.4 * rate)), int(round(0.1 * rate))
    if len(y) < block:
        return None
    power = np.concatenate((np.zeros((1, y.shape[1])), np.cumsum(y * y, axis=0)))
    starts = np.arange(0, len(y) - block + 1, step)
    z = (power[starts + block] - power[starts]) / block          # blocks x channels
    total = z.sum(axis=1)
    with np.errstate(divide="ignore"):
        loud = -0.691 + 10 * np.log10(total)
    kept = loud > -70
    if not kept.any():
        return None
    relative = -0.691 + 10 * math.log10(total[kept].mean()) - 10
    kept &= loud > relative
    if not kept.any():
        return None
    return -0.691 + 10 * math.log10(total[kept].mean())


# ------------------------------------------------------------------------------- measures

def db(value):
    return 20 * math.log10(max(float(value), 1e-12))


def moving_rms(m, n):
    """RMS over the `n` samples centered on each sample of the mono signal `m`."""
    n = max(1, int(n))
    sq = np.concatenate(([0.0], np.cumsum(m * m)))
    half = n // 2
    lo = np.clip(np.arange(len(m)) - half, 0, len(m))
    hi = np.clip(np.arange(len(m)) - half + n, 0, len(m))
    return np.sqrt(np.maximum((sq[hi] - sq[lo]) / n, 0.0))


def spectrum_at(m, rate, start, size=2048):
    """(centroid Hz, 85 % rolloff Hz) of the Hann-windowed `size` samples from `start`, over
    20 Hz to 20 kHz; None when the window is past the end or 60 dB under the peak."""
    seg = m[start:start + size]
    peak = float(np.abs(m).max()) if len(m) else 0.0
    if len(seg) < size or peak == 0 or np.sqrt(np.mean(seg ** 2)) < peak * 1e-3:
        return None
    power = np.abs(np.fft.rfft(seg * np.hanning(size))) ** 2
    freqs = np.fft.rfftfreq(size, 1 / rate)
    band = (freqs >= 20) & (freqs <= 20000)
    power, freqs = power[band], freqs[band]
    total = power.sum()
    if total <= 0:
        return None
    centroid = float((freqs * power).sum() / total)
    rolloff = float(freqs[np.searchsorted(np.cumsum(power), 0.85 * total)])
    return centroid, rolloff


def onset(m, peak):
    """The first sample at or above -40 dB of `peak`."""
    above = np.nonzero(np.abs(m) >= peak * 0.01)[0]
    return int(above[0]) if len(above) else 0


def attack_ms(m, rate, start, stop):
    """(ms, still rising) of the 10 % to 90 % rise of the 5 ms RMS envelope from `start`, to
    90 % of its loudest within the first second (a step reads about 4 ms: the envelope's own
    rise). Why the first second: a tremolo or a rotary speaker puts the loudest point of a
    held note anywhere; "still rising" when the envelope later (before `stop`) gets more than
    1 dB louder than that, as a slow swell does."""
    env = moving_rms(m[start:stop], 0.005 * rate)
    first = env[:int(rate)]
    top = first.max() if len(first) else 0.0
    if top <= 0:
        return None, False
    t10 = int(np.argmax(first >= 0.1 * top))
    t90 = int(np.argmax(first >= 0.9 * top))
    return (t90 - t10) / rate * 1000, bool(env.max() > top * 10 ** (1 / 20))


def decay_ms(m, rate, start, stop):
    """Time from the 20 ms RMS envelope's peak to its first fall 20 dB below it, before
    `stop`; None when it never falls that far there (the note sustains)."""
    env = moving_rms(m[start:stop], 0.02 * rate)
    i_peak = int(np.argmax(env))
    below = np.nonzero(env[i_peak:] < env[i_peak] * 0.1)[0]
    return float(below[0]) / rate * 1000 if len(below) else None


def width(x):
    """(L/R correlation, side over mid in dB) of a stereo section; (1.0, None) when the two
    sides are the same (mono)."""
    left, right = x[:, 0], x[:, 1]
    if np.array_equal(left, right):
        return 1.0, None
    mid, side = (left + right) / 2, (left - right) / 2
    corr = float(np.corrcoef(left, right)[0, 1]) if left.std() > 0 and right.std() > 0 else 0.0
    return corr, 10 * math.log10(max(float((side ** 2).sum()), 1e-30)
                                 / max(float((mid ** 2).sum()), 1e-30))


def yin(seg, rate, fmin, fmax):
    """The median f0 in Hz of `seg` by YIN (de Cheveigne & Kawahara 2002: the cumulative-mean-
    normalized difference, its first dip under 0.15 or its minimum, parabolic refinement) over
    20 ms hops, with the share of frames whose dip is under 0.3; (None, 0) without frames."""
    tau_lo = max(2, int(rate / fmax))
    tau_hi = int(math.ceil(rate / fmin)) + 2
    size = int(max(0.04, 3.0 / fmin) * rate)
    estimates, voiced, frames = [], 0, 0
    for s in range(0, len(seg) - size - tau_hi, int(0.02 * rate)):
        frame = seg[s:s + size + tau_hi]
        a = frame[:size]
        if not a.any():
            continue
        frames += 1
        n = 1 << (len(frame) + size).bit_length()
        r = np.fft.irfft(np.conj(np.fft.rfft(a, n)) * np.fft.rfft(frame, n), n)[:tau_hi + 1]
        sq = np.concatenate(([0.0], np.cumsum(frame ** 2)))
        d = sq[size] + (sq[np.arange(tau_hi + 1) + size] - sq[np.arange(tau_hi + 1)]) - 2 * r
        d[0] = 0.0
        cmnd = np.ones(tau_hi + 1)
        cmnd[1:] = d[1:] * np.arange(1, tau_hi + 1) / np.maximum(np.cumsum(d[1:]), 1e-30)
        window = cmnd[tau_lo:tau_hi]
        dips = np.nonzero(window < 0.15)[0]
        if len(dips):
            t = tau_lo + int(dips[0])
            while t + 1 < tau_hi and cmnd[t + 1] < cmnd[t]:
                t += 1
        else:
            t = tau_lo + int(np.argmin(window))
        if cmnd[t] < 0.3:
            voiced += 1
        if 0 < t < tau_hi:
            c0, c1, c2 = cmnd[t - 1], cmnd[t], cmnd[t + 1]
            denom = c0 - 2 * c1 + c2
            t = t + (0.5 * (c0 - c2) / denom if denom > 0 else 0.0)
        estimates.append(rate / t)
    if not estimates:
        return None, 0.0
    return float(np.median(estimates)), voiced / frames


def key_frequency(key):
    return 440.0 * 2 ** ((key - 69) / 12)


def pitch_cents(m, rate, start, stop, key):
    """(cents, octaves) of the steady part of a note at `key` (from 0.15 s after its start,
    up to 1 s) against the key: searched within 6 semitones, then, when under half the frames
    voice there, from 25 Hz to 4.2 kHz, the octaves then saying how far its period lies
    (-1: it sounds an octave below the key); (None, None) when neither voices."""
    seg = m[start + int(0.15 * rate):min(stop, start + int(1.15 * rate))]
    nominal = key_frequency(key)
    f0, voiced = yin(seg, rate, nominal / 2 ** 0.5, nominal * 2 ** 0.5)
    if f0 is not None and voiced >= 0.5:
        return 1200 * math.log2(f0 / nominal), 0
    f0, voiced = yin(seg, rate, 25.0, 4200.0)
    if f0 is None or voiced < 0.5:
        return None, None
    octaves = round(math.log2(f0 / nominal))
    return 1200 * math.log2(f0 / (nominal * 2 ** octaves)), octaves


def heard_seams(x, regions, key, byte, note_start, note_frames):
    """How the wraps of the loudest looping zone a held note sounds come out of the render `x`
    (frames x 2), where the note starts at frame `note_start` and is held `note_frames`: the
    wrap frames follow Dawsome's loop scaling and the voice's speed. A loop whose length at the
    project's rate is a whole number of frames (or a file at that rate) is converted with the
    file, its start rounded half up and its inclusive end as the frame after it, rounded, less
    one; any other loop Dawsome reads at its file's own rate, so it wraps every loop length of
    file frames, a fraction of an output frame apart (dawsome 7c11378, 2026-10-27: before, those
    were rounded too and every wrap slipped). Per wrap (up to 50 in the hold):
    the click (the largest second-difference spike within 6 frames of the wrap, over the local
    RMS) and the spectral and level jump across it (50 ms windows), each beside its median at
    the midpoints between wraps. None when the note sounds no looping zone."""
    zones = [r for r in zones_for(regions, key, byte) if loops(r)]
    if not zones:
        return None
    zone = max(zones, key=lambda r: r["volume"])
    file_rate = sf.info(str(zone["sample"])).samplerate
    speed = 2 ** ((key - zone["pitch_keycenter"] + zone["transpose"]) / 12 + zone["tune"] / 1200)

    length = zone["loop_end"] + 1 - zone["loop_start"]
    at_file_rate = file_rate != RATE and length * RATE % file_rate != 0

    def scaled(frame):
        if at_file_rate:
            return frame * RATE / file_rate
        return int(math.floor(frame * RATE / file_rate + 0.5))

    start, end_ex = scaled(zone["loop_start"]), scaled(zone["loop_end"] + 1)
    first = (end_ex - scaled(zone["offset"])) / speed
    period = (end_ex - start) / speed
    wraps = [first + k * period for k in range(50) if first + k * period < note_frames - RATE * 0.06]
    out = {"zone_file": Path(zone["sample"]).name, "first_wrap_s": first / RATE,
           "period_s": period / RATE, "wraps": len(wraps)}
    if not wraps:
        return out
    m = x.mean(axis=1)
    win, n = int(0.05 * RATE), int(0.005 * RATE)

    def at(frame):
        f = int(round(note_start + frame))
        clicks = []
        for ch in x.T:
            seg = ch[f - n:f + n]
            d2 = np.abs(np.diff(seg, 2))
            centre = np.zeros(len(d2), bool)
            centre[n - 8:n + 6] = True
            local = float(np.sqrt(np.mean(d2[~centre] ** 2)))
            clicks.append(float(d2[centre].max()) / max(local, 1e-9))
        before, after = m[f - win:f], m[f:f + win]
        return (max(clicks), spectral_distance(before, after, RATE),
                db(np.sqrt(np.mean(after ** 2)) / max(np.sqrt(np.mean(before ** 2)), 1e-12)))

    seams = [at(w) for w in wraps]
    middles = [at(w + period / 2) for w in wraps if w + period / 2 < note_frames - RATE * 0.06]
    if not middles:
        middles = [at(w - period / 2) for w in wraps if w - period / 2 > RATE * 0.06]
    out.update({
        "click": float(np.median([c for c, _, _ in seams])),
        "click_max": float(max(c for c, _, _ in seams)),
        "click_inside": float(np.median([c for c, _, _ in middles])) if middles else None,
        "spectral_jump_db": float(np.median([s for _, s, _ in seams])),
        "spectral_inside_db": float(np.median([s for _, s, _ in middles])) if middles else None,
        "level_jump_db": float(np.median([lv for _, _, lv in seams])),
    })
    return out


def section_measures(x, rate, held_frames, key=None):
    """The measures of one probe note: `x` is its section (frames x 2) of the render, the
    note starting at frame 0 and held `held_frames` (the section's length for a drum)."""
    m = x.mean(axis=1)
    peak = float(np.abs(x).max())
    if peak == 0:
        return {"silent": True}
    start = onset(m, peak)
    at_500 = start + int(0.5 * rate)
    first, later = spectrum_at(m, rate, start), spectrum_at(m, rate, at_500)
    corr, side_db = width(x)
    attack, rising = attack_ms(m, rate, max(0, start - int(0.01 * rate)), held_frames)
    cents, octaves = pitch_cents(m, rate, start, held_frames, key) if key is not None else (None, None)
    return {
        "lufs": integrated_loudness(x, rate),
        "peak_db": db(peak),
        "onset_ms": start / rate * 1000,
        "centroid_onset": first and first[0], "rolloff_onset": first and first[1],
        "centroid_500ms": later and later[0], "rolloff_500ms": later and later[1],
        "attack_ms": attack, "attack_rising": rising,
        "decay20_ms": decay_ms(m, rate, start, held_frames),
        "lr_correlation": corr, "side_db": side_db,
        "pitch_cents": cents, "pitch_octaves": octaves,
    }


def tail_measures(path, region):
    """From the sound file a zone plays: its tail's noise floor (the 10th percentile of the
    50 ms RMS windows after the sound first falls 40 dB below its loudest, the last 100 ms and
    digital silence left out), that floor's spectral flatness (near 1 for hiss, near 0 for a
    tone), and the level of its last 20 ms, all in dB of its own peak. A zone whose file never
    falls 40 dB, or loops, has no tail floor."""
    x, rate = sf.read(str(path), dtype="float64", always_2d=True)
    m = x.mean(axis=1)
    peak = float(np.abs(m).max())
    out = {"file_seconds": len(m) / rate, "floor_db": None, "floor_flatness": None,
           "end_db": None}
    if peak == 0:
        return out
    tail = m[-int(0.02 * rate):]
    out["end_db"] = db(np.sqrt(np.mean(tail ** 2)) / peak)
    win, hop = int(0.05 * rate), int(0.025 * rate)
    starts = np.arange(0, max(1, len(m) - win), hop)
    rms = np.array([np.sqrt(np.mean(m[s:s + win] ** 2)) for s in starts])
    if not len(rms):
        return out
    top = rms.max()
    below = np.nonzero(rms < top * 0.01)[0]
    looping = region.get("loop_mode") in ("loop_continuous", "loop_sustain")
    if looping or not len(below):
        return out
    usable = [i for i in range(int(below[0]), len(starts))
              if starts[i] + win <= len(m) - int(0.1 * rate) and rms[i] > 1e-7]
    if not usable:
        return out
    floor = float(np.percentile(rms[usable], 10))
    out["floor_db"] = db(floor / peak)
    quiet = [i for i in usable if rms[i] <= np.percentile(rms[usable], 20)]
    power = np.mean([np.abs(np.fft.rfft(m[starts[i]:starts[i] + win] * np.hanning(win))) ** 2
                     for i in quiet], axis=0)
    freqs = np.fft.rfftfreq(win, 1 / rate)
    band = power[(freqs >= 200) & (freqs <= min(16000, rate / 2 - 1))] + 1e-30
    out["floor_flatness"] = float(np.exp(np.mean(np.log(band))) / np.mean(band))
    return out


THIRD_OCTAVES = [1000 * 2 ** (k / 3) for k in range(-13, 13)]     # 50 Hz .. 16 kHz


def band_levels(seg, rate):
    """Energy in dB per third-octave band (50 Hz to 16 kHz) of the Hann-windowed `seg`."""
    power = np.abs(np.fft.rfft(seg * np.hanning(len(seg)))) ** 2
    freqs = np.fft.rfftfreq(len(seg), 1 / rate)
    levels = []
    for centre in THIRD_OCTAVES:
        lo, hi = centre / 2 ** (1 / 6), centre * 2 ** (1 / 6)
        if hi >= rate / 2:
            break
        levels.append(10 * math.log10(float(power[(freqs >= lo) & (freqs < hi)].sum()) + 1e-20))
    return np.array(levels)


def spectral_distance(a, b, rate):
    """The RMS difference in dB of the third-octave levels of `a` and `b`, over the bands
    within 60 dB of the louder one's loudest band."""
    la, lb = band_levels(a, rate), band_levels(b, rate)
    keep = np.maximum(la, lb) > max(la.max(), lb.max()) - 60
    return float(np.sqrt(np.mean((la[keep] - lb[keep]) ** 2))) if keep.any() else 0.0


def click_ratio(y, at, rate):
    """The second difference of `y` around index `at` (3 samples either side), over the RMS of
    the second difference in the 5 ms around it with those samples left out."""
    d2 = np.abs(np.diff(y, 2))
    n = int(0.0025 * rate)
    lo, hi = max(0, at - n), min(len(d2), at + n)
    centre = np.zeros(len(d2), bool)
    centre[max(0, at - 4):min(len(d2), at + 3)] = True
    around = d2[lo:hi][~centre[lo:hi]]
    local = float(np.sqrt(np.mean(around ** 2))) if len(around) else 0.0
    spike = float(d2[max(0, at - 4):min(len(d2), at + 3)].max())
    return spike / max(local, 1e-9)


def best_end(m, start, end, rate, reach=3):
    """(offset, click) of the loop end within `reach` frames that joins the loop's start most
    smoothly (the smallest second-difference spike at the wrap, over the local RMS): 0 when
    loop_end is already the best; -1 when the frame at loop_end repeats the one at
    loop_start (an exclusive end written as an inclusive one)."""
    n = int(0.005 * rate)
    best = None
    for offset in range(-reach, reach + 1):
        e = end + offset
        if e + 1 - n < 0 or e >= len(m) or start + n > len(m):
            continue
        y = np.concatenate((m[e + 1 - n:e + 1], m[start:start + n]))
        ratio = click_ratio(y, n - 1, rate)
        if best is None or ratio < best[1] - 1e-9 or (abs(ratio - best[1]) <= 1e-9 and abs(offset) < abs(best[0])):
            best = (offset, ratio)
    return best if best else (None, None)


def seam_measures(path, region):
    """How a zone's loop sounds where it wraps, from its file: the loop's start and length in
    seconds; the level jump (dB, the 50 ms after loop_start over the 50 ms up to loop_end);
    the spectral jump (third-octave distance across the wrap, in dB) and the same distance
    between neighboring 50 ms windows inside the loop (its median, what the sound's own
    change gives); the click at the wrap and the same ratio's median inside the loop; and the
    match (correlation of the 10 ms leading into the wrap with the 10 ms leading into
    loop_start: 1 when the end flows on exactly as the start's own lead-in did)."""
    x, rate = sf.read(str(path), dtype="float64", always_2d=True)
    start, end = int(region["loop_start"]), int(region["loop_end"])
    m = x.mean(axis=1)
    win = int(0.05 * rate)
    length = end - start + 1
    out = {"loop_start_s": start / rate, "loop_seconds": length / rate}
    if length < 2 * win or start < win or end + 1 > len(m):
        out["short"] = True
        win = max(16, min(win, length // 2, start))
    pre, post = m[end + 1 - win:end + 1], m[start:start + win]
    out["level_jump_db"] = db(np.sqrt(np.mean(post ** 2)) / max(np.sqrt(np.mean(pre ** 2)), 1e-12))
    out["spectral_jump_db"] = spectral_distance(pre, post, rate)
    inside = [spectral_distance(m[p - win:p], m[p:p + win], rate)
              for p in np.linspace(start + win, end + 1 - win, 8).astype(int)] if length >= 2 * win else []
    out["spectral_inside_db"] = float(np.median(inside)) if inside else None
    n = int(0.005 * rate)
    worst = 0.0
    for ch in x.T:
        y = np.concatenate((ch[max(0, end + 1 - n):end + 1], ch[start:start + n]))
        worst = max(worst, click_ratio(y, min(n, end + 1) - 1, rate))
    out["seam_click"] = worst
    out["end_offset"], out["seam_click_best"] = best_end(m, start, end, rate)
    marks = np.linspace(start + n, end + 1 - n, 8).astype(int) if length > 2 * n else []
    out["click_inside"] = max(float(np.median([click_ratio(ch[p - n:p + n], n - 1, rate)
                                               for p in marks])) for ch in x.T) if len(marks) else None
    lead = int(0.01 * rate)
    if start >= lead:
        a, b = m[end + 1 - lead:end + 1], m[start - lead:start]
        out["match"] = (float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else None)
    else:
        out["match"] = None
    return out


# ------------------------------------------------------------------------------ SFZ regions

REGION_DEFAULTS = {"lokey": 0, "hikey": 127, "pitch_keycenter": 60, "lovel": 1, "hivel": 127,
                   "loop_mode": None, "ampeg_release": 0.0, "ampeg_attack": 0.0,
                   "amp_veltrack": 100.0, "volume": 0.0, "tune": 0.0, "transpose": 0,
                   "pan": 0.0, "offset": 0, "seq_length": 1, "seq_position": 1}
INT_OPCODES = {"lokey", "hikey", "pitch_keycenter", "lovel", "hivel", "loop_start", "loop_end",
               "offset", "end", "seq_length", "seq_position", "transpose"}


def region_of(raw, sample):
    """A region of the subset with Dawsome's defaults, from `raw` (opcode: text, as written)."""
    region = dict(REGION_DEFAULTS)
    if "key" in raw:
        key = kitlib.note_number(raw["key"])
        region.update(lokey=key, hikey=key, pitch_keycenter=key)
    for name in kitlib.SFZ_OPCODES - {"default_path", "sample", "key"}:
        if name not in raw:
            continue
        value = raw[name]
        if name in ("lokey", "hikey", "pitch_keycenter"):
            region[name] = kitlib.note_number(value)
        elif name in INT_OPCODES:
            region[name] = int(float(value))
        elif name == "loop_mode":
            region[name] = value
        else:
            region[name] = float(value)
    region["sample"] = sample
    return region


def loops(region):
    """Whether the zone loops as Dawsome plays it: a looping mode, or loop points alone."""
    mode = region.get("loop_mode")
    return "loop_end" in region and (mode in ("loop_continuous", "loop_sustain") or mode is None)


def read_set(sfz_path):
    """The regions of one of our .sfz files, every opcode in the subset."""
    sfz = kitlib.read_sfz(sfz_path)
    folder = Path(sfz_path).parent
    return [region_of(raw, (folder / sfz.sample_path(raw)).resolve()) for raw in sfz.regions
            if "sample" in raw]


def zones_for(regions, key, byte):
    """The zones the first note at `key` and velocity byte `byte` sounds, as Dawsome picks
    them: every zone holding both, each round robin at its first position."""
    return [r for r in regions if r["lokey"] <= key <= r["hikey"]
            and r["lovel"] <= byte <= r["hivel"] and r["seq_position"] == 1]


def key_range(regions):
    return [min(r["lokey"] for r in regions), max(r["hikey"] for r in regions)]


def root_key(regions, prefer=None):
    """The phrase's root: `prefer` (the class's note) when the phrase fits there, else the C
    nearest it (C3 = 48 without one) that fits, else the key nearest it from which root ..
    root + 12 lies inside the set's keys. The probe note is `prefer` whenever the set plays it
    (run_set), else this key."""
    lo, hi = key_range(regions)
    playable = {k for r in regions for k in range(r["lokey"], r["hikey"] + 1)}
    centre = DEFAULT_ROOT if prefer is None else prefer

    def fits(root):
        return all(k in playable for k in range(root, root + PHRASE_SPAN + 1))

    if prefer is not None and fits(prefer):
        return prefer
    for root in sorted(range(0, 128 - PHRASE_SPAN, 12), key=lambda k: abs(k - centre)):
        if fits(root):
            return root
    fitting = [k for k in range(lo, hi - PHRASE_SPAN + 1) if fits(k)]
    if fitting:
        return min(fitting, key=lambda k: abs(k - centre))
    return min(range(lo, hi + 1), key=lambda k: abs(k - centre))


def class_note(cls):
    """The key a class is measured and played at (CLASS_NOTES; C3 when it names none)."""
    return CLASS_NOTES.get(str(cls), DEFAULT_ROOT)


def set_note(name):
    """The key one of our sets is measured at: its class's, the class it is named for (every
    word of the class in its name, as bass-808 in bass-808-c1, or every word of its name in the
    class, as bass-guitar in bass-guitar-5); C3 when none is. By name and not by the index, so
    that our sets measure the same with or without --refs."""
    words = set(tokens(name))
    for cls, key in CLASS_NOTES.items():
        theirs = set(tokens(cls))
        if theirs <= words or words <= theirs:
            return key
    return DEFAULT_ROOT


def note_name(key):
    names = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
    return f"{names[key % 12]}{key // 12 - 1}"


# --------------------------------------------------------------- reference SFZ, as data only

TRIGGER_ATTACK = {"attack", "first"}
SW_OPCODES = {"sw_last", "sw_lokey", "sw_hikey", "sw_down", "sw_up", "sw_previous", "sw_vel"}


def inside(path, folder):
    """`path` resolved, when it lies inside `folder` (links followed), else None."""
    resolved = Path(path).resolve()
    return resolved if resolved.is_relative_to(Path(folder).resolve()) else None


def sfz_relative(path):
    """An SFZ sample path as a path from the .sfz's folder: backslashes as slashes, and a
    leading one dropped (Windows-made libraries write default_path=\\samples\\ meaning the
    folder beside the file, never the drive's root)."""
    return path.replace("\\", "/").lstrip("/")


def find_case(folder, relative):
    """The file `relative` names under `folder`, matching each part's case loosely when the
    exact name is not there (libraries made on Windows or macOS name `Samples/` `samples/`)."""
    exact = Path(folder) / relative
    if exact.is_file():
        return exact
    current = Path(folder)
    for part in Path(relative).parts:
        if part == "..":
            current = current.parent
            continue
        if not current.is_dir():
            return None
        match = next((c for c in current.iterdir() if c.name.lower() == part.lower()), None)
        if match is None:
            return None
        current = match
    return current if current.is_file() else None


def check_includes(sfz_path, folder):
    """Refuses an SFZ whose #include chain leaves `folder` or runs over Dawsome's limits,
    before kitlib reads it: the same walk (includes from the first file's folder, #define
    variables expanded in their paths, in file order, 16 deep at most)."""
    first = Path(sfz_path)
    defines, seen, total = {}, set(), [0]

    def walk(current, depth):
        target = inside(current, folder)
        if target is None:
            raise kitlib.Problem(f"{current}: outside the library's folder")
        if depth > 16:
            raise kitlib.Problem(f"{first}: #include goes deeper than 16 files")
        seen.add(target)
        if len(seen) > 256:
            raise kitlib.Problem(f"{first}: more than 256 includes")
        total[0] += target.stat().st_size
        if total[0] > MAX_SFZ_BYTES:
            raise kitlib.Problem(f"{first}: over {MAX_SFZ_BYTES >> 20} MB of SFZ text")
        text = re.sub(r"/\*.*?\*/", " ", target.read_bytes().decode("utf-8", "replace"),
                      flags=re.DOTALL)
        for line in text.splitlines():
            line = line.split("//", 1)[0]
            if m := re.match(r"\s*#define\s+(\$\w+)\s+(.*?)\s*$", line):
                defines[m[1]] = m[2]
                continue
            line = re.sub(r"\$\w+", lambda v: defines.get(v[0], v[0]), line)
            if m := re.match(r'\s*#include\s+"([^"]+)"', line):
                walk(first.parent / m[1].replace("\\", "/"), depth + 1)

    walk(first, 0)


def included_files(found):
    """The .sfz files another of `found` includes (fragments, not programs), resolved from
    the including file's folder with its own #define values."""
    included = set()
    for path in found:
        defines = {}
        text = path.read_bytes()[:MAX_SFZ_BYTES].decode("utf-8", "replace")
        for line in text.splitlines():
            line = line.split("//", 1)[0]
            if m := re.match(r"\s*#define\s+(\$\w+)\s+(.*?)\s*$", line):
                defines[m[1]] = m[2]
            elif m := re.match(r'\s*#include\s+"([^"]+)"', line):
                name = re.sub(r"\$\w+", lambda v: defines.get(v[0], v[0]), m[1])
                included.add((path.parent / name.replace("\\", "/")).resolve())
    return included


def playable_regions(sfz_path, folder):
    """(keys, regions) of an SFZ whose samples are there: how many distinct keys its playable
    regions cover, and how many there are; (0, 0) when it cannot be read."""
    try:
        check_includes(sfz_path, folder)
        sfz = kitlib.read_sfz(sfz_path)
    except (kitlib.Problem, OSError, ValueError):
        return 0, 0
    base, found, keys, count = Path(sfz_path).parent, {}, set(), 0
    for raw in sfz.regions:
        sample = raw.get("sample", "")
        if not sample or sample.startswith("*"):
            continue
        name = sfz_relative(sfz.control.get("default_path", "") + sample)
        if name not in found:
            found[name] = find_case(base, name) is not None
        if not found[name]:
            continue
        count += 1
        try:
            if "key" in raw:
                keys.add(kitlib.note_number(raw["key"]))
            else:
                keys.update(range(kitlib.note_number(raw.get("lokey", "0")),
                                  kitlib.note_number(raw.get("hikey", "127")) + 1))
        except kitlib.Problem:
            pass
    return len(keys), count


def wav_loop(path):
    """The first loop of a WAV file's smpl chunk as (start, end inclusive), or None. Chunks
    other than smpl are skipped unread, so a huge chunk costs nothing."""
    try:
        with open(path, "rb") as file:
            head = file.read(12)
            if head[:4] != b"RIFF" or head[8:12] != b"WAVE":
                return None
            while len(header := file.read(8)) == 8:
                chunk, size = header[:4], struct.unpack("<I", header[4:])[0]
                if chunk == b"smpl" and 60 <= size <= 1 << 20:
                    body = file.read(size)
                    if struct.unpack("<I", body[28:32])[0] >= 1:
                        start, end = struct.unpack("<II", body[44:52])
                        return (start, end) if end > start else None
                    return None
                file.seek(size + (size & 1), os.SEEK_CUR)
    except (OSError, struct.error):
        return None
    return None


def default_cc(control):
    values = dict(DEFAULT_CC)
    for name, value in control.items():
        if m := re.fullmatch(r"set_cc(\d+)", name):
            try:
                values[int(m[1])] = float(value)
            except ValueError:
                pass
    return values


def flatten_sfz(sfz_path, folder, skipped=None):
    """A reference SFZ as regions of Dawsome's subset, played at default controllers, and an
    account of what that leaves out: {"regions", "release_regions", "dropped": {why: count},
    "ignored": {opcode: count}, "notes": [..]}.

    Kept as the library means them: key and velocity ranges (note_offset and octave_offset
    applied), round robins (seq_*, and lorand/hirand groups made sequential), loops (points
    from the .sfz, else the WAV's smpl chunk), volume (amplitude folded in, and amplitude_ccN
    at the controller's default: its share of N's value, as a mic fader's), tune, transpose,
    pan, offset, end, attack, release, veltrack; pitch_keytrack=0 becomes one region per key.
    Left out, and counted: release triggers, regions a controller, a keyswitch other than the
    default, a pitch bend or another MIDI channel would select or silence, generated samples;
    every other opcode is ignored (filters, envelopes beyond attack and release, curves, a
    controller's curve too, choke groups...).
    """
    check_includes(sfz_path, folder)
    sfz = kitlib.read_sfz(sfz_path)
    control = sfz.control
    base = Path(sfz_path).parent
    cc = default_cc(control)
    shift = 0
    for name, scale in (("note_offset", 1), ("octave_offset", 12)):
        if name in control:
            shift += scale * int(float(control[name]))
    sw_default = control.get("sw_default")
    if sw_default is None:
        sw_default = next((r["sw_default"] for r in sfz.regions if "sw_default" in r), None)
    sw_values = sorted({kitlib.note_number(r["sw_last"]) for r in sfz.regions if "sw_last" in r})
    if sw_default is not None:
        sw_keep = kitlib.note_number(sw_default)
    else:
        sw_keep = sw_values[0] if sw_values else None
    known = (kitlib.SFZ_OPCODES | {"key", "trigger", "loopmode", "loopstart", "loopend",
                                   "amplitude", "pitch_keytrack", "lorand", "hirand", "sw_default",
                                   "lochan", "hichan", "lobend", "hibend", "loop_crossfade"}
             | SW_OPCODES)
    # amplitude_ccN (or amplitude_onccN) is a share of the amplitude that controller N sets: A %
    # of its value / 127, multiplied in (sfizz, ARIA), so a library's mic faders are mixed at their
    # defaults (Big Rusty Drums: amplitude_cc81=100 with set_cc81=40 plays its snare top mic at
    # 40/127), and a region whose controller is at 0 is silent. Read linearly: a curve named for
    # the controller (amplitude_curveccN) is counted under "ignored".
    amplitude_cc = re.compile(r"amplitude_(?:on)?cc(\d+)")
    dropped, ignored, kept, release = {}, {}, [], 0
    crossfades, unreadable_seen = set(), set()

    def drop(why):
        dropped[why] = dropped.get(why, 0) + 1

    for raw in sfz.regions:
        raw = dict(raw)
        for alias, name in (("loopmode", "loop_mode"), ("loopstart", "loop_start"),
                            ("loopend", "loop_end")):
            if alias in raw and name not in raw:
                raw[name] = raw.pop(alias)
        for name in raw:
            if name not in known and not re.fullmatch(r"(lo|hi)cc\d+", name) and not (
                    amplitude_cc.fullmatch(name)):
                ignored[name] = ignored.get(name, 0) + 1
        trigger = raw.get("trigger", "attack")
        if trigger in ("release", "release_key"):
            release += 1
            continue
        if trigger not in TRIGGER_ATTACK:
            drop(f"trigger={trigger}")
            continue
        if any(re.fullmatch(r"on_(lo|hi)cc\d+", name) for name in raw):
            drop("started by a controller")
            continue
        selected = True
        for name, value in raw.items():
            if m := re.fullmatch(r"(lo|hi)cc(\d+)", name):
                number, level = int(m[2]), float(value)
                at = cc.get(number, 0.0)
                if (m[1] == "lo" and at < level) or (m[1] == "hi" and at > level):
                    selected = False
        if not selected:
            drop("a controller at its default does not select it")
            continue
        share = 1.0
        for name, value in raw.items():
            if m := amplitude_cc.fullmatch(name):
                share *= float(value) / 100 * cc.get(int(m[1]), 0.0) / 127
        if share == 0:
            drop("a controller at its default silences it")
            continue
        if share != 1.0:
            raw["amplitude"] = str(float(raw.get("amplitude", 100)) * share)
        if "sw_last" in raw and sw_keep is not None and kitlib.note_number(raw["sw_last"]) != sw_keep:
            drop("another keyswitch")
            continue
        if any(name in raw for name in ("sw_down", "sw_up", "sw_previous")):
            drop("a held or previous keyswitch")
            continue
        if not (int(float(raw.get("lochan", 1))) <= 1 <= int(float(raw.get("hichan", 16)))):
            drop("another MIDI channel")
            continue
        if not (float(raw.get("lobend", -8192)) <= 0 <= float(raw.get("hibend", 8192))):
            drop("a pitch bend")
            continue
        sample = raw.get("sample", "")
        if not sample or sample.startswith("*"):
            drop("no sample, or a generated one")
            continue
        if raw.get("end") in ("-1", "0") or raw.get("hikey") == "-1":
            drop("switched off (end or hikey -1)")
            continue
        relative = sfz_relative(control.get("default_path", "") + sample)
        found = find_case(base, relative)
        if found is None or inside(found, folder) is None:
            drop("its sample is missing or outside the library")
            continue
        if found.suffix.lower() not in AUDIO_EXTENSIONS:
            drop(f"a {found.suffix} sample Dawsome does not decode")
            continue
        if macos_junk(found) or (why := unreadable(found)) is not None:
            if found not in unreadable_seen:
                skip(skipped, found, "macOS resource fork" if macos_junk(found) else why)
                unreadable_seen.add(found)
            drop("its sample cannot be read")
            continue
        if "amplitude" in raw:
            raw["volume"] = str(float(raw.get("volume", 0)) + db(float(raw["amplitude"]) / 100))
        if "loop_crossfade" in raw:
            crossfades.add(float(raw["loop_crossfade"]))
        if "loop_mode" in raw:
            mode = LOOP_MODE_ALIASES.get(raw["loop_mode"].strip().lower(), raw["loop_mode"])
            if mode not in kitlib.LOOP_MODES:
                ignored[f"loop_mode={raw['loop_mode']}"] = ignored.get(
                    f"loop_mode={raw['loop_mode']}", 0) + 1
                raw.pop("loop_mode")
            else:
                raw["loop_mode"] = mode
        try:
            region = region_of(raw, found.resolve())
        except (kitlib.Problem, ValueError) as error:
            drop(f"a value Dawsome cannot read ({error})")
            continue
        if region["loop_mode"] in ("loop_continuous", "loop_sustain", None) and "loop_end" not in region:
            points = wav_loop(found)
            if points:
                region["loop_start"], region["loop_end"] = points
                if region["loop_mode"] is None:
                    region["loop_mode"] = "loop_continuous"
            elif region["loop_mode"] is not None:
                drop("loops with no loop points in the .sfz or the file (played without)")
                region["loop_mode"] = "no_loop"
        for name in ("lokey", "hikey", "pitch_keycenter"):
            region[name] = region[name] - shift
        if region["lokey"] > region["hikey"] or region["hikey"] < 0 or region["lokey"] > 127:
            drop("an empty key range")
            continue
        region["lokey"], region["hikey"] = max(0, region["lokey"]), min(127, region["hikey"])
        # A round robin's lorand defaults to 0, so the first of a set often names hirand alone
        # (Karoryfer's Fashionbass: hirand=0.333, then lorand=0.333 hirand=0.666, then
        # lorand=0.666); left out of the group, it sounded under every other one.
        region["_rand"] = (float(raw.get("lorand", 0)) if "lorand" in raw or "hirand" in raw
                           else None)
        if float(raw.get("pitch_keytrack", 100)) == 0 and region["lokey"] != region["hikey"]:
            for key in range(region["lokey"], region["hikey"] + 1):
                kept.append({**region, "lokey": key, "hikey": key, "pitch_keycenter": key})
        else:
            if float(raw.get("pitch_keytrack", 100)) == 0:
                region["pitch_keycenter"] = region["lokey"]
            kept.append(region)
    regions = sequence_random(kept)
    notes = fit_ranges(regions)
    if sw_values:
        notes.append(f"keyswitches {', '.join(map(str, sw_values))}: kept {sw_keep}")
    return {"regions": regions, "release_regions": release, "dropped": dropped,
            "ignored": ignored, "crossfades": sorted(crossfades), "notes": notes}


LOOP_MODE_ALIASES = {"continuous": "loop_continuous", "loop": "loop_continuous",
                     "sustain": "loop_sustain", "oneshot": "one_shot", "noloop": "no_loop",
                     "no-loop": "no_loop", "one-shot": "one_shot"}


def fit_ranges(regions):
    """Values brought inside the ranges Dawsome reads (it leaves out a region with a value
    outside them), keeping what they mean: volumes above +6 dB shift the whole set down by
    the excess (the layers' levels to each other kept; the listening files are normalized
    anyway), tune beyond a semitone moves its whole semitones into transpose, pan, veltrack,
    attack and release are clamped. Returns a note per change made."""
    notes = []
    top = max((r["volume"] for r in regions), default=0.0)
    if top > 6:
        for r in regions:
            r["volume"] = max(-144.0, r["volume"] - (top - 6))
        notes.append(f"volumes shifted {-(top - 6):+.1f} dB (Dawsome reads at most +6 dB)")
    moved = clamped = 0
    for r in regions:
        if abs(r["tune"]) > 100:
            semis = int(r["tune"] / 100)
            r["tune"] -= 100 * semis
            r["transpose"] += semis
            moved += 1
        for name, low, high in (("pan", -100, 100), ("amp_veltrack", -100, 100),
                                ("ampeg_attack", 0, 100), ("ampeg_release", 0, 100)):
            if not low <= r[name] <= high:
                r[name] = min(high, max(low, r[name]))
                clamped += 1
    if moved:
        notes.append(f"{moved} tunes beyond a semitone moved into transpose")
    if clamped:
        notes.append(f"{clamped} values clamped to Dawsome's ranges")
    return notes


def sequence_random(regions):
    """lorand/hirand round robins as Dawsome's sequential ones: regions sharing a key and
    velocity range and differing by lorand take seq_length = their count and seq_position by
    lorand order. Dawsome plays them in turn, not at random."""
    groups = {}
    for region in regions:
        if region.get("_rand") is not None:
            key = (region["lokey"], region["hikey"], region["lovel"], region["hivel"])
            groups.setdefault(key, []).append(region)
    for members in groups.values():
        members.sort(key=lambda r: r["_rand"])
        for position, region in enumerate(members, 1):
            region["seq_length"], region["seq_position"] = len(members), position
    for region in regions:
        region.pop("_rand", None)
    return regions


def number(value):
    """`value` as SFZ text that reads back as the same double: an integer plainly, else the
    shortest repr."""
    value = float(value)
    return str(int(value)) if value.is_integer() else repr(value)


def write_sfz(path, regions, comment, names=None):
    """An SFZ of the subset holding `regions`, each sample by `names[sample]` (a path from the
    SFZ's folder) or as it is (the sample last on each line, so a path may hold spaces)."""
    lines = [f"// {line}" for line in comment.splitlines()] + [""]
    for r in regions:
        parts = [f"lokey={r['lokey']}", f"hikey={r['hikey']}",
                 f"pitch_keycenter={r['pitch_keycenter']}", f"lovel={r['lovel']}",
                 f"hivel={r['hivel']}"]
        if r.get("loop_mode"):
            parts.append(f"loop_mode={r['loop_mode']}")
        for name in ("loop_start", "loop_end", "offset", "end"):
            if r.get(name) is not None and (name != "offset" or r[name]):
                parts.append(f"{name}={r[name]}")
        for name, default in (("ampeg_attack", 0.0), ("ampeg_release", 0.0),
                              ("amp_veltrack", 100.0), ("volume", 0.0), ("tune", 0.0),
                              ("pan", 0.0)):
            if float(r[name]) != default:
                parts.append(f"{name}={number(r[name])}")
        if r["transpose"]:
            parts.append(f"transpose={r['transpose']}")
        if r["seq_length"] != 1:
            parts.append(f"seq_length={r['seq_length']} seq_position={r['seq_position']}")
        parts.append(f"sample={names[r['sample']] if names else r['sample']}")
        lines.append("<region> " + " ".join(parts))
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def mirror_set(regions, folder, comment):
    """`regions` as a set Dawsome can play: <folder>/set.sfz with every sample hard-linked (or
    copied, across filesystems) into <folder>/samples/ under a numbered name. Why: Dawsome
    reads a set's samples only inside the set's own folder, links resolved, so an SFZ written
    here cannot name the reference's files where they lie, and none is written beside them."""
    folder = Path(folder)
    if folder.exists():
        shutil.rmtree(folder)
    (folder / "samples").mkdir(parents=True)
    names = {}
    for region in regions:
        source = Path(region["sample"])
        if source in names:
            continue
        target = place_sound(source, folder / "samples" / f"{len(names):04d}")
        names[source] = f"samples/{target.name}"
    write_sfz(folder / "set.sfz", regions, comment, names)
    return folder / "set.sfz"


DAWSOME_DECODES = {".wav", ".flac", ".ogg", ".mp3"}


def canonical_wav(path):
    """Whether a WAV file's fmt chunk has a size its format's spec gives (16 or 18 bytes for
    PCM and IEEE float, 40 for WAVE_FORMAT_EXTENSIBLE): Dawsome's decoder refuses others
    ("malformed fmt_ieee chunk" for the vowel choir's 20-byte float fmt), libsndfile reads
    them."""
    try:
        with open(path, "rb") as file:
            head = file.read(12)
            if head[:4] != b"RIFF" or head[8:12] != b"WAVE":
                return False
            while len(header := file.read(8)) == 8:
                chunk, size = header[:4], struct.unpack("<I", header[4:])[0]
                if chunk == b"fmt ":
                    tag = struct.unpack("<H", file.read(2))[0]
                    return (tag in (1, 3) and size in (16, 18)) or (tag == 0xFFFE and size == 40)
                file.seek(size + (size & 1), os.SEEK_CUR)
    except (OSError, struct.error):
        return False
    return False


def place_sound(source, stem):
    """`source` at `stem` + its extension, hard-linked (or copied across filesystems); a
    format Dawsome does not decode (AIFF, a WAV with an odd fmt chunk) written as a 32-bit
    float WAV of the same frames instead (loop points are frames, so they hold)."""
    source = Path(source)
    if source.suffix.lower() in DAWSOME_DECODES and (
            source.suffix.lower() != ".wav" or canonical_wav(source)):
        target = Path(f"{stem}{source.suffix.lower()}")
        try:
            os.link(source, target)
        except OSError:
            shutil.copyfile(source, target)
        return target
    target = Path(f"{stem}.wav")
    x, rate = sf.read(str(source), dtype="float32", always_2d=True)
    sf.write(str(target), x, rate, subtype="FLOAT")
    return target


# ------------------------------------------------------------------ packs of sound files

DYNAMIC_ORDER = ("ppp", "pp", "p", "mp", "mf", "f", "ff", "fff")
NOTE_RE = re.compile(r"(?<![A-Za-z])([A-Ga-g])(#|b|s)?(-?\d)(?!\d)")
CLASS_RE = re.compile(r"^([A-Ga-g])(#|b|s)?$")


def tokens(name):
    """The lower-case words of a file name: split at non-letters and digits, and camelCase."""
    spaced = re.sub(r"([a-z])([A-Z])", r"\1 \2", name)
    return [t for t in re.split(r"[^a-z0-9]+", spaced.lower()) if t]


def name_key(stem):
    """(pitch class, key or None) a file name gives: a note name with its octave (C4 = 60), or
    a bare note name (the whole name, or one word of it) with no octave; None without one."""
    pitch = {"c": 0, "d": 2, "e": 4, "f": 5, "g": 7, "a": 9, "b": 11}
    for word in re.split(r"[ _\-.]+", stem):
        if m := CLASS_RE.match(word):
            pc = (pitch[m[1].lower()] + {"#": 1, "s": 1, "b": -1, None: 0}[m[2]]) % 12
            return pc, None
    matches = list(NOTE_RE.finditer(stem))
    if matches:
        m = matches[0]
        pc = pitch[m[1].lower()] + {"#": 1, "s": 1, "b": -1, None: 0}[m[2]]
        key = (int(m[3]) + 1) * 12 + pc
        return pc % 12, key if 0 <= key <= 127 else None
    return None


_KEYS = {}


def detect_key_cached(path):
    """detect_key, once per file, None for a file it cannot read."""
    path = Path(path)
    if path not in _KEYS:
        try:
            _KEYS[path] = detect_key(path)
        except Exception as error:  # an unreadable file places by name alone
            print(f"compare: pitch detection skipped {path}: {type(error).__name__}: {error}",
                  file=sys.stderr)
            _KEYS[path] = None
    return _KEYS[path]


def detect_key(path):
    """(key, cents off it, voiced share) by YIN over up to 1 s of the file from where it is
    first within 6 dB of its loudest (25 Hz to 4.2 kHz)."""
    x, rate = sf.read(str(path), dtype="float64", always_2d=True)
    m = x.mean(axis=1)
    if not np.abs(m).max():
        return None
    env = moving_rms(m, 0.01 * rate)
    start = int(np.argmax(env >= env.max() * 0.5)) + int(0.05 * rate)
    f0, voiced = yin(m[start:start + rate], rate, 25.0, 4200.0)
    if f0 is None:
        return None
    exact = 69 + 12 * math.log2(f0 / 440.0)
    return int(round(exact)), 100 * (exact - round(exact)), voiced


def pack_key(path):
    """The key a pack's file plays at and how it was found: its name's note when pitch
    detection agrees within a semitone; the name's pitch class at the detected octave when
    they differ by octaves (libraries name octaves C3 = 60 or C4 = 60); detection alone with
    no note in the name; the name's note when detection fails or disagrees on the pitch class."""
    named = name_key(path.stem)
    found = detect_key_cached(path)
    detected = found[0] if found and found[2] >= 0.5 else None
    if named is None:
        return (detected, "pitch detection") if detected is not None else (None, "no pitch")
    pc, key = named
    if detected is None:
        return (key, "name (no steady pitch found)") if key is not None else (None, "no octave")
    if key is not None and abs(detected - key) <= 1:
        return key, "name, detection agrees"
    if (detected - pc) % 12 in (0, 1, 11):
        octave_key = pc + 12 * round((detected - pc) / 12)
        return octave_key, "name's note, detected octave"
    if key is not None:
        return key, f"name (detection said {note_name(detected)})"
    return detected, f"detection (name said pitch class {pc})"


def dynamic_rank(stem):
    """The dynamic a file name marks (pp .. ff, or v1, vl3, vel64), as a sortable number."""
    for word in tokens(stem):
        if word in DYNAMIC_ORDER:
            return DYNAMIC_ORDER.index(word)
        if m := re.fullmatch(r"(?:v|vl|vel|velocity|layer|l)(\d+)", word):
            return 100 + int(m[1])
    return None


def pack_regions(files, release=0.3):
    """Regions for a pitched pack: one per file at its key, each key's files split into
    velocity layers by their dynamic marks (equal shares of 1..127) and round robins (files
    with the same mark), each key's zone reaching halfway to its neighbors and the ends
    max(2, half the median gap) past the first and last; a WAV's smpl loop kept; a release
    of `release` s (an assumption: a pack carries none). Returns (regions, log)."""
    placed, log = [], []
    for path in files:
        key, how = pack_key(path)
        log.append({"file": path.name, "key": key, "how": how})
        if key is not None:
            placed.append((key, dynamic_rank(path.stem), path))
    keys = sorted({k for k, _, _ in placed})
    if not keys:
        return [], log
    gaps = [b - a for a, b in zip(keys, keys[1:])]
    edge = max(2, (sorted(gaps)[len(gaps) // 2] // 2) if gaps else 2)
    regions = []
    for i, key in enumerate(keys):
        lo = max(0, key - edge) if i == 0 else (keys[i - 1] + key) // 2 + 1
        hi = min(127, key + edge) if i == len(keys) - 1 else (key + keys[i + 1]) // 2
        members = [(rank, path) for k, rank, path in placed if k == key]
        ranks = sorted({rank for rank, _ in members}, key=lambda r: (r is None, r))
        for layer, rank in enumerate(ranks):
            lovel = 1 + (127 * layer) // len(ranks)
            hivel = (127 * (layer + 1)) // len(ranks)
            same = sorted((p for r, p in members if r == rank), key=lambda p: p.name)
            for position, path in enumerate(same, 1):
                region = dict(REGION_DEFAULTS, lokey=lo, hikey=hi, pitch_keycenter=key,
                              lovel=lovel, hivel=hivel, ampeg_release=release,
                              seq_length=len(same), seq_position=position,
                              sample=path.resolve(), loop_mode="no_loop")
                points = wav_loop(path) if path.suffix.lower() == ".wav" else None
                if points:
                    region.update(loop_mode="loop_continuous", loop_start=points[0],
                                  loop_end=points[1])
                regions.append(region)
    return regions, log


# ------------------------------------------------------------------------------ drum roles

# Drum roles from names, in two tiers. Full words first: they mean the same in every
# library. Drum-machine voice abbreviations last (BD, SD, OH, LT, after the TR-808/909's
# panels): they are what one-shot packs are named by, but two letters mean other things
# elsewhere (Salamander's "OH" is its overhead microphones, Karoryfer's "ht" its hi-hat), so
# in a kit laid out on General MIDI keys the key decides before them.
ROLE_RULES = {
    "kick": [{"kick"}, {"bassdrum"}, {"kik"}, {"kdrum"}],
    "snare": [{"snare", "center"}, {"snare"}, {"snr"}],
    "hat_closed": [{"hh", "closed"}, {"hihat", "closed"}, {"hat", "closed"}, {"chh"},
                   {"closedhat"}, {"hh"}, {"hihat"}, {"hat"}],
    "hat_open": [{"hh", "open"}, {"hat", "open"}, {"hihat", "open"}, {"ohh"}, {"openhat"}],
    "crash": [{"crash"}, {"cymbal"}, {"cym"}, {"cymb"}, {"splash"}, {"china"}, {"csh"}],
    "ride": [{"ride"}],
    "tom_hi": [{"htom"}, {"hitom"}, {"tom", "hi"}, {"tom", "high"}, {"tom1"}, {"racktom"},
               {"hightom"}],
    "tom_mid": [{"mtom"}, {"midtom"}, {"tom", "mid"}, {"tom2"}],
    "tom_lo": [{"ltom"}, {"lowtom"}, {"ftom"}, {"floortom"}, {"tom", "low"}, {"tom", "lo"},
               {"tom", "floor"}, {"tom3"}],
    "clap": [{"clap"}, {"handclap"}, {"handclp"}, {"clp"}],
    "rim": [{"rimshot"}, {"rim"}, {"crossstick"}, {"sidestick"}, {"cross", "stick"},
            {"snare", "stick"}],
    "cowbell": [{"cowbell"}, {"cow"}],
}
ROLE_ABBREVIATIONS = {
    "kick": [{"bd"}], "snare": [{"sd"}, {"sn"}], "hat_closed": [{"ch"}, {"hhc"}, {"hhcd"}],
    "hat_open": [{"oh"}, {"hho"}], "crash": [{"cy"}], "ride": [{"rd"}], "tom_hi": [{"ht"}],
    "tom_mid": [{"mt"}], "tom_lo": [{"lt"}, {"ft"}], "clap": [{"cp"}], "rim": [{"rs"}],
    "cowbell": [{"cb"}],
}
# Last of all: a tom whose name gives no size (Tom01; its number says neither high nor low)
# plays the mid tom.
ROLE_GENERIC = {"tom_mid": [{"tom"}, {"toms"}]}
ROLE_EXCLUDE = {
    "snare": {"rim", "rimshot", "cross", "crossstick", "side", "sidestick", "stick", "flam",
              "roll", "brush"},
    "hat_closed": {"open", "pedal", "foot", "ohh", "openhat", "oh", "hho", "chik", "footspl"},
    "kick": {"loop"},
}
BEAT_ROLES = ("kick", "snare", "hat_closed", "hat_open", "crash", "tom_hi", "tom_mid", "tom_lo")
# General MIDI's percussion keys by role (35/36 kick, 38/40 snare, 42 closed and 46 open hat,
# the six toms 41 to 50 low to high, 49/57 crash, 51/59 ride...), each role's keys in the
# order the beat prefers them.
GM_ROLES = {"kick": [36, 35], "snare": [38, 40], "hat_closed": [42], "hat_open": [46],
            "crash": [49, 57, 55, 52], "ride": [51, 59, 53], "tom_hi": [50, 48],
            "tom_mid": [47, 45], "tom_lo": [43, 41], "clap": [39], "rim": [37],
            "cowbell": [56]}
GM_KEY_ROLE = {key: role for role, keys in GM_ROLES.items() for key in keys}


def role_words(name):
    """A name's words for telling drums apart: its words, and each word split where letters
    and digits meet (BD0075 -> bd, OH25 -> oh, cowbell1 -> cowbell)."""
    words = set(tokens(str(name)))
    for word in list(words):
        words.update(w for w in re.split(r"(?<=[a-z])(?=\d)|(?<=\d)(?=[a-z])", word) if w)
    return words


def role_of(words, tier=0):
    """The first role whose rules of `tier` (0 full words, 1 drum-machine abbreviations,
    2 an unsized tom) a name's words meet, with the rule's rank; None when none does."""
    table = (ROLE_RULES, ROLE_ABBREVIATIONS, ROLE_GENERIC)[int(tier)]
    for role, rules in table.items():
        if words & ROLE_EXCLUDE.get(role, set()):
            continue
        for rank, rule in enumerate(rules):
            if rule <= words:
                return role, rank
    return None


def general_midi_kit(keys):
    """Whether a kit's keys look laid out on General MIDI: a kick key (35/36) and a snare key
    (38/40), and nine in ten of its keys inside GM's percussion range (35 to 81)."""
    keys = set(keys)
    return bool(keys & {35, 36}) and bool(keys & {38, 40}) and (
        sum(35 <= k <= 81 for k in keys) >= 0.9 * len(keys))


def kit_role(words, key, gm):
    """A kit key's role: its name's full words, else (a General MIDI kit) its key, else its
    name's abbreviations, else its key."""
    found = role_of(words) or (None if gm and key in GM_KEY_ROLE else
                               role_of(words, 1) or role_of(words, 2))
    if found:
        return found[0]
    return GM_KEY_ROLE.get(key)


def pick_roles(hits, gm=False):
    """{role: hit id} for the beat, from `hits` [{"id", "role", "key"}]: per role, in a General
    MIDI kit the hit on the role's preferred GM key, else the first hit of that role. Missing
    toms borrow another tom."""
    roles = {}
    for role in ROLE_RULES:
        own = [h for h in hits if h.get("role") == role]
        if not own:
            continue
        preferred = [h for key in GM_ROLES.get(role, []) for h in own if gm and h.get("key") == key]
        roles[role] = (preferred or own)[0]["id"]
    return borrow_toms(roles)


def borrow_toms(roles):
    """A missing tom plays another tom's hit (a kit of two toms still has its fill)."""
    for role, others in (("tom_mid", ("tom_hi", "tom_lo")), ("tom_hi", ("tom_mid", "tom_lo")),
                         ("tom_lo", ("tom_mid", "tom_hi"))):
        if role not in roles:
            for other in others:
                if other in roles:
                    roles[role] = roles[other]
                    break
    return roles


# ------------------------------------------------------------------------------ rendering

class Renderer:
    """Runs `dawsome render` on projects written under `work`, with HOME and every XDG and
    Dawsome folder pointed into a temporary folder that must be empty when it is closed."""

    def __init__(self, binary, work, keep=False):
        self.binary = str(binary)
        self.keep = keep
        self.work = Path(work)
        self.work.mkdir(parents=True, exist_ok=True)
        self.home = Path(tempfile.mkdtemp(prefix="home-", dir=self.work))
        self.cache = self.home / "stretch"
        names = ("XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_CACHE_HOME", "XDG_STATE_HOME",
                 "DAWSOME_DATA_HOME", "DAWSOME_CACHE_HOME")
        self.env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": str(self.home),
                    "TMPDIR": str(self.home), "LANG": "C.UTF-8"}
        self.env.update({name: str(self.home / name.lower()) for name in names})
        self.renders = 0

    def render(self, project, name, bars):
        """Writes `project` as <work>/<name>.json, renders `bars` bars to <name>.wav (32-bit
        float, removed once read unless `keep`), returns (frames x 2 array, warnings)."""
        path = self.work / f"{name}.json"
        out = self.work / f"{name}.wav"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(project, indent=1) + "\n", encoding="utf-8")
        result = subprocess.run(
            [self.binary, "render", str(path), str(out), "--bars", str(bars),
             "--cache-dir", str(self.cache)],
            env=self.env, cwd=str(self.home), capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            raise kitlib.Problem(f"dawsome render {path} failed: {result.stderr.strip()}")
        self.renders += 1
        x, rate = sf.read(str(out), dtype="float64", always_2d=True)
        if not self.keep:
            out.unlink()
        if rate != RATE:
            raise kitlib.Problem(f"{out}: {rate} Hz, not {RATE}")
        warnings = [line for line in result.stderr.splitlines() if line.strip()]
        return x, warnings

    def close(self):
        """Removes the temporary home; returns what was written into it (should be nothing)."""
        left = sorted(str(p.relative_to(self.home)) for p in self.home.rglob("*"))
        shutil.rmtree(self.home, ignore_errors=True)
        return left


def project_json(channels, notes, bars):
    """A project file: `channels` [(kind, path)], one pattern of `bars` bars holding `notes`
    [(tick, length, channel index, key, velocity byte)], at 48 kHz and 120 BPM, every channel
    and the mixer at unity with no effects (the default mixer)."""
    chans = []
    for i, (kind, path) in enumerate(channels, 1):
        chan = {"id": i, "name": Path(path).stem[:40], "kind": kind, "volume": 1.0, "pan": 0.0,
                "mute": False}
        chan["instrument" if kind == "instrument" else "sample"] = str(path)
        chans.append(chan)
    pattern = []
    for tick, length, channel, key, byte in sorted(notes, key=lambda n: (n[0], n[2], n[3])):
        note = {"channel": channel, "tick": tick, "length": length, "velocity": velocity(byte)}
        if key != 60:
            note["key"] = key
        pattern.append(note)
    return {"format_version": 4, "sample_rate": RATE, "tempo_mbpm": TEMPO_MBPM, "ppq": PPQ,
            "next_channel_id": len(chans) + 1, "channels": chans,
            "patterns": [{"id": 1, "name": "Pattern 1", "length_bars": bars, "notes": pattern}]}


def frames(ticks):
    return int(round(ticks * SECONDS_PER_TICK * RATE))


def normalise(x, path):
    """Writes `x` scaled to TARGET_LUFS as a 32-bit float WAV at `path`; returns
    {lufs_before, gain_db, peak_after_db}."""
    loud = integrated_loudness(x, RATE)
    gain_db = (TARGET_LUFS - loud) if loud is not None else 0.0
    y = x * 10 ** (gain_db / 20)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), y.astype(np.float32), RATE, subtype="FLOAT")
    return {"lufs_before": loud, "gain_db": gain_db, "peak_after_db": db(np.abs(y).max())}


# ----------------------------------------------------------------------------- the sets

def our_sets(repo):
    """Every set we ship, as {name, origin, kind, title, source, sfz | hits}."""
    sets = []
    for folder in sorted((ROOT / "sets").iterdir()):
        about_path = folder / "set.json"
        if not about_path.is_file():
            continue
        about = kitlib.read_json(about_path)
        entry = {"name": folder.name, "origin": "dawsome-kit", "title": about.get("name"),
                 "source": about.get("source"), "folder": folder}
        if about.get("kind") == "samples":
            entry.update(kind="samples", files=sorted(p for p in (folder / "samples").iterdir()
                                                      if p.suffix.lower() in AUDIO_EXTENSIONS))
        else:
            entry.update(kind="drums" if folder.name.startswith("drums") else "pitched",
                         sfz=folder / about["sfz"])
        sets.append(entry)
    for folder in sorted((repo / "assets" / "instruments").iterdir()):
        about_path = folder / "set.json"
        if about_path.is_file():
            about = kitlib.read_json(about_path)
            sets.append({"name": folder.name, "origin": "dawsome assets/instruments",
                         "title": about.get("name"), "source": about.get("source"),
                         "folder": folder, "kind": "pitched", "sfz": folder / about["sfz"]})
    kit = kitlib.read_json(repo / "assets" / "sounds" / "kit.json")
    hits = [{"id": s["name"], "name": s["name"], "file": repo / "assets" / "sounds" / s["file"],
             "source": s.get("source")} for s in kit["sounds"] if s["note"] == "drum"]
    sets.append({"name": "starter-kit", "origin": "dawsome assets/sounds", "title": "Starter kit",
                 "source": "made for Dawsome", "kind": "drum-pack",
                 "folder": repo / "assets" / "sounds", "hits": hits})
    return sets


def file_facts(paths):
    """{rates, bits, channels, files, bytes, decoded_bytes} of the sound files `paths`."""
    rates, bits, channels = set(), set(), set()
    total = decoded = 0
    for path in paths:
        try:
            info = kitlib.audio_info(path)
            rate, depth, ch, count = info.rate, info.bits, info.channels, info.frames
        except kitlib.Problem:
            meta = sf.info(str(path))
            rate, ch, count = meta.samplerate, meta.channels, meta.frames
            depth = {"PCM_16": 16, "PCM_24": 24, "PCM_32": 32, "FLOAT": 32, "DOUBLE": 64,
                     "PCM_S8": 8, "PCM_U8": 8}.get(meta.subtype, meta.subtype)
        rates.add(rate)
        bits.add(depth)
        channels.add(ch)
        total += path.stat().st_size
        decoded += count * ch * 4
    return {"rates": sorted(rates), "bits": sorted(bits, key=str), "channels": sorted(channels),
            "files": len(paths), "bytes": total, "decoded_bytes": decoded}


def inventory(regions, key, extra=None):
    """The inventory row of a set's regions, at its matched key."""
    files = sorted({r["sample"] for r in regions})
    at_key = [r for r in regions if r["lokey"] <= key <= r["hikey"]]
    layers = {(r["lovel"], r["hivel"]) for r in at_key}

    def layer_count(k):
        return len({(r["lovel"], r["hivel"]) for r in regions if r["lokey"] <= k <= r["hikey"]})

    lo, hi = key_range(regions)
    looping = [r for r in regions if loops(r)]
    row = {"keys": [lo, hi], "zones": len(regions), "velocity_layers": len(layers),
           "velocity_layers_max": max(layer_count(k) for k in range(lo, hi + 1)),
           "round_robins": max(r["seq_length"] for r in regions),
           "looping_zones": len(looping), "release_regions": 0, "loop_crossfade": None,
           "release_s": sorted({r["ampeg_release"] for r in at_key}),
           "amp_veltrack": sorted({r["amp_veltrack"] for r in at_key}),
           **file_facts([Path(p) for p in files])}
    row.update(extra or {})
    return row


def zone_measures(regions, key):
    """Tail and seam measures of the loudest file the matched note plays, and the worst seam of
    any looping zone in the set (by the spectral jump over the loop's own change)."""
    zones = zones_for(regions, key, MATCH_VELOCITY)
    out = {"zone_files": [Path(r["sample"]).name for r in zones]}
    if not zones:
        return out
    zone = max(zones, key=lambda r: r["volume"])
    out["zone_file"] = Path(zone["sample"]).name
    out["tail"] = tail_measures(zone["sample"], zone)
    speed = 2 ** ((key - zone["pitch_keycenter"] + zone["transpose"]) / 12 + zone["tune"] / 1200)
    out["plays_seconds"] = None if loops(zone) else out["tail"]["file_seconds"] / speed
    if loops(zone):
        out["seam"] = seam_measures(zone["sample"], zone)
    worst, offsets = None, {}
    for region in regions:
        if loops(region):
            seam = seam_measures(region["sample"], region)
            score = seam["spectral_jump_db"] - (seam["spectral_inside_db"] or 0)
            if worst is None or score > worst[0]:
                worst = (score, Path(region["sample"]).name, seam)
            # An end counts as off only when moving it at least halves the click.
            off = seam["end_offset"] if seam["end_offset"] and (
                seam["seam_click_best"] <= seam["seam_click"] / 2) else 0
            offsets[off] = offsets.get(off, 0) + 1
    if worst:
        out["worst_seam"] = {"file": worst[1], **worst[2]}
        out["loop_end_offsets"] = dict(sorted(offsets.items()))
    return out


def measure_pitched(renderer, entry, regions, key, tag):
    """Probe render (velocity 100, 40, 120) of a pitched set at `key`; the measures."""
    channels = [("instrument", entry["render_sfz"])]
    notes = [(t, length, 1, k, b) for t, length, k, b in probe_notes(key)]
    bars = PROBE_SECTION_BARS * 3
    x, warnings = renderer.render(project_json(channels, notes, bars), f"{tag}/probe", bars)
    section = frames(PROBE_SECTION_BARS * BAR)
    held = frames(HOLD_BARS * BAR)
    parts = [x[i * section:(i + 1) * section] for i in range(3)]
    main = section_measures(parts[0], RATE, held, key)
    soft = section_measures(parts[1], RATE, held, key)
    hard = section_measures(parts[2], RATE, held, key)
    return {"key": key, "note": note_name(key), **main, **dynamics(soft, hard),
            **zone_measures(regions, key), "warnings": warnings}


def dynamics(soft, hard):
    out = {"lufs_v40": soft.get("lufs"), "lufs_v120": hard.get("lufs"),
           "dyn_db": None, "brightness_ratio": None}
    if soft.get("lufs") is not None and hard.get("lufs") is not None:
        out["dyn_db"] = hard["lufs"] - soft["lufs"]
    if soft.get("centroid_onset") and hard.get("centroid_onset"):
        out["brightness_ratio"] = hard["centroid_onset"] / soft["centroid_onset"]
    return out


def listen_pitched(renderer, entry, regions, key, path, tag):
    channels = [("instrument", entry["render_sfz"])]
    notes = [(t, length, 1, k, b) for t, length, k, b in phrase_notes(key)]
    x, warnings = renderer.render(project_json(channels, notes, PHRASE_BARS), f"{tag}/phrase",
                                  PHRASE_BARS)
    held = notes[-1]
    heard = heard_seams(x, regions, key, held[4], frames(held[0]), frames(held[1]))
    return {"file": str(path), **normalise(x, path), "heard_seam": heard, "warnings": warnings}


def drum_hits_of_set(regions):
    """The hits of a drum set: one per key that has zones, named by the file its velocity-100
    note plays, with the role that name gives (or None)."""
    hits = []
    keys = sorted({k for r in regions for k in range(r["lokey"], r["hikey"] + 1)})
    gm = general_midi_kit(keys)
    for key in keys:
        zones = zones_for(regions, key, MATCH_VELOCITY) or [
            r for r in regions if r["lokey"] <= key <= r["hikey"]]
        if zones:
            stem = Path(zones[0]["sample"]).stem
            hits.append({"id": key, "name": f"{key} {stem}", "key": key, "gm": gm,
                         "words": role_words(stem), "role": kit_role(role_words(stem), key, gm)})
    return hits


def subset_sfz(entry, regions, keys, name):
    """A mirrored set of just the zones on `keys` (a reference kit can hold thousands of
    zones, and Dawsome loads every zone of a set for each render)."""
    zones = [r for r in regions if any(r["lokey"] <= k <= r["hikey"] for k in keys)]
    folder, comment = entry["mirror"]
    return mirror_set(zones, folder.parent / name, comment)


def hit_channel(entry, hit, regions=None, index=0):
    """(channels, key) a hit plays on: the set's instrument channel at its key (for a
    reference, a set of that key's zones alone), or a sampler channel of its own file (key 60,
    as recorded)."""
    if regions is not None and entry.get("mirror"):
        return [("instrument", subset_sfz(entry, regions, [hit["key"]],
                                          f"hits/{index:02d}"))], hit["key"]
    if regions is not None:
        return [("instrument", entry["render_sfz"])], hit["key"]
    return [("sampler", sampler_file(entry, hit["file"]))], 60


def sampler_file(entry, path):
    """A pack's file as a sampler channel can play it: as it is, or (AIFF) converted under
    the reference's work folder."""
    path = Path(path)
    if not entry.get("work") or (path.suffix.lower() in DAWSOME_DECODES and (
            path.suffix.lower() != ".wav" or canonical_wav(path))):
        return path
    folder = Path(entry["work"]) / "converted"
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / f"{abs(hash(str(path))) % 10 ** 10:010d}.wav"
    return target if target.is_file() else place_sound(path, target.with_suffix(""))


def measure_drums(renderer, entry, regions, tag, listen_dir):
    """Every hit's measures and listening pattern, and the kit's beat (a kit's: not for a
    class of one drum)."""
    wanted = entry.get("roles_wanted")
    if regions is not None:
        hits = drum_hits_of_set(regions)
        if wanted:
            hits = [h for h in hits if h["role"] in wanted]
            if not hits:
                raise kitlib.Problem(f"no {', '.join(sorted(wanted))} key found by name")
    else:
        hits = []
        for h in entry["hits"]:
            role = h.get("role") or role_of_file(h["name"])
            hits.append(dict(h, key=None, words=role_words(h["name"]), role=role))
    rows = []
    for index, hit in enumerate(hits):
        channels, key = hit_channel(entry, hit, regions, index)
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(hit["name"]))[:60]
        notes = [(t, 0, 1, key, b) for t, _, _, b in probe_notes(key, drum=True)]
        bars = DRUM_SECTION_BARS * 3
        x, warnings = renderer.render(project_json(channels, notes, bars),
                                      f"{tag}/hits/{index:02d}-{safe}-probe", bars)
        section = frames(DRUM_SECTION_BARS * BAR)
        parts = [x[i * section:(i + 1) * section] for i in range(3)]
        row = {"hit": hit["name"], "role": hit["role"], **section_measures(parts[0], RATE, section),
               **dynamics(section_measures(parts[1], RATE, section),
                          section_measures(parts[2], RATE, section))}
        if regions is not None:
            zones = zones_for(regions, key, MATCH_VELOCITY)
            if zones:
                zone = max(zones, key=lambda r: r["volume"])
                row["zone_file"] = Path(zone["sample"]).name
                row["tail"] = tail_measures(zone["sample"], zone)
        else:
            row["zone_file"] = Path(hit["file"]).name
            row["tail"] = tail_measures(hit["file"], {})
        pattern = [(t, 0, 1, key, b) for t, b in HIT_PATTERN]
        y, more = renderer.render(project_json(channels, pattern, HIT_BARS),
                                  f"{tag}/hits/{index:02d}-{safe}", HIT_BARS)
        row["listen"] = normalise(y, listen_dir / f"{index:02d} {safe}.wav")
        row["listen"]["file"] = str(listen_dir / f"{index:02d} {safe}.wav")
        row["warnings"] = warnings + more
        rows.append(row)
    roles = pick_roles(hits, bool(hits) and hits[0].get("gm", False))
    roles = {role: hit for role, hit in roles.items() if role in BEAT_ROLES}
    by_id = {h["id"]: h for h in hits}
    beat = None
    if wanted is None and ("kick" in roles or "snare" in roles):
        if regions is not None:
            keys = sorted({by_id[i]["key"] for i in roles.values()})
            sfz = (subset_sfz(entry, regions, keys, "beat") if entry.get("mirror")
                   else entry["render_sfz"])
            channels = [("instrument", sfz)]
            notes = [(t, 0, 1, by_id[roles[role]]["key"], b) for t, role, b in beat_notes()
                     if role in roles]
        else:
            # Channels in role order, so two kits' beats mix in the same order.
            used = list(dict.fromkeys(roles[r] for r in BEAT_ROLES if r in roles))
            channels = [("sampler", sampler_file(entry, by_id[i]["file"])) for i in used]
            notes = [(t, 0, used.index(roles[role]) + 1, 60, b) for t, role, b in beat_notes()
                     if role in roles]
        x, warnings = renderer.render(project_json(channels, notes, BEAT_BARS), f"{tag}/beat",
                                      BEAT_BARS)
        beat = {"roles": {role: by_id[i]["name"] for role, i in roles.items()},
                **normalise(x, listen_dir / "beat.wav"), "file": str(listen_dir / "beat.wav"),
                "warnings": warnings}
    return rows, beat


def measure_samples_set(entry):
    """The one-shots folder: each file measured as it is (a sampler plays it as recorded)."""
    rows = []
    for path in entry["files"]:
        x, rate = sf.read(str(path), dtype="float64", always_2d=True)
        if x.shape[1] == 1:
            x = np.repeat(x, 2, axis=1)
        m = x.mean(axis=1)
        corr, side_db = width(x)
        first = spectrum_at(m, rate, onset(m, float(np.abs(x).max()) or 1.0))
        rows.append({"file": path.name, "seconds": len(x) / rate, "rate": rate,
                     "lufs": integrated_loudness(x, rate), "peak_db": db(np.abs(x).max()),
                     "centroid_onset": first and first[0], "lr_correlation": corr,
                     "side_db": side_db, "tail": tail_measures(path, {})})
    return rows


def run_set(renderer, entry, listen_root, prefer_root=None):
    """Inventory, measures and listening files of one set (ours or a reference)."""
    tag = entry["tag"]
    result = {k: entry.get(k) for k in ("name", "origin", "kind", "title", "source", "class",
                                        "library", "licence", "format")}
    result.update({k: entry[k] for k in ("flatten", "pack_log", "pair", "pair_hits", "status",
                                         "index_tag", "skipped") if k in entry})
    if entry["kind"] == "samples":
        result["inventory"] = file_facts(entry["files"])
        result["files"] = measure_samples_set(entry)
        return result
    if entry["kind"] == "drum-pack":
        files = [Path(h["file"]) for h in entry["hits"]]
        result["inventory"] = {**file_facts(files), "hits": len(files)}
        result["hits"], result["beat"] = measure_drums(renderer, entry, None, tag,
                                                       listen_root / entry["listen"])
        return result
    regions = entry["regions"]
    if not regions:
        result["problem"] = "no playable regions"
        return result
    extra = entry.get("inventory_extra", {})
    if entry["kind"] == "drums":
        key = min(r["lokey"] for r in regions)
        result["inventory"] = inventory(regions, key, extra)
        result["inventory"]["hits"] = len(drum_hits_of_set(regions))
        result["hits"], result["beat"] = measure_drums(renderer, entry, regions, tag,
                                                       listen_root / entry["listen"])
        return result
    key = root_key(regions, prefer_root)
    # The measured note matches the paired set's whenever this set plays it, even where the
    # whole phrase does not fit (then the phrase alone moves).
    probe = prefer_root if prefer_root is not None and zones_for(regions, prefer_root,
                                                                 MATCH_VELOCITY) else key
    if entry.get("mirror"):
        # Only the zones the phrase and the probes can sound: Dawsome loads a whole set.
        low, high = min(key, probe), max(key, probe) + PHRASE_SPAN
        played = [r for r in regions if r["hikey"] >= low and r["lokey"] <= high]
        entry["render_sfz"] = mirror_set(played, *entry["mirror"])
    result["inventory"] = inventory(regions, probe, extra)
    result["measures"] = measure_pitched(renderer, entry, regions, probe, tag)
    result["measures"]["phrase_root"] = key
    result["listen"] = listen_pitched(renderer, entry, regions, key,
                                      listen_root / f"{entry['listen']}.wav", tag)
    return result


# ------------------------------------------------------------------------------ references

def read_index(path):
    """The entries of a reference index: a JSON list, or the first list an object holds."""
    data = kitlib.read_json(path)
    if isinstance(data, dict):
        data = next((v for v in data.values() if isinstance(v, list)), [])
    entries = [e for e in data if isinstance(e, dict)]
    base = Path(path).resolve().parent
    for entry in entries:
        entry["folder_text"] = entry.get("folder")
        if entry.get("folder"):
            folder = Path(entry["folder"])
            entry["folder"] = folder if folder.is_absolute() else base / folder
    return entries


def ours_by_class(items, sets):
    """{class: {"set", "sets", "hits"}} from the index's own rows for what Dawsome ships
    (status "ours"): each names our set folders, or starter-kit files, " ; " between them
    (a bare name after the first is beside it)."""
    folders = {Path(s["folder"]).resolve(): s["name"] for s in sets
               if s.get("folder") and s["kind"] not in ("drum-pack", "samples")}
    starter = next((s for s in sets if s["kind"] == "drum-pack"), {"hits": []})
    hit_files = {Path(h["file"]).resolve(): h["name"] for h in starter["hits"]}
    out = {}
    for item in items:
        if item.get("status") != "ours" or not item.get("folder_text"):
            continue
        found_sets, found_hits, base = [], [], None
        for part in (x.strip() for x in str(item["folder_text"]).split(";")):
            path = Path(part)
            if not path.is_absolute() and base is not None:
                path = base / path
            base = path.parent
            resolved = path.resolve()
            if resolved in folders:
                found_sets.append(folders[resolved])
            elif resolved in hit_files:
                found_hits.append(hit_files[resolved])
        out[str(item.get("class"))] = {"set": found_sets[0] if found_sets else None,
                                       "sets": found_sets, "hits": found_hits or None}
    return out


def is_drum_class(entry, pair):
    text = " ".join(str(entry.get(k, "")) for k in ("class", "tag")).lower()
    return bool(re.search(r"drum|kit\b|percussion|perc\b", text)) or (
        pair is not None and pair.startswith("drums"))


def pair_of(cls, ours, pairs):
    """Our set a reference class is compared with: --pair, the same name, or the one set whose
    name holds the class's words (or theirs the set's); None when that is not one set."""
    if cls in pairs:
        return pairs[cls]
    names = [s["name"] for s in ours if s["kind"] != "samples"]
    if cls in names:
        return cls
    words = set(tokens(cls))
    found = [n for n in names if words and (words <= set(tokens(n)) or set(tokens(n)) <= words)]
    return found[0] if len(found) == 1 else None


def macos_junk(path):
    """A macOS resource fork or metadata file an archive made on a Mac carries: anything under
    __MACOSX, or named ._*; not a sound, whatever its extension says."""
    return "__MACOSX" in Path(path).parts or Path(path).name.startswith("._")


_UNREADABLE = {}


def unreadable(path):
    """Why soundfile cannot open `path`, or None when it can (remembered per file, since
    libraries share folders)."""
    path = Path(path)
    if path not in _UNREADABLE:
        try:
            info = sf.info(str(path))
            _UNREADABLE[path] = None if info.frames > 0 else "no frames"
        except Exception as error:  # libsndfile raises its own error types
            _UNREADABLE[path] = f"{type(error).__name__}: {error}".splitlines()[0][:200]
    return _UNREADABLE[path]


def skip(skipped, path, why):
    """Records and logs (one line) a file left out of a library."""
    print(f"compare: skipped {path}: {why}", file=sys.stderr)
    if skipped is not None:
        skipped.append({"file": str(path), "why": why})


def choose_sfz(folder, entry):
    """The .sfz of a library: the one the entry names (`sfz`), else among its programs (the
    .sfz files no other one includes) the one whose samples are there, nearest the library's
    name, then covering the most keys (a full kit over its snare alone), then with the most
    playable regions; with the others listed."""
    found = sorted(p for p in Path(folder).rglob("*.sfz")
                   if inside(p, folder) and not macos_junk(p) and p.is_file())
    if entry.get("sfz"):
        named = inside(Path(folder) / entry["sfz"], folder)
        return named, [str(p.relative_to(folder)) for p in found]
    if len(found) <= 1:
        return (found[0] if found else None), []
    included = included_files(found)
    programs = [p for p in found if p.resolve() not in included] or found
    want = set(tokens(str(entry.get("library", "")))) | set(tokens(str(entry.get("class", ""))))

    def score(path):
        keys, playable = playable_regions(path, folder)
        return (playable > 0, len(set(tokens(path.stem)) & want), keys, playable)

    best = max(programs, key=score)
    return best, [str(p.relative_to(folder)) for p in programs if p != best]


def audio_files(folder, skipped=None):
    """The sound files of a library, sorted: macOS forks, links leaving the folder and files
    soundfile cannot open are left out, each with a line in the log and in `skipped`."""
    out = []
    for path in sorted(Path(folder).rglob("*")):
        if path.suffix.lower() not in AUDIO_EXTENSIONS or not path.is_file():
            continue
        if macos_junk(path):
            skip(skipped, path, "macOS resource fork")
        elif inside(path, folder) is None:
            skip(skipped, path, "a link out of the library")
        elif (why := unreadable(path)) is not None:
            skip(skipped, path, why)
        else:
            out.append(path)
    return out


LOOP_FOLDER_WORDS = {"loop", "loops", "basslines", "lines", "phrases", "grooves", "riffs"}
RELEASE_FOLDER_WORDS = {"rel", "release", "releases"}


def tempo_loop(relative):
    """Whether a file is a tempo-labeled loop or phrase, not a note: a tempo (120bpm, (130BPM))
    anywhere in its path, or a folder named for loops, lines or riffs. A file named "...-loop"
    in a folder of notes is a looped note and stays."""
    relative = Path(relative)
    if any(re.search(r"\d+\s*bpm|bpm\s*\d+", part, re.I) for part in relative.parts):
        return True
    return any(set(tokens(part)) & LOOP_FOLDER_WORDS for part in relative.parts[:-1])


def instrument_groups(files, folder):
    """A pack's files by folder, each folder one instrument (packs ship several synths' or
    articulations' multisamples side by side)."""
    groups = {}
    for path in files:
        groups.setdefault(path.relative_to(folder).parent, []).append(path)
    return groups


def pitched_pack_files(files, folder, log):
    """The files of a pitched pack to play: tempo loops and release folders left out, the
    folder holding the most named notes from C2 to C5 (then the most notes, then files)
    when there are several, and above 200 files at most one file per semitone: the one
    whose detected pitch lies nearest the key its name (or, unnamed, its pitch) gives."""
    notes = [p for p in files if not tempo_loop(p.relative_to(folder))]
    if len(notes) < len(files):
        log.append(f"{len(files) - len(notes)} tempo-labeled loops left out")
    releases = [p for p in notes if any(set(tokens(part)) & RELEASE_FOLDER_WORDS
                                        for part in p.relative_to(folder).parts[:-1])]
    if releases:
        log.append(f"{len(releases)} release samples left out (Dawsome plays no release trigger)")
        notes = [p for p in notes if p not in releases]
    if not notes:
        raise kitlib.Problem("only tempo-labeled loops: no notes to play as an instrument")
    groups = instrument_groups(notes, folder)
    if len(groups) > 1:
        def score(item):
            keys = {k for p in item[1] if (n := name_key(p.stem)) and (k := n[1]) is not None}
            return (len({k for k in keys if 36 <= k <= 72}), len(keys), len(item[1]))

        best, notes = max(groups.items(), key=score)
        log.append(f"instrument folder {best} of {len(groups)} ({len(notes)} files)")
    if len(notes) > 200:
        notes = one_per_semitone(notes, log)
    return notes


def one_per_semitone(files, log, candidates=3, detect_cap=200):
    """At most one file per key: for each key the name gives (up to `candidates` files each),
    or, for files whose name gives none, up to `detect_cap` spread across them, the file
    whose detected pitch lies nearest the key."""
    by_key, unnamed = {}, []
    for path in files:
        named = name_key(path.stem)
        if named and named[1] is not None:
            by_key.setdefault(named[1], []).append(path)
        else:
            unnamed.append(path)
    if len(unnamed) > detect_cap:
        step = len(unnamed) / detect_cap
        unnamed = [unnamed[int(i * step)] for i in range(detect_cap)]
    best = {}
    for path in [p for group in by_key.values() for p in group[:candidates]] + unnamed:
        key, _how = pack_key(path)
        found = detect_key_cached(path)
        cents = abs(found[1]) if found and found[0] == key else 100.0
        if key is not None and (key not in best or cents < best[key][0]):
            best[key] = (cents, path)
    chosen = [best[k][1] for k in sorted(best)]
    log.append(f"{len(files)} files: one per semitone kept ({len(chosen)})")
    return chosen


CLASS_ROLES = {"kick-808": {"kick"}, "kick-909": {"kick"}, "snare": {"snare"},
               "clap": {"clap"}, "hihats": {"hat_closed", "hat_open"},
               "toms": {"tom_hi", "tom_mid", "tom_lo"}, "rimshot": {"rim"},
               "cowbell": {"cowbell"}, "crash": {"crash"}}
PACK_HITS_PER_ROLE, PACK_HITS_PER_CLASS = 3, 24


def role_of_file(relative, library=""):
    """A drum file's role: from the full words of its name, else of its folders' names, else
    of the library's name, then from their drum-machine abbreviations in that order (a pack
    of one drum machine voice: BD0075.WAV in tr808-bd)."""
    relative = Path(relative)
    places = (role_words(relative.stem),
              set().union(*(role_words(p) for p in relative.parts[:-1]))
              if relative.parts[:-1] else set(), role_words(library))
    for tier in (0, 1, 2):
        for words in places:
            if words and (found := role_of(words, tier)):
                return found[0]
    return None


def spread(items, count):
    """`count` items evenly across `items` (all when there are no more)."""
    if len(items) <= count:
        return list(items)
    step = len(items) / count
    return [items[int(i * step)] for i in range(count)]


def drum_pack_hits(files, folder, cls, library, log):
    """The hits of a drum pack to render: its one-shots (tempo loops left out) whose role the
    class wants (a kit class: every role), up to PACK_HITS_PER_ROLE per role for a kit and
    PACK_HITS_PER_CLASS for a one-drum class, spread across the pack. Why by role and not one
    per semitone: a drum has no note, and a 4,000-file pack is mostly other drums."""
    shots = [p for p in files if not tempo_loop(p.relative_to(folder))]
    if len(shots) < len(files):
        log.append(f"{len(files) - len(shots)} tempo-labeled loops left out")
    wanted = CLASS_ROLES.get(cls)
    by_role = {}
    for path in shots:
        role = role_of_file(path.relative_to(folder), library)
        if role and (wanted is None or role in wanted):
            by_role.setdefault(role, []).append(path)
    if wanted is None:
        chosen = [p for role in ROLE_RULES for p in spread(by_role.get(role, []),
                                                           PACK_HITS_PER_ROLE)]
    else:
        chosen = spread([p for role in ROLE_RULES for p in by_role.get(role, [])],
                        PACK_HITS_PER_CLASS)
    left = len(shots) - len(chosen)
    if left:
        log.append(f"{len(chosen)} of {len(shots)} one-shots rendered (by role: "
                   + ", ".join(f"{r} {len(v)}" for r, v in by_role.items()) + ")")
    if not chosen:
        raise kitlib.Problem(f"no {', '.join(sorted(wanted)) if wanted else 'drum'} one-shots "
                             f"found by name in {len(shots)} files")
    return [{"id": str(p.relative_to(folder)), "name": str(p.relative_to(folder).with_suffix("")),
             "file": p, "role": role_of_file(p.relative_to(folder), library)} for p in chosen]


def prepare_ref(entry, ours, pairs, work, ours_by_class=None, skipped=None):
    """A reference index entry as a set to run: its regions (or hits), the SFZ Dawsome renders
    (generated under `work`), and the log of what reading it left out (`skipped` lists every
    file left out). Raises kitlib.Problem when there is nothing to play."""
    folder = Path(entry["folder"])
    cls, library = str(entry.get("class", "unknown")), str(entry.get("library", folder.name))
    safe_cls = re.sub(r"[^A-Za-z0-9_.-]+", "_", cls)
    safe_lib = re.sub(r"[^A-Za-z0-9_.-]+", "_", library)
    mine = (ours_by_class or {}).get(cls, {})
    pair = mine.get("set") or pair_of(cls, ours, pairs)
    ref = {"name": f"{cls}/{library}", "origin": "reference", "class": cls, "library": library,
           "licence": entry.get("licence"), "format": entry.get("format"),
           "status": entry.get("status"), "index_tag": entry.get("tag"),
           "source": f"{library} ({entry.get('licence')})", "pair": pair,
           "pair_hits": mine.get("hits"), "roles_wanted": CLASS_ROLES.get(cls),
           "tag": f"refs/{safe_cls}/{safe_lib}", "listen": f"{safe_cls}/{safe_lib}",
           "skipped": [] if skipped is None else skipped, "pack_log": [],
           "work": Path(work) / f"refs/{safe_cls}/{safe_lib}"}
    if not folder.is_dir():
        raise kitlib.Problem(f"{folder} is not a folder")
    drum = is_drum_class(entry, pair) or cls in CLASS_ROLES
    fmt = str(entry.get("format", "")).lower()
    sfz_path, others = (None, []) if re.search(r"\bno sfz\b", fmt) else choose_sfz(folder, entry)
    if sfz_path is not None:
        flat = flatten_sfz(sfz_path, folder, ref["skipped"])
        ref.update(kind="drums" if drum else "pitched", regions=flat["regions"],
                   mirror=(Path(work) / ref["tag"] / "set",
                           f"flattened by compare.py from {sfz_path}\n"
                           "reference data, read as data; not shipped"), flatten={
                       "sfz": str(sfz_path), "other_sfz": others,
                       "release_regions": flat["release_regions"], "dropped": flat["dropped"],
                       "ignored": flat["ignored"], "notes": flat["notes"]},
                   inventory_extra={"release_regions": flat["release_regions"],
                                    "loop_crossfade": flat["crossfades"] or None})
        if not flat["regions"]:
            raise kitlib.Problem(f"{sfz_path}: no region Dawsome can play")
        return ref
    files = audio_files(folder, ref["skipped"])
    if not files:
        raise kitlib.Problem(f"no readable sound files in {folder}")
    if drum:
        ref.update(kind="drum-pack", hits=drum_pack_hits(files, folder, cls, library,
                                                        ref["pack_log"]))
        return ref
    notes = pitched_pack_files(files, folder, ref["pack_log"])
    regions, log = pack_regions(notes)
    if not regions:
        raise kitlib.Problem(f"none of {len(notes)} files could be placed on a key")
    ref.update(kind="pitched", regions=regions, pack_log=ref["pack_log"] + log,
               mirror=(Path(work) / ref["tag"] / "set",
                       f"made by compare.py from the sound files in {folder}\n"
                       "reference data, read as data; not shipped"))
    return ref


# ------------------------------------------------------------------------------ the report

def fmt(value, digits=1, unit=""):
    if value is None:
        return "–"
    if isinstance(value, float):
        if math.isinf(value):
            return "–"
        return f"{value:.{digits}f}{unit}"
    return f"{value}{unit}"


def hz(value):
    if value is None:
        return "–"
    return f"{value / 1000:.2f}k" if value >= 1000 else f"{value:.0f}"


def size_mb(value):
    return f"{value / 1e6:.1f}"


def pitch_text(m):
    if m.get("pitch_cents") is None:
        return "–"
    octaves = m.get("pitch_octaves") or 0
    return (f"{octaves:+d} oct " if octaves else "") + f"{m['pitch_cents']:+.1f}"


def width_text(m):
    if m.get("side_db") is None:
        return "mono"
    return f"{m['lr_correlation']:.2f} / {m['side_db']:.0f}"


def table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("-" for _ in headers) + "|"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def inventory_rows(results):
    rows = []
    for r in results:
        inv = r.get("inventory")
        if not inv:
            rows.append([f"`{r['name']}`", r.get("problem", "–")] + ["–"] * 9)
            continue
        keys = inv.get("keys")
        loops_text = "–"
        if "looping_zones" in inv:
            loops_text = f"{inv['looping_zones']}/{inv['zones']}"
            if inv.get("loop_crossfade"):
                loops_text += f", xfade {inv['loop_crossfade']} s"
            elif inv["looping_zones"]:
                loops_text += ", no xfade declared"
        rel = inv.get("release_s")
        rows.append([
            f"`{r['name']}`", (r.get("source") or "–")[:70],
            f"{note_name(keys[0])}–{note_name(keys[1])} ({keys[0]}–{keys[1]})" if keys else
            f"{inv['hits']} hits" if "hits" in inv else f"{inv['files']} files",
            f"{inv.get('velocity_layers', '–')} (max {inv.get('velocity_layers_max', '–')})"
            if "velocity_layers" in inv else "–",
            inv.get("round_robins", "–"), loops_text,
            f"{inv.get('release_regions', 0)}; ampeg_release {', '.join(f'{v:g}' for v in rel)} s"
            if rel is not None else "–",
            "/".join(f"{v / 1000:g}k" for v in inv["rates"]) + " · "
            + "/".join(str(b) for b in inv["bits"]) + " bit",
            "/".join(str(c) for c in inv["channels"]),
            f"{inv['files']} · {size_mb(inv['bytes'])} MB ({size_mb(inv['decoded_bytes'])} dec.)",
        ])
    return table(["Set", "Source", "Keys", "Vel. layers at key (max)", "RR", "Loops",
                  "Release samples; release", "Rate · bits", "Ch", "Files · size"], rows)


def pitched_rows(results):
    rows = []
    for r in results:
        m = r.get("measures")
        if not m:
            continue
        tail = m.get("tail", {})
        rows.append([
            f"`{r['name']}`", f"{m['note']} ({m['key']})", fmt(m.get("lufs")),
            fmt(m.get("peak_db")), f"{hz(m.get('centroid_onset'))} / {hz(m.get('rolloff_onset'))}",
            f"{hz(m.get('centroid_500ms'))} / {hz(m.get('rolloff_500ms'))}",
            fmt(m.get("attack_ms"), 0) + (" (rising)" if m.get("attack_rising") else ""),
            fmt(m.get("decay20_ms"), 0) if m.get("decay20_ms") is not None else "sustains",
            fmt(tail.get("floor_db")) + (f" ({tail['floor_flatness']:.2f})"
                                          if tail.get("floor_flatness") is not None else ""),
            fmt(tail.get("end_db")), width_text(m), pitch_text(m),
            fmt(m.get("dyn_db")), fmt(m.get("brightness_ratio"), 2),
            "loops" if m.get("seam") else fmt(m.get("plays_seconds")),
        ])
    return table(["Set", "Note", "LUFS", "Peak dBFS", "Centroid / rolloff, onset",
                  "Centroid / rolloff, +500 ms", "Attack ms", "Decay −20 dB ms",
                  "Tail floor dB (flatness)", "Last 20 ms dB", "Width corr / S−M dB",
                  "Pitch ¢", "v40→v120 dB", "Bright v120/v40", "Plays s"], rows)


def seam_row(name, file, seam):
    offset = seam.get("end_offset")
    return [name, file, f"{seam['loop_start_s']:.2f}", f"{seam['loop_seconds']:.2f}",
            fmt(seam["level_jump_db"], 2), fmt(seam["spectral_jump_db"]),
            fmt(seam.get("spectral_inside_db")), fmt(seam["seam_click"]),
            fmt(seam.get("click_inside")),
            f"{offset:+d} ({fmt(seam.get('seam_click_best'))}×)" if offset else "0",
            fmt(seam.get("match"), 2)]


def heard_text(heard):
    if not heard:
        return ["–"] * 4
    if not heard.get("wraps"):
        return [f"0 (first at {heard['first_wrap_s']:.1f} s)", "–", "–", "–"]
    return [f"{heard['wraps']} every {heard['period_s']:.2f} s",
            f"{fmt(heard['click'])} (max {fmt(heard['click_max'])}) / {fmt(heard.get('click_inside'))}",
            f"{fmt(heard['spectral_jump_db'])} / {fmt(heard.get('spectral_inside_db'))}",
            fmt(heard["level_jump_db"], 2)]


def loop_rows(results):
    rows = []
    for r in results:
        m = r.get("measures") or {}
        heard = (r.get("listen") or {}).get("heard_seam")
        if m.get("seam"):
            rows.append(seam_row(f"`{r['name']}` at {m['note']}", m["zone_file"], m["seam"])
                        + heard_text(heard))
        elif heard:
            rows.append([f"`{r['name']}` at {m.get('note')}", heard["zone_file"]] + ["–"] * 9
                        + heard_text(heard))
        if m.get("worst_seam"):
            w = m["worst_seam"]
            offsets = m.get("loop_end_offsets") or {}
            tally = ", ".join(f"{k:+d}: {v}" if k else f"right: {v}" for k, v in offsets.items())
            rows.append(seam_row(f"`{r['name']}` worst", w["file"], w)
                        + [f"all {sum(offsets.values())} loops' ends: {tally}", "–", "–", "–"])
    return table(["Set", "File", "Loop from s", "Loop s", "Level jump dB",
                  "Spectral jump dB", "…inside loop dB", "Seam click ×", "…inside ×",
                  "Best end (click)", "Match", "Heard: wraps in the 12 s hold",
                  "Heard click × / inside", "Heard spectral dB / inside",
                  "Heard level jump dB"], rows)


def drum_rows(results):
    rows = []
    for r in results:
        for h in r.get("hits") or []:
            tail = h.get("tail", {})
            rows.append([
                f"`{r['name']}`", h["hit"], fmt(h.get("lufs")), fmt(h.get("peak_db")),
                f"{hz(h.get('centroid_onset'))} / {hz(h.get('rolloff_onset'))}",
                f"{hz(h.get('centroid_500ms'))} / {hz(h.get('rolloff_500ms'))}",
                fmt(h.get("attack_ms"), 1),
                fmt(h.get("decay20_ms"), 0) if h.get("decay20_ms") is not None else "> 4000",
                fmt(tail.get("floor_db")) + (f" ({tail['floor_flatness']:.2f})"
                                              if tail.get("floor_flatness") is not None else ""),
                fmt(tail.get("end_db")), width_text(h), fmt(h.get("dyn_db")),
                fmt(h.get("brightness_ratio"), 2)])
    return table(["Kit", "Hit", "LUFS", "Peak dBFS", "Centroid / rolloff, onset",
                  "Centroid / rolloff, +500 ms", "Attack ms", "Decay −20 dB ms",
                  "Tail floor dB (flatness)", "Last 20 ms dB", "Width corr / S−M dB",
                  "v40→v120 dB", "Bright v120/v40"], rows)


def samples_rows(results):
    rows = []
    for r in results:
        for f in r.get("files") or []:
            tail = f.get("tail", {})
            rows.append([f"`{r['name']}`", f["file"], f"{f['seconds']:.1f}", fmt(f["lufs"]),
                         fmt(f["peak_db"]), hz(f.get("centroid_onset")), width_text(f),
                         fmt(tail.get("floor_db")), fmt(tail.get("end_db"))])
    return table(["Set", "File", "s", "LUFS", "Peak dBFS", "Centroid, onset",
                  "Width corr / S−M dB", "Tail floor dB", "Last 20 ms dB"], rows)


def listen_rows(results, root):
    rows = []
    for r in results:
        items = []
        if r.get("listen"):
            items.append(("phrase", r["listen"]))
        for h in r.get("hits") or []:
            items.append((h["hit"], h["listen"]))
        if r.get("beat"):
            items.append(("beat", r["beat"]))
        for what, item in items:
            peak = item["peak_after_db"]
            rows.append([f"`{r['name']}`", what, f"`{Path(item['file']).relative_to(root)}`",
                         fmt(item["lufs_before"]), fmt(item["gain_db"]), fmt(peak)
                         + (" **over 0 dBFS**" if peak > 0 else "")])
    return table(["Set", "What", "File (under the listening folder)", "LUFS rendered",
                  "Gain dB", "Peak after dBFS"], rows)


def comparison_rows(ours, refs):
    """Each reference beside the set it is paired with: the main measures, ours then theirs."""
    by_name = {r["name"]: r for r in ours}
    rows = []
    for ref in refs:
        if ref.get("hits") is not None or ref.get("kind") in ("missing", "failed"):
            continue
        mine = by_name.get(ref.get("pair"))
        for who, r in (("ours", mine), ("ref", ref)):
            if r is None:
                rows.append([f"`{ref['name']}`", who, "no paired set"] + ["–"] * 9)
                continue
            m = r.get("measures") or {}
            inv = r.get("inventory") or {}
            tail = m.get("tail", {})
            seam = m.get("seam") or {}
            rows.append([f"`{ref['name']}`", f"{who} `{r['name']}`",
                         f"{m['note']}" if m.get("note") else "–",
                         f"{inv.get('velocity_layers', '–')}/{inv.get('round_robins', '–')}",
                         f"{hz(m.get('centroid_onset'))} → {hz(m.get('centroid_500ms'))}",
                         fmt(m.get("attack_ms"), 0),
                         fmt(m.get("decay20_ms"), 0) if m.get("decay20_ms") else "sustains",
                         fmt(tail.get("floor_db")), width_text(m) if m else "–",
                         fmt(m.get("dyn_db")), fmt(m.get("brightness_ratio"), 2),
                         fmt(seam.get("spectral_jump_db"))])
    return table(["Reference", "Who", "Note", "Layers/RR", "Centroid onset → +500 ms",
                  "Attack ms", "Decay −20 dB ms", "Tail floor dB", "Width",
                  "v40→v120 dB", "Bright", "Seam spectral jump dB"], rows)


def drum_comparison_rows(ours, refs):
    """Each drum reference beside its paired kit, role by role (the hits the beats used)."""
    by_name = {r["name"]: r for r in ours}
    rows = []
    for ref in refs:
        if not ref.get("hits"):
            continue
        mine = by_name.get(ref.get("pair"))
        for role in ROLE_RULES:
            for who, r in (("ours", mine), ("ref", ref)):
                if r is None:
                    continue
                hit_name = ((r.get("beat") or {}).get("roles") or {}).get(role)
                h = next((h for h in r.get("hits") or [] if h["hit"] == hit_name), None)
                if h is None:
                    rows.append([f"`{ref['name']}`", role, f"{who}: none"] + ["–"] * 8)
                    continue
                tail = h.get("tail", {})
                rows.append([f"`{ref['name']}`", role, f"{who} `{r['name']}`: {h['hit']}",
                             fmt(h.get("lufs")), fmt(h.get("peak_db")),
                             f"{hz(h.get('centroid_onset'))} / {hz(h.get('rolloff_onset'))}",
                             fmt(h.get("attack_ms"), 1),
                             fmt(h.get("decay20_ms"), 0) if h.get("decay20_ms") is not None
                             else "> 4000", fmt(tail.get("floor_db")), width_text(h),
                             f"{fmt(h.get('dyn_db'))} / {fmt(h.get('brightness_ratio'), 2)}"])
    return table(["Reference", "Role", "Who: hit", "LUFS", "Peak dBFS",
                  "Centroid / rolloff, onset", "Attack ms", "Decay −20 dB ms",
                  "Tail floor dB", "Width", "v40→v120 dB / bright"], rows)


def flatten_notes(refs):
    lines = []
    for ref in refs:
        if ref.get("problem"):
            lines.append(f"- `{ref['name']}`: {ref['problem']}")
        f = ref.get("flatten")
        if f:
            dropped = ", ".join(f"{k}: {v}" for k, v in sorted(f["dropped"].items())) or "none"
            ignored = ", ".join(f"{k} ×{v}" for k, v in sorted(f["ignored"].items())) or "none"
            lines.append(f"- `{ref['name']}` from `{f['sfz']}`: {f['release_regions']} release "
                         f"regions left out; dropped: {dropped}; opcodes Dawsome ignores: "
                         f"{ignored}" + (f"; {'; '.join(f['notes'])}" if f["notes"] else "")
                         + (f"; other .sfz files: {', '.join(f['other_sfz'][:6])}"
                            if f["other_sfz"] else ""))
        if ref.get("pack_log"):
            log = ref["pack_log"]
            placed = [e for e in log if isinstance(e, dict)]
            how = {}
            for e in placed:
                how[e["how"]] = how.get(e["how"], 0) + 1
            unplaced = [e["file"] for e in placed if e["key"] is None]
            lines.append(f"- `{ref['name']}`: "
                         + (f"pack of {len(placed)} files placed; " if placed else "pack; ")
                         + ", ".join(f"{k}: {v}" for k, v in how.items())
                         + (f"; not placed: {', '.join(unplaced[:8])}" if unplaced else "")
                         + "".join(f"; {e}" for e in log if isinstance(e, str)))
        if ref.get("skipped"):
            first = ", ".join(Path(e["file"]).name + f" ({e['why']})" for e in ref["skipped"][:2])
            lines.append(f"- `{ref['name']}`: {len(ref['skipped'])} files skipped, e.g. {first}")
    return "\n".join(lines)


def warnings_text(results):
    lines = []
    for r in results:
        seen = []
        for item in [r.get("measures"), r.get("listen"), r.get("beat")] + (r.get("hits") or []):
            for w in (item or {}).get("warnings", []):
                if w not in seen:
                    seen.append(w)
        if seen:
            lines.append(f"- `{r['name']}`: " + " / ".join(seen[:4]))
    return "\n".join(lines) or "- none"


METHOD = """\
## How it was measured

- **One sampler.** Every number below the inventory and every listening file comes from
  `dawsome render` (the CLI's offline render, the function live playback uses) at 48 kHz,
  120 BPM, the channel and the mixer at unity with no effects (no limiter): what Dawsome
  plays, not what the files hold. Sample rate conversion, velocity curve and envelopes are
  Dawsome's.
- **Matched note and velocity.** A pitched set is measured at its class's note (C4 = 60;
  `LISTEN.md` lists them: the basses at C2, the 808 at C1, the piano, leads, plucks and
  electric pianos at C4, the rest at C3), ours at the note of the class they are named for,
  each reference at its class's; the phrase starts there when it (root to root + 12) fits
  the set's keys, else at the nearest C that fits, else at the nearest key that does, and the
  measured note stays the class's whenever the set plays it. Each
  probe note is velocity byte 100 held 4 s, then 4 s of release; velocity 40 and 120 probes
  give the dynamics columns. A drum hit is a step hit at velocity 100 (40 and 120 likewise),
  4 s each. Velocity bytes are written as byte/128, which Dawsome reads back exactly.
  Instrument sets follow the SFZ `amp_veltrack` curve; the starter kit plays on sampler
  channels, where velocity is a linear gain (as in the app's step sequencer).
- **Loudness**: ITU-R BS.1770-4 integrated (K-weighted, 400 ms blocks, -70 LUFS and -10 LU
  gates), checked against EBU Tech 3341's test cases and the standard's 48 kHz filter
  coefficients (`tests/test_compare.py`). **Peak**: sample peak of the note's section.
- **Spectrum**: centroid and 85 % rolloff of a 2048-sample Hann window at the onset (first
  sample within 40 dB of the peak) and 500 ms later (– when that window is 60 dB down).
- **Attack**: 10 % to 90 % of the 5 ms RMS envelope, to 90 % of its loudest within the
  first second (a step reads about 4 ms; a tremolo, which would otherwise put the target
  anywhere in the held note, can still add up to a quarter of its period);
  "(rising)" when the note later gets more than 1 dB louder than that. **Decay**:
  from the 20 ms envelope's peak to 20 dB below it, while the note is held (pitched) or
  within 4 s (drums); "sustains" when it does not fall that far.
- **Tail floor** (from the file the note plays, the loudest zone): 10th percentile of 50 ms
  RMS windows after the sound first falls 40 dB, the last 100 ms left out, in dB of the file's
  peak, with the spectral flatness of the quietest windows (near 1 = hiss, near 0 = tone);
  – when the file loops or never falls 40 dB. **Last 20 ms**: RMS of the file's end in dB of
  its peak (a non-looping file ending loud is cut, not faded).
- **Plays s**: how long the matched note's file lasts at the speed Dawsome plays it (a
  single-sample set an octave below its root plays twice as long), or "loops".
- **Width**: L/R correlation and side-over-mid energy of the note's section; "mono" when
  both sides are identical.
- **Pitch**: YIN median from 0.15 s to 1.15 s of the note, cents off A4 = 440 Hz equal
  temperament, searched within 6 semitones of the key; when nothing voices there, over
  25 Hz–4.2 kHz, and "-1 oct" says the period found lies an octave below the key (a
  dominant sub-octave partial, such as an organ's 16' drawbar).
- **Loop seam** (from the file): level jump over the wrap (50 ms either side), third-octave
  spectral distance across it beside the median of the same distance between neighboring
  windows inside the loop (the sound's own change; a seam well above it is a jump), the
  second-difference spike at the wrap beside the same ratio inside the loop, the loop end
  within 3 frames that joins most smoothly ("-1": the frame at loop_end repeats the one at
  loop_start, an exclusive end written as inclusive, so every wrap repeats a frame), and the
  match (correlation of the 10 ms leading into the wrap with the 10 ms leading into
  loop_start). **Heard**: the same jumps measured in the phrase's render at the wrap frames
  Dawsome's loop scaling gives (after its resampling to 48 kHz), for the zone the 12 s held
  note sounds, beside their medians halfway between wraps.
- **Listening files**: every one normalized to **−18 LUFS integrated** (the gain is in the
  last table), written as 32-bit float WAV so nothing is clipped in the file; a file marked
  "over 0 dBFS" peaks above full scale at that loudness (a spiky hit), so turn the player
  down by that much or it clips on playback. Level never decides. The
  pitched phrase: the motif C–E–G–C as eighths at velocity 40, then 80, then 120, a run of
  32nds up and down the octave at 80, then the root held 12 s at 80 (the loops of ours and
  the references first wrap up to 9.6 s into a held note at their class's note, so a shorter
  hold would end before the loop it exposes), 20 s in all. A drum hit: quarters at 40/80/120/80, four 16ths at 80, one at 120, then rest
  (6 s). A kit's beat: 4 bars of eighth hats, kick and snare, an open hat closing bars 2 and
  4, a tom fill, a crash on 1 and a last crash with a kick left to ring (12 s); the hits it
  used are listed in the report's JSON (`beat.roles`).
"""


def report_md(results, refs, out, listen_root, binary):
    parts = ["# Instrument comparison", "",
             f"Written by `dawsome-kit/compare.py`. Renders: `{binary}`. Listening folder: "
             f"`{listen_root}`. Every listening file is normalized to −18 LUFS integrated.", "",
             "Suspicions, not verdicts: these numbers say where to listen, not how it sounds.",
             "", METHOD]
    ours_pitched = [r for r in results if r.get("measures")]
    parts += ["## Inventory", "", inventory_rows(results), "",
              "## Pitched sets at the matched note", "", pitched_rows(ours_pitched), "",
              "## Loops", "", loop_rows(ours_pitched), "",
              "## Drums, one row per hit", "", drum_rows(results), "",
              "## One-shots and loops (measured as files)", "",
              samples_rows([r for r in results if r["kind"] == "samples"]), "",
              "## What Dawsome said while rendering", "", warnings_text(results), "",
              "## Listening files", "", listen_rows(results, listen_root), ""]
    if refs:
        parts += ["## References", "", "How each was read:", "", flatten_notes(refs), "",
                  inventory_rows(refs), "", pitched_rows([r for r in refs if r.get("measures")]),
                  "", loop_rows([r for r in refs if r.get("measures")]), "",
                  drum_rows(refs), "", "### Ours beside theirs", "", comparison_rows(results, refs),
                  "", "### Drums beside theirs, role by role", "",
                  drum_comparison_rows(results, refs), "", "### What Dawsome said rendering the references", "", warnings_text(refs),
                  "", listen_rows(refs, listen_root), ""]
    (out / "compare-report.md").write_text("\n".join(parts), encoding="utf-8")


# How far apart two values of a measure are, in units of a plainly audible difference: half
# an octave of brightness, a doubling of attack or decay, 10 dB of noise floor, 0.3 of stereo
# correlation, 6 dB of velocity span, 10 cents, 2 layers or round robins. A judgement, written
# down so the ranking can be argued with; LISTEN.md says so.
UNITS = {
    "hz": lambda a, b: abs(math.log2(a / b)) / 0.5,
    "ms": lambda a, b: abs(math.log2((a + 1) / (b + 1))),
    "decay": lambda a, b: abs(math.log2(a / b)),
    "db": lambda a, b: abs(a - b) / 10,
    "corr": lambda a, b: abs(a - b) / 0.3,
    "span": lambda a, b: abs(a - b) / 6,
    "ratio": lambda a, b: abs(math.log2(a / b)) / 0.5,
    "cents": lambda a, b: min(abs(a - b) / 10, 12),
    "count": lambda a, b: abs(a - b) / 2,
}


def shown(kind, value):
    if kind == "hz":
        return f"{value / 1000:.2f} kHz" if value >= 1000 else f"{value:.0f} Hz"
    if kind == "ms":
        return f"{value:.0f} ms"
    if kind == "decay":
        return "sustains" if value >= HOLD_BARS * BAR * SECONDS_PER_TICK * 1000 else f"{value / 1000:.2f} s"
    if kind in ("db", "span"):
        return f"{value:.0f} dB"
    if kind == "corr":
        return f"{value:.2f}"
    if kind == "ratio":
        return f"{value:.2f}×"
    if kind == "cents":
        octaves = round(value / 1200)
        return (f"{octaves:+d} oct " if octaves else "") + f"{value - 1200 * octaves:+.0f} ¢"
    return f"{value:g}"


def hit_values(hit):
    held = HOLD_BARS * BAR * SECONDS_PER_TICK * 1000
    tail = hit.get("tail") or {}
    return {"brightness at the onset": ("hz", hit.get("centroid_onset")),
            "rolloff at the onset": ("hz", hit.get("rolloff_onset")),
            "attack 10–90 %": ("ms", hit.get("attack_ms")),
            "decay to −20 dB": ("decay", min(hit["decay20_ms"], held)
                                if hit.get("decay20_ms") is not None else held),
            "tail noise floor": ("db", tail.get("floor_db")),
            "stereo correlation": ("corr", hit.get("lr_correlation")),
            "velocity 40→120 loudness": ("span", hit.get("dyn_db")),
            "velocity 40→120 brightness": ("ratio", hit.get("brightness_ratio"))}


def pitched_values(result):
    m, inv = result.get("measures") or {}, result.get("inventory") or {}
    values = hit_values(m)
    values["brightness 500 ms in"] = ("hz", m.get("centroid_500ms"))
    if m.get("pitch_cents") is not None:
        values["pitch"] = ("cents", m["pitch_cents"] + 1200 * (m.get("pitch_octaves") or 0))
    values["velocity layers"] = ("count", inv.get("velocity_layers"))
    values["round robins"] = ("count", inv.get("round_robins"))
    return values


def role_values(hits, roles=None):
    """{"<role> <measure>": (kind, median)} over the hits of each role (of `roles`)."""
    by_role = {}
    for hit in hits:
        if hit.get("role") and (roles is None or hit["role"] in roles):
            by_role.setdefault(hit["role"], []).append(hit_values(hit))
    out = {}
    for role, rows in by_role.items():
        for name in rows[0]:
            kind = rows[0][name][0]
            numbers = [r[name][1] for r in rows if r[name][1] is not None]
            if numbers:
                out[f"{role.replace('_', ' ')}: {name}"] = (kind, float(np.median(numbers)))
    return out


def class_values(result, hits_only=None, roles=None):
    if result.get("measures"):
        return pitched_values(result)
    hits = result.get("hits") or []
    if hits_only is not None:
        hits = [h for h in hits if h["hit"] in hits_only]
    return role_values(hits, roles)


def differences(mine, theirs):
    """[(units, measure, ours shown, theirs shown)] for the measures both have, largest first."""
    out = []
    for name, (kind, a) in mine.items():
        if name not in theirs or a is None or theirs[name][1] is None:
            continue
        b = theirs[name][1]
        if kind in ("hz", "decay", "ratio") and (a <= 0 or b <= 0):
            continue
        out.append((UNITS[kind](a, b), name, shown(kind, a), shown(kind, b)))
    return sorted(out, reverse=True)


def listen_files(result, out, hits_only=None):
    """The listening files of a result, relative to `out`."""
    if result.get("listen"):
        return [Path(result["listen"]["file"]).relative_to(out)]
    if result.get("beat") and hits_only is None:
        return [Path(result["beat"]["file"]).relative_to(out)]
    hits = [h for h in result.get("hits") or [] if hits_only is None or h["hit"] in hits_only]
    return [Path(h["listen"]["file"]).relative_to(out) for h in hits]


RANK_CAP = 4.0


def listen_md(results, refs, items, mapping, out):
    """LISTEN.md: per class, our file, then each reference's file with the three measures that
    differ most from ours; at the top, the five classes where ours is furthest from even its
    nearest reference (references that are our own sources left out of that ranking)."""
    by_name = {r["name"]: r for r in results}
    starter = next((r for r in results if r["kind"] == "drum-pack"), None)
    classes = list(dict.fromkeys(str(i.get("class")) for i in items))
    sections, ranking, pitched = [], [], []
    for cls in classes:
        mine_map = mapping.get(cls, {})
        roles = CLASS_ROLES.get(cls)
        if mine_map.get("hits") and starter:
            mine, hits_only = starter, mine_map["hits"]
        else:
            mine, hits_only = by_name.get(mine_map.get("set") or ""), None
        if not roles and not (mine and mine.get("kind") in ("drums", "drum-pack")):
            pitched.append(cls)
        class_refs = [r for r in refs if str(r.get("class")) == cls]
        not_got = [i for i in items if str(i.get("class")) == cls and not i.get("folder")]
        lines = [f"## {cls}", ""]
        if mine is None:
            lines.append("ours: none paired (the index names no set of ours for this class)")
            mine_values = {}
        else:
            files = listen_files(mine, out, hits_only)
            lines.append("ours: " + ", ".join(f"`{f}`" for f in files)
                         + f" (`{mine['name']}`" + (f": {', '.join(hits_only)}" if hits_only else "")
                         + ")")
            mine_values = class_values(mine, hits_only, roles)
        lines.append("")
        nearest = None
        for ref in class_refs:
            tag = ref.get("index_tag") or "–"
            if ref.get("status") == "already-used":
                tag += "; the source ours was made from"
            head = f"- {ref.get('library')} ({tag}): "
            if ref.get("problem"):
                lines.append(head + f"not rendered: {ref['problem']}")
                continue
            files = listen_files(ref, out)
            shown_files = (", ".join(f"`{f}`" for f in files) if len(files) <= 2 else
                           f"`{files[0].parent}/` ({len(files)} hits)")
            lines.append(head + shown_files)
            diffs = differences(mine_values, class_values(ref, None, roles))
            if not diffs:
                lines.append("  - no measure in common with ours")
                continue
            for units, name, a, b in diffs[:3]:
                lines.append(f"  - {name}: ours {a}, theirs {b} ({units:.1f} units)")
            # Each measure counts at most RANK_CAP units: an octave's mapping or 16 round
            # robins against 1 is one big difference, not the whole distance.
            distance = sum(min(d[0], RANK_CAP) for d in diffs) / len(diffs)
            if ref.get("status") != "already-used" and (nearest is None or distance < nearest[0]):
                nearest = (distance, ref.get("library"), diffs[:3])
        if not_got:
            lines.append(f"- not downloaded: {', '.join(str(i.get('library')) for i in not_got)}")
        if nearest:
            ranking.append((nearest[0], cls, nearest[1], nearest[2]))
        sections.append("\n".join(lines))
    ranking.sort(reverse=True)
    top = ["# What to listen to", "",
           f"Listening files are under `{out}`, every one at −18 LUFS integrated, rendered by "
           "Dawsome at 48 kHz (the same sampler for ours and theirs). Measures: see "
           "`compare-report.md`. A reference made from many files (a drum pack) is compared by "
           "the median of its hits of each role.", "",
           "\"Units\" put the measures on one scale of a plainly audible difference: half an "
           "octave of brightness, a doubling of attack or decay time, 10 dB of noise floor, 0.3 "
           "of stereo correlation, 6 dB of velocity span, 10 cents, 2 velocity layers or round "
           "robins. That scale is a judgement, not a standard; a class's distance from a "
           "reference is the mean over the measures both have, each counting at most "
           f"{RANK_CAP:g} units (the lines below show the full value). A pitch an octave off is "
           "usually how a library maps its keys (basses are often mapped an octave low), not "
           "how it sounds.", "",
           "Notes (C4 = 60), each pitched class's, where its sets are measured and its phrase "
           "starts when the phrase fits their keys: "
           + "; ".join(f"{cls} {note_name(class_note(cls))} ({class_note(cls)})"
                       for cls in pitched) + ".", "",
           "## The five classes furthest from every reference", "",
           "Each is ranked by its distance from its *nearest* reference (references that are the "
           "source our set was made from are left out of this ranking).", ""]
    for distance, cls, library, diffs in ranking[:5]:
        detail = "; ".join(f"{name} ours {a} vs {b}" for _, name, a, b in diffs)
        top.append(f"1. **{cls}**: {distance:.2f} units from the nearest, {library} ({detail})")
    if not ranking:
        top.append("- no class has a rendered reference")
    failed = [r for r in refs if r.get("problem")]
    skipped = sum(len(r.get("skipped") or []) for r in refs)
    top += ["", f"{len(refs) - len(failed)} references rendered, {len(failed)} not (each "
            f"says why under its class); {skipped} files skipped while reading them "
            "(`compare-report.json`, `skipped`, lists each).", ""]
    (out / "LISTEN.md").write_text("\n".join(top) + "\n" + "\n\n".join(sections) + "\n",
                                   encoding="utf-8")


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(v) for v in (sorted(value, key=str) if isinstance(value, set) else value)]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if np is not None and isinstance(value, np.generic):
        return jsonable(value.item())
    return value


# ------------------------------------------------------------------------------------ main

def default_binary():
    target = os.environ.get("CARGO_TARGET_DIR") or str(ROOT.parent / "dawsome" / "target")
    return Path(os.environ.get("DAWSOME_BIN") or Path(target) / "release" / "dawsome")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True,
                        help="the folder for the report, the listening files (ours/, refs/) "
                             "and the work files (work/: projects, raw renders, generated SFZ)")
    parser.add_argument("--dawsome", type=Path, default=default_binary())
    parser.add_argument("--dawsome-repo", type=Path, default=ROOT.parent / "dawsome")
    parser.add_argument("--refs", type=Path, help="the reference libraries' index.json")
    parser.add_argument("--pair", action="append", default=[], metavar="CLASS=SET",
                        help="compare the references of CLASS with our SET")
    parser.add_argument("--keep-renders", action="store_true",
                        help="keep the raw renders under work/ (removed once measured)")
    parser.add_argument("--only", action="append", default=[], metavar="NAME",
                        help="only these sets of ours (and references of these classes)")
    args = parser.parse_args(argv)
    if IMPORT_ERROR is not None:
        print(f"compare: needs numpy, scipy and soundfile ({IMPORT_ERROR})", file=sys.stderr)
        return 2
    if not args.dawsome.is_file():
        print(f"compare: no dawsome binary at {args.dawsome} (build it, or pass --dawsome)",
              file=sys.stderr)
        return 2
    out = args.out.resolve()
    work, listen_root = out / "work", out
    out.mkdir(parents=True, exist_ok=True)
    pairs = dict(p.split("=", 1) for p in args.pair)
    renderer = Renderer(args.dawsome, work, args.keep_renders)
    try:
        ours = our_sets(args.dawsome_repo)
        results = []
        for entry in ours:
            if args.only and entry["name"] not in args.only:
                continue
            entry["tag"] = f"ours/{entry['name']}"
            entry["listen"] = (f"ours/drums/{entry['name']}" if entry["kind"] in ("drums", "drum-pack")
                               else f"ours/{entry['name']}")
            if entry.get("sfz"):
                entry["regions"] = read_set(entry["sfz"])
                entry["render_sfz"] = entry["sfz"]
            print(f"compare: {entry['name']}", file=sys.stderr)
            results.append(run_set(renderer, entry, listen_root, set_note(entry["name"])))
        refs, items, mapping = [], [], {}
        if args.refs:
            by_name = {r["name"]: r for r in results}
            items = read_index(args.refs)
            mapping = ours_by_class(items, ours)
            for item in items:
                if args.only and str(item.get("class")) not in args.only:
                    continue
                if item.get("status") == "ours" or not item.get("folder"):
                    continue
                name = f"{item.get('class')}/{item.get('library')}"
                print(f"compare: reference {name}", file=sys.stderr)
                skipped = []
                try:
                    ref = prepare_ref(item, ours, pairs, work, mapping, skipped)
                    ref["listen"] = "refs/" + ref["listen"]
                    mine = by_name.get(ref.get("pair")) or {}
                    prefer = class_note(item.get("class"))
                    if (mine.get("measures") or {}).get("key") not in (None, prefer):
                        print(f"compare: {name}: ours, {mine['name']}, is measured at"
                              f" {note_name(mine['measures']['key'])}, the class at"
                              f" {note_name(prefer)}", file=sys.stderr)
                    refs.append(run_set(renderer, ref, listen_root, prefer))
                except Exception as error:  # one library's failure is reported, not fatal
                    problem = f"{type(error).__name__}: {error}".splitlines()[0][:300]
                    print(f"compare: reference {name} failed: {problem}", file=sys.stderr)
                    refs.append({"name": name, "class": item.get("class"),
                                 "library": item.get("library"), "status": item.get("status"),
                                 "index_tag": item.get("tag"), "kind": "failed",
                                 "problem": problem, "skipped": skipped})
    finally:
        left = renderer.close()
    if left:
        print(f"compare: the renders wrote into their temporary home: {left}", file=sys.stderr)
        return 1
    report = {"tool": "dawsome-kit/compare.py", "dawsome": str(args.dawsome),
              "rate": RATE, "target_lufs": TARGET_LUFS, "match_velocity": MATCH_VELOCITY,
              "renders": renderer.renders, "ours": results, "refs": refs}
    (out / "compare-report.json").write_text(json.dumps(jsonable(report), indent=1) + "\n",
                                             encoding="utf-8")
    report_md(results, refs, out, listen_root, args.dawsome)
    if args.refs:
        listen_md(results, refs, items, mapping, out)
    print(f"compare: {renderer.renders} renders; {out / 'compare-report.md'}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
