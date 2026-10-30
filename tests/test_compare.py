# SPDX-License-Identifier: GPL-3.0-or-later
"""compare.py against answers from outside it: ITU-R BS.1770-4's filter coefficients and EBU
Tech 3341's loudness test signals, signals whose measures follow from how they are built
(a sine's centroid, a ramp's rise, an exponential's decay, a loop's level jump), the note
mapping and drum names the shipped .sfz files state, the SFZ rules for what a player plays
at default controllers, and Dawsome's own render of a shipped set (the stand-in reference
must sound exactly as the set does).

Needs numpy, scipy and soundfile; skipped without them. The render tests need a built
dawsome (DAWSOME_BIN or CARGO_TARGET_DIR/release/dawsome) and are skipped without it."""

import json
import math
import struct
import tempfile
import unittest
import wave
from pathlib import Path

import compare

ROOT = compare.ROOT
NEEDS = unittest.skipIf(compare.IMPORT_ERROR is not None, "needs numpy, scipy and soundfile")
BINARY = compare.default_binary()
NEEDS_DAWSOME = unittest.skipUnless(BINARY.is_file() and compare.IMPORT_ERROR is None,
                                    f"needs a built dawsome at {BINARY}")

if compare.IMPORT_ERROR is None:
    import numpy as np
    import soundfile as sf

RATE = 48000


def sine(freq, seconds, amp=1.0, rate=RATE):
    t = np.arange(int(round(seconds * rate))) / rate
    return amp * np.sin(2 * np.pi * freq * t)


def stereo(m):
    return np.stack((m, m), axis=1)


def dbfs(level):
    return 10 ** (level / 20)


@NEEDS
class Loudness(unittest.TestCase):
    def test_k_weighting_is_the_standards_48k_filter(self):
        # ITU-R BS.1770-4, Annex 1, Tables 1 and 2 (48 kHz).
        (b1, a1), (b2, a2) = compare.k_weighting(48000)
        for got, want in zip(b1 + a1[1:], (1.53512485958697, -2.69169618940638,
                                           1.19839281085285, -1.69065929318241,
                                           0.73248077421585)):
            self.assertAlmostEqual(got, want, places=7)
        self.assertEqual(b2, [1.0, -2.0, 1.0])
        self.assertAlmostEqual(a2[1], -1.99004745483398, places=7)
        self.assertAlmostEqual(a2[2], 0.99007225036621, places=7)

    def test_ebu_tech_3341_cases(self):
        # EBU Tech 3341 (2016), table 1: stereo 1 kHz sines; integrated -23.0 +- 0.1 LUFS
        # (cases 1 and 2: -23.0 and -33.0).
        cases = [
            ([(-23, 20)], -23.0),
            ([(-33, 20)], -33.0),
            ([(-36, 10), (-23, 60), (-36, 10)], -23.0),
            ([(-72, 10), (-36, 10), (-23, 60), (-36, 10), (-72, 10)], -23.0),
            ([(-26, 20), (-20, 20.1), (-26, 20)], -23.0),
        ]
        for parts, want in cases:
            m = np.concatenate([sine(1000, s, dbfs(level)) for level, s in parts])
            got = compare.integrated_loudness(stereo(m), RATE)
            self.assertAlmostEqual(got, want, delta=0.1, msg=parts)

    def test_silence_and_short_signals_have_no_loudness(self):
        self.assertIsNone(compare.integrated_loudness(np.zeros((RATE, 2)), RATE))
        self.assertIsNone(compare.integrated_loudness(stereo(sine(1000, 0.3, 0.5)), RATE))

    def test_normalised_files_land_on_the_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            x = stereo(np.concatenate((sine(440, 2, 0.05), np.zeros(RATE))))
            info = compare.normalise(x, Path(tmp) / "a.wav")
            y, rate = sf.read(str(Path(tmp) / "a.wav"), always_2d=True)
            self.assertEqual(sf.info(str(Path(tmp) / "a.wav")).subtype, "FLOAT")
            self.assertAlmostEqual(compare.integrated_loudness(y, rate), -18.0, delta=0.01)
            self.assertAlmostEqual(info["gain_db"], -18.0 - info["lufs_before"], places=9)


@NEEDS
class Measures(unittest.TestCase):
    def test_a_sines_centroid_and_rolloff_are_its_frequency(self):
        m = sine(1000, 1, 0.5)
        centroid, rolloff = compare.spectrum_at(m, RATE, 0)
        self.assertAlmostEqual(centroid, 1000, delta=10)
        self.assertAlmostEqual(rolloff, 1000, delta=24)       # one bin: 48000 / 2048

    def test_attack_of_a_linear_ramp(self):
        # A 100 ms linear rise: 10 % at 10 ms, 90 % at 90 ms, 80 ms apart.
        n = int(0.1 * RATE)
        m = sine(1000, 1, 1.0) * np.concatenate((np.arange(n) / n, np.ones(RATE - n)))
        ms, rising = compare.attack_ms(m, RATE, 0, len(m))
        self.assertAlmostEqual(ms, 80, delta=3)
        self.assertFalse(rising)

    def test_attack_through_a_tremolo_and_a_swell(self):
        # An instant onset at the trough of a 2.5 Hz, 3 dB tremolo whose loudest point comes at
        # 3.9 s: 90 % is reached on the tremolo's first rise, within its quarter period (100 ms),
        # not seconds later where the held note is loudest.
        t = np.arange(4 * RATE) / RATE
        tremolo = 10 ** ((-1.5 + 1.5 * np.sin(2 * np.pi * 2.5 * t) + 0.3 * t / 4) / 20)
        ms, rising = compare.attack_ms(np.sin(2 * np.pi * 300 * t) * tremolo, RATE, 0, len(t))
        self.assertLess(ms, 100)
        self.assertFalse(rising)
        # A 2 s linear swell: still rising after the first second.
        swell = np.minimum(t / 2, 1)
        self.assertTrue(compare.attack_ms(np.sin(2 * np.pi * 300 * t) * swell, RATE, 0, len(t))[1])

    def test_decay_of_an_exponential(self):
        # e^(-t/0.2) falls 20 dB at t = 0.2 ln 10 = 460.5 ms.
        t = np.arange(2 * RATE) / RATE
        m = np.sin(2 * np.pi * 1000 * t) * np.exp(-t / 0.2)
        self.assertAlmostEqual(compare.decay_ms(m, RATE, 0, len(m)), 460.5, delta=12)
        self.assertIsNone(compare.decay_ms(sine(1000, 2), RATE, 0, 2 * RATE))

    def test_width(self):
        m = sine(300, 1, 0.5)
        self.assertEqual(compare.width(stereo(m)), (1.0, None))
        rng = np.random.default_rng(1)
        corr, side = compare.width(rng.standard_normal((RATE, 2)))
        self.assertAlmostEqual(corr, 0, delta=0.02)
        self.assertAlmostEqual(side, 0, delta=0.2)
        corr, side = compare.width(np.stack((m, -m), axis=1))
        self.assertAlmostEqual(corr, -1, places=6)

    def test_pitch(self):
        # A3 = key 57 = 220 Hz; 10 cents sharp is 220 * 2^(10/1200).
        cents, octaves = compare.pitch_cents(sine(220, 2), RATE, 0, 2 * RATE, 57)
        self.assertEqual(octaves, 0)
        self.assertAlmostEqual(cents, 0, delta=1)
        sharp = sine(220 * 2 ** (10 / 1200), 2)
        self.assertAlmostEqual(compare.pitch_cents(sharp, RATE, 0, 2 * RATE, 57)[0], 10, delta=1)

    def test_a_dominant_sub_octave_reads_an_octave_low(self):
        # C3 = 130.81 Hz with a louder C2 below it and C4 above (16', 8', 4' drawbars): the
        # period is C2's, so the note sounds an octave below its key.
        c3 = 130.8128
        m = sine(c3 / 2, 2, 0.5) + sine(c3, 2, 0.2) + sine(2 * c3, 2, 0.4)
        cents, octaves = compare.pitch_cents(m, RATE, 0, 2 * RATE, 48)
        self.assertEqual(octaves, -1)
        self.assertAlmostEqual(cents, 0, delta=2)

    def test_tail_floor_of_a_decay_into_noise(self):
        # A tone decaying into white noise 60 dB under its peak: the floor reads -60 dB (RMS of
        # the noise against the tone's peak of 1, so 20 log10(0.001) = -60) and it is noise.
        rng = np.random.default_rng(2)
        t = np.arange(6 * RATE) / RATE
        m = np.sin(2 * np.pi * 440 * t) * np.exp(-t / 0.3) + 0.001 * rng.standard_normal(len(t))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "t.wav"
            sf.write(str(path), m, RATE, subtype="FLOAT")
            tail = compare.tail_measures(path, {})
        self.assertAlmostEqual(tail["floor_db"], -60, delta=1.5)
        self.assertGreater(tail["floor_flatness"], 0.5)
        self.assertAlmostEqual(tail["end_db"], -60, delta=1.5)


def write_loop_file(path, m, start, end):
    sf.write(str(path), m, RATE, subtype="FLOAT")
    return {"loop_start": start, "loop_end": end, "loop_mode": "loop_continuous"}


@NEEDS
class Seams(unittest.TestCase):
    def test_a_whole_number_of_periods_is_seamless(self):
        m = sine(1000, 3, 0.5)                          # 48 samples a period
        with tempfile.TemporaryDirectory() as tmp:
            region = write_loop_file(Path(tmp) / "a.wav", m, 48000, 48000 + 48 * 1000 - 1)
            seam = compare.seam_measures(Path(tmp) / "a.wav", region)
        self.assertAlmostEqual(seam["level_jump_db"], 0, delta=0.01)
        self.assertLess(seam["spectral_jump_db"], 0.5)
        self.assertLess(seam["seam_click"], 2 * seam["click_inside"] + 1)
        self.assertAlmostEqual(seam["match"], 1, places=6)
        self.assertEqual(seam["end_offset"], 0)

    def test_a_repeated_frame_is_an_end_one_too_late(self):
        # loop_end one frame past a whole number of periods: the frame at loop_end is the one at
        # loop_start again, so the best end is one frame earlier.
        m = sine(1000, 3, 0.5)
        with tempfile.TemporaryDirectory() as tmp:
            region = write_loop_file(Path(tmp) / "a.wav", m, 48000, 48000 + 48 * 1000)
            seam = compare.seam_measures(Path(tmp) / "a.wav", region)
        self.assertEqual(seam["end_offset"], -1)
        self.assertGreater(seam["seam_click"], 5 * seam["seam_click_best"])

    def test_a_loop_over_a_decay_jumps_by_the_decay(self):
        # e^(-t/tau) sine: the window after loop_start is louder than the one before loop_end by
        # e^((le + 1 - 2400 - ls) / (tau rate)), both windows whole periods of the sine.
        tau, ls, le = 1.0, 24000, 24000 + 48 * 500 - 1
        t = np.arange(3 * RATE) / RATE
        m = np.sin(2 * np.pi * 1000 * t) * np.exp(-t / tau)
        want = 20 * math.log10(math.exp((le + 1 - 2400 - ls) / (tau * RATE)))
        with tempfile.TemporaryDirectory() as tmp:
            region = write_loop_file(Path(tmp) / "a.wav", m, ls, le)
            seam = compare.seam_measures(Path(tmp) / "a.wav", region)
        self.assertAlmostEqual(seam["level_jump_db"], want, delta=0.01)

    def test_a_broken_period_clicks(self):
        m = sine(1000, 3, 0.5)
        with tempfile.TemporaryDirectory() as tmp:
            region = write_loop_file(Path(tmp) / "a.wav", m, 48000, 48000 + 48 * 1000 + 11)
            seam = compare.seam_measures(Path(tmp) / "a.wav", region)
        self.assertGreater(seam["seam_click"], 10 * seam["click_inside"])
        self.assertLess(seam["match"], 0.5)


def region(lo, hi, centre, **more):
    return {**compare.REGION_DEFAULTS, "lokey": lo, "hikey": hi, "pitch_keycenter": centre,
            "sample": "x", **more}


class Keys(unittest.TestCase):
    def test_velocity_bytes_read_back_exactly(self):
        for byte in range(1, 128):
            self.assertEqual(round(compare.velocity(byte) * 128), byte)

    def test_note_names(self):
        # C4 = 60 (Dawsome's and SFZ's convention).
        self.assertEqual(compare.name_key("Piano_C#4_v2"), (1, 61))
        self.assertEqual(compare.name_key("cello_Db3"), (1, 49))
        self.assertEqual(compare.name_key("A0vL"), (9, 21))
        self.assertEqual(compare.name_key("E"), (4, None))
        self.assertEqual(compare.name_key("F#"), (6, None))
        self.assertIsNone(compare.name_key("kick_vl1_rr2"))

    def test_root_key(self):
        self.assertEqual(compare.root_key([region(21, 108, 60)]), 48)
        self.assertEqual(compare.root_key([region(0, 127, 48)]), 48)
        # bass-electric's keys, 26-45: no C fits C..C+12, so the key nearest C3 that does.
        self.assertEqual(compare.root_key([region(26, 45, 33)]), 33)
        self.assertEqual(compare.root_key([region(28, 59, 33)]), 36)
        self.assertEqual(compare.root_key([region(21, 108, 60)], prefer=33), 33)
        self.assertEqual(compare.root_key([region(40, 100, 60)], prefer=33), 48)
        # bass-electric at its class's C2: neither C2..C3 nor C1..C2 lies in 26-45, so the key
        # nearest C2 from which the phrase does: A1.
        self.assertEqual(compare.root_key([region(26, 45, 33)], prefer=36), 33)
        self.assertEqual(compare.root_key([region(28, 59, 33)], prefer=36), 36)
        # The search centers on the note asked for: keys 30-127 asked for C1 take C2, the C
        # nearest C1 that fits (centered on C3, it was C3).
        self.assertEqual(compare.root_key([region(30, 127, 48)], prefer=24), 36)

    def test_class_notes(self):
        # Where each class's instruments live (C4 = 60): the basses at C2, the 808 at C1, the
        # piano, the leads, plucks and electric pianos at C4 (the shipped lead-c4, pluck-c4 and
        # ep-c4 are rooted there), the pads at C3 (pad-warm-c3, pad-bright-c3), the rest at C3.
        for cls, key in (("bass-electric", 36), ("bass-guitar-5", 36), ("bass-synth", 36),
                         ("bass-analog", 36), ("bass-808", 24), ("piano", 60), ("lead", 60),
                         ("pluck", 60), ("ep", 60), ("pad-warm", 48), ("pad-bright", 48),
                         ("pad-choir", 48), ("organ", 48), ("marimba", 48), ("strings", 48)):
            self.assertEqual(compare.class_note(cls), key, cls)

    def test_our_sets_take_the_note_of_the_class_they_are_named_for(self):
        # The sets Dawsome ships, dawsome-kit's sets/ and the Dawsome repository's
        # assets/instruments/, by folder name.
        for name, key in (("bass-electric", 36), ("bass-guitar", 36), ("bass-synth", 36),
                          ("bass-synth-c2", 36), ("bass-808-c1", 24), ("bass-808-c2", 24),
                          ("piano", 60), ("lead-c4", 60), ("pluck-c4", 60), ("ep-c4", 60),
                          ("pad-warm-c3", 48), ("pad-bright-c3", 48), ("pad-sweep", 48),
                          ("pad-choir", 48), ("mallets", 48), ("organ", 48), ("strings", 48),
                          ("guitar-acoustic", 48), ("guitar-electric", 48)):
            self.assertEqual(compare.set_note(name), key, name)

    def test_phrase(self):
        notes = compare.phrase_notes(48)
        self.assertEqual({n[3] for n in notes[:12]}, {40, 80, 120})
        self.assertTrue(all(48 <= n[2] <= 60 for n in notes))
        held = notes[-1]
        # 12 s: the latest first wrap of a loop, ours or a reference's, at its class's note was
        # 9.6 s (2026-10-27), and the hold keeps 2 s more than that.
        self.assertEqual(held[1] * compare.SECONDS_PER_TICK, 12.0)
        self.assertGreaterEqual(held[1] * compare.SECONDS_PER_TICK, 9.6 + 2)
        self.assertLess((held[0] + held[1]) * compare.SECONDS_PER_TICK + 2,
                        compare.PHRASE_BARS * compare.BAR * compare.SECONDS_PER_TICK)


@NEEDS
class Packs(unittest.TestCase):
    def test_pitch_detection_places_the_bass_files_where_the_shipped_sfz_does(self):
        # sets/bass-electric's files are named by pitch class alone (E.flac, F#.flac ...); its
        # .sfz says which key each is (pitch_keycenter), measured when the set was built.
        folder = ROOT / "sets" / "bass-electric"
        shipped = {Path(r["sample"]).name: r["pitch_keycenter"]
                   for r in compare.read_set(folder / "bass-electric.sfz")}
        regions, log = compare.pack_regions(sorted((folder / "samples").iterdir()))
        placed = {Path(r["sample"]).name: r["pitch_keycenter"] for r in regions}
        self.assertEqual(placed, shipped, log)

    def test_unnamed_sines_are_placed_by_pitch(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name, freq in (("one", 110.0), ("two", 220.0), ("three", 261.6256)):
                sf.write(str(Path(tmp) / f"{name}.wav"), sine(freq, 1.5, 0.5), RATE)
            regions, _ = compare.pack_regions(sorted(Path(tmp).iterdir()))
        self.assertEqual(sorted(r["pitch_keycenter"] for r in regions), [45, 57, 60])
        ranges = sorted((r["lokey"], r["hikey"]) for r in regions)
        self.assertEqual(ranges, [(39, 51), (52, 58), (59, 66)])

    def test_dynamics_and_round_robins(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name in ("C3_pp_rr1", "C3_pp_rr2", "C3_ff_rr1", "C3_ff_rr2"):
                sf.write(str(Path(tmp) / f"{name}.wav"), sine(130.8128, 1.0, 0.5), RATE)
            regions, _ = compare.pack_regions(sorted(Path(tmp).iterdir()))
        layers = sorted({(r["lovel"], r["hivel"], r["seq_length"]) for r in regions})
        self.assertEqual(layers, [(1, 63, 2), (64, 127, 2)])
        self.assertTrue(all(r["pitch_keycenter"] == 48 for r in regions))

    def test_wav_smpl_loop(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "l.wav"
            with wave.open(str(path), "wb") as out:
                out.setnchannels(1)
                out.setsampwidth(2)
                out.setframerate(RATE)
                out.writeframes(b"\0\0" * 1000)
            smpl = struct.pack("<9I", 0, 0, 20833, 60, 0, 0, 0, 1, 0) + struct.pack(
                "<6I", 0, 0, 100, 899, 0, 0)
            data = path.read_bytes() + b"smpl" + struct.pack("<I", len(smpl)) + smpl
            data = data[:4] + struct.pack("<I", len(data) - 8) + data[8:]
            path.write_bytes(data)
            self.assertEqual(compare.wav_loop(path), (100, 899))


SFZ_TEXT = """<control>
default_path=s/
set_cc64=0
<global>
ampeg_release=0.5
<group> sw_last=c1
<region> sample=a.wav key=60
<region> sample=a.wav key=61 trigger=release
<region> sample=a.wav key=62 locc64=64
<region> sample=a.wav key=63 hicc64=63
<region> sample=a.wav key=64 lorand=0.5 hirand=1
<region> sample=a.wav key=64 lorand=0 hirand=0.5
<region> sample=a.wav lokey=36 hikey=38 pitch_keytrack=0
<region> sample=../outside.wav key=65
<region> sample=a.wav key=66 fil_type=lpf_2p cutoff=500 amplitude=50
<group> sw_last=d1
<region> sample=a.wav key=70
"""


@NEEDS
class Flatten(unittest.TestCase):
    def library(self, tmp, text=SFZ_TEXT):
        folder = Path(tmp) / "lib"
        (folder / "s").mkdir(parents=True)
        sf.write(str(folder / "s" / "a.wav"), sine(440, 0.2, 0.5), RATE)
        sf.write(str(Path(tmp) / "outside.wav"), sine(440, 0.2, 0.5), RATE)
        (folder / "inst.sfz").write_text(text)
        return folder

    def test_what_a_player_plays_at_default_controllers(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = self.library(tmp)
            flat = compare.flatten_sfz(folder / "inst.sfz", folder)
        keys = sorted((r["lokey"], r["seq_position"], r["seq_length"]) for r in flat["regions"])
        # 60 plays; 61 is a release trigger; 62 needs the pedal down (CC64 at 0 is below 64);
        # 63 plays (0 <= 63); the two lorand halves of 64 play in turn, lorand 0 first;
        # pitch_keytrack=0 over 36-38 is a region per key at its own pitch; ../outside.wav is
        # outside the library; 66 plays, its filter ignored; 70 is the other keyswitch.
        self.assertEqual(keys, [(36, 1, 1), (37, 1, 1), (38, 1, 1), (60, 1, 1), (63, 1, 1),
                                (64, 1, 2), (64, 2, 2), (66, 1, 1)])
        self.assertEqual(flat["release_regions"], 1)
        self.assertEqual(flat["ignored"], {"fil_type": 1, "cutoff": 1})
        by_key = {r["lokey"]: r for r in flat["regions"]}
        self.assertEqual(by_key[37]["pitch_keycenter"], 37)
        self.assertAlmostEqual(by_key[66]["volume"], 20 * math.log10(0.5))
        self.assertEqual(by_key[60]["ampeg_release"], 0.5)
        rr = sorted((r for r in flat["regions"] if r["lokey"] == 64), key=lambda r: r["seq_position"])
        self.assertTrue(all(r["sample"].name == "a.wav" for r in rr))
        self.assertIn("another keyswitch", flat["dropped"])

    def test_a_first_round_robin_naming_hirand_alone_takes_its_turn(self):
        # Karoryfer's Fashionbass: lorand defaults to 0, so its first round robin says only
        # hirand=0.333; the three take turns, none sounds under another.
        text = ("<region> sample=s/a.wav key=60 hirand=0.333\n"
                "<region> sample=s/a.wav key=60 lorand=0.333 hirand=0.666\n"
                "<region> sample=s/a.wav key=60 lorand=0.666\n")
        with tempfile.TemporaryDirectory() as tmp:
            flat = compare.flatten_sfz(self.library(tmp, text) / "inst.sfz", Path(tmp) / "lib")
        self.assertEqual([(r["seq_position"], r["seq_length"]) for r in flat["regions"]],
                         [(1, 3), (2, 3), (3, 3)])

    def test_amplitude_controllers_play_at_their_defaults(self):
        # Big Rusty Drums' mic faders: amplitude_cc70=100 at set_cc70=100 is 100/127 of full
        # amplitude; amplitude_oncc71=70 at set_cc71=40 over amplitude=50 is 0.5 x 0.7 x 40/127;
        # amplitude_cc72 with CC72 at its default 0 is silent, so the region is left out.
        text = ("<control>\nset_cc70=100\nset_cc71=40\n"
                "<region> sample=s/a.wav key=60 amplitude_cc70=100\n"
                "<region> sample=s/a.wav key=61 amplitude_oncc71=70 amplitude=50\n"
                "<region> sample=s/a.wav key=62 amplitude_cc72=100\n")
        with tempfile.TemporaryDirectory() as tmp:
            flat = compare.flatten_sfz(self.library(tmp, text) / "inst.sfz", Path(tmp) / "lib")
        volumes = {r["lokey"]: r["volume"] for r in flat["regions"]}
        self.assertEqual(sorted(volumes), [60, 61])
        self.assertAlmostEqual(volumes[60], 20 * math.log10(100 / 127))
        self.assertAlmostEqual(volumes[61], 20 * math.log10(0.5 * 0.7 * 40 / 127))
        self.assertEqual(flat["dropped"], {"a controller at its default silences it": 1})
        self.assertEqual(flat["ignored"], {})

    def test_include_paths_take_define_variables(self):
        # Splendid Grand writes #define $DIR Data, then #include "$DIR/...": the variable is
        # expanded in the path, as SFZ players do.
        with tempfile.TemporaryDirectory() as tmp:
            folder = self.library(tmp, '#define $DIR parts\n#include "$DIR/keys.txt"\n')
            (folder / "parts").mkdir()
            (folder / "parts" / "keys.txt").write_text("<region> sample=s/a.wav key=60\n")
            flat = compare.flatten_sfz(folder / "inst.sfz", folder)
            self.assertEqual([r["lokey"] for r in flat["regions"]], [60])
            # A variable that leads out of the library is refused like a written path.
            (folder / "inst.sfz").write_text('#define $UP ..\n#include "$UP/evil.sfz"\n')
            (Path(tmp) / "evil.sfz").write_text("<region> sample=x.wav\n")
            with self.assertRaises(compare.kitlib.Problem):
                compare.flatten_sfz(folder / "inst.sfz", folder)

    def test_values_outside_dawsomes_ranges_are_brought_in(self):
        # volume=12 and 8 (the wavestate pads): the set shifts down 6 dB, so the 4 dB between
        # them stays; tune=-250 is -2 semitones and -50 cents; loop_mode=continuous is
        # Karoryfer's loop_continuous.
        text = ("<region> sample=s/a.wav key=60 volume=12 tune=-250 loop_mode=continuous "
                "loop_start=10 loop_end=4000\n<region> sample=s/a.wav key=61 volume=8\n")
        with tempfile.TemporaryDirectory() as tmp:
            folder = self.library(tmp, text)
            flat = compare.flatten_sfz(folder / "inst.sfz", folder)
            out = Path(tmp) / "m"
            sfz = compare.mirror_set(flat["regions"], out, "test")
            back = compare.read_set(sfz)
        a, b = flat["regions"]
        self.assertEqual((a["volume"], b["volume"]), (6.0, 2.0))
        self.assertEqual((a["transpose"], a["tune"]), (-2, -50.0))
        self.assertEqual(a["loop_mode"], "loop_continuous")
        self.assertEqual([r["volume"] for r in back], [6.0, 2.0])

    def test_aiff_is_converted_for_dawsome(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "a.aif"
            sf.write(str(src), sine(440, 0.1, 0.5), RATE, format="AIFF", subtype="PCM_16")
            target = compare.place_sound(src, Path(tmp) / "0000")
            self.assertEqual(target.name, "0000.wav")
            x, rate = sf.read(str(target))
            y, _ = sf.read(str(src))
        self.assertEqual(rate, RATE)
        self.assertTrue(np.allclose(x, y, atol=1e-7))

    def test_a_windows_default_path_is_beside_the_file(self):
        # The vowel choir writes default_path=\choir Ah samples\ (a leading backslash).
        with tempfile.TemporaryDirectory() as tmp:
            folder = self.library(tmp, "<control>\ndefault_path=\\s\\\n<region> sample=a.wav key=60\n")
            flat = compare.flatten_sfz(folder / "inst.sfz", folder)
        self.assertEqual([(r["lokey"], r["sample"].name) for r in flat["regions"]], [(60, "a.wav")])

    def test_an_include_outside_the_library_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = self.library(tmp, '#include "../evil.sfz"\n<region> sample=a.wav\n')
            (Path(tmp) / "evil.sfz").write_text("<region> sample=x.wav\n")
            with self.assertRaises(compare.kitlib.Problem):
                compare.flatten_sfz(folder / "inst.sfz", folder)

    def test_generated_sfz_reads_back_the_same(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = self.library(tmp)
            flat = compare.flatten_sfz(folder / "inst.sfz", folder)
            out = Path(tmp) / "flat.sfz"
            compare.write_sfz(out, flat["regions"], "test")
            back = compare.read_set(out)
        strip = [{k: v for k, v in r.items()} for r in flat["regions"]]
        self.assertEqual([{**r, "sample": Path(r["sample"])} for r in back],
                         [{**r, "sample": Path(r["sample"])} for r in strip])


class Roles(unittest.TestCase):
    def hits(self, sfz):
        return compare.drum_hits_of_set(compare.read_set(ROOT / "sets" / sfz / f"{sfz}.sfz"))

    @NEEDS
    def test_acoustic_kit_roles(self):
        # The shipped kit is laid out on General MIDI keys, and its files say so: mid_kick (36),
        # mid_snare_center (38; 37 crossstick and 40 rimshot are rims), mid_hh_closed (42),
        # mid_hh_open (46), mid_crash (49), mid_htom (48), mid_ltom (41); no mid tom, so it
        # borrows the high one; the rim is GM's side stick key, 37.
        hits = self.hits("drums-acoustic")
        self.assertTrue(hits[0]["gm"])
        roles = compare.pick_roles(hits, True)
        self.assertEqual(roles, {"kick": 36, "snare": 38, "hat_closed": 42, "hat_open": 46,
                                 "crash": 49, "tom_hi": 48, "tom_mid": 48, "tom_lo": 41,
                                 "rim": 37, "ride": 51})

    @NEEDS
    def test_electronic_kit_is_not_general_midi(self):
        # Keys 24 to 60 by sample list: names decide, and key 49 (hat_mess) is no crash.
        hits = self.hits("drums-electronic")
        self.assertFalse(hits[0]["gm"])
        roles = compare.pick_roles(hits, False)
        self.assertEqual({k: roles.get(k) for k in ("kick", "snare", "hat_closed", "crash")},
                         {"kick": 24, "snare": 38, "hat_closed": 42, "crash": None})

    def test_a_general_midi_kit_named_by_microphone(self):
        # Salamander's names carry its microphones (OH: overheads) and Karoryfer's abbreviate
        # the hi-hat as "ht": on GM keys, full words first, then the key.
        cases = [("ride2_OH_FF_1", 51, "ride"), ("loTom_OH_FF_1", 43, "tom_lo"),
                 ("hihatClosed_OH_F_1", 42, "hat_closed"), ("snareStick_OH_F_1", 41, "rim"),
                 ("crash1_OH_FF_1", 55, "crash"), ("ht_tc_vl7_rr1", 42, "hat_closed"),
                 ("ht_open_vl5_rr1", 46, "hat_open"), ("t18_vl7_rr1", 43, "tom_lo"),
                 ("k_vl11_rr1", 36, "kick"), ("sn_ss_vl4_rr1", 37, "rim")]
        for name, key, role in cases:
            self.assertEqual(compare.kit_role(compare.role_words(name), key, True), role, name)
        # Off GM keys the abbreviations decide (Sonic Pi's bd_ and sn_ at 24 and 39).
        self.assertEqual(compare.kit_role(compare.role_words("bd_808"), 24, False), "kick")
        self.assertEqual(compare.kit_role(compare.role_words("sn_dub"), 39, False), "snare")

    def test_pack_names(self):
        names = ("Kick 808", "Kick 909", "Snare", "Clap", "Hat closed", "Hat open", "Tom low",
                 "Tom mid", "Tom high", "Rim", "Cowbell", "Crash")
        hits = [{"id": n, "role": compare.role_of_file(n), "key": None} for n in names]
        self.assertEqual(compare.pick_roles(hits), {
            "kick": "Kick 808", "snare": "Snare", "hat_closed": "Hat closed",
            "hat_open": "Hat open", "crash": "Crash", "tom_hi": "Tom high", "tom_mid": "Tom mid",
            "tom_lo": "Tom low", "clap": "Clap", "rim": "Rim", "cowbell": "Cowbell"})

    def test_drum_machine_voice_names(self):
        # The TR-808/909 voice abbreviations (BD bass drum, SD snare, CH/OH closed/open hat,
        # CY cymbal, LT/MT/HT toms, RS rimshot, CP clap, CB cowbell, CL claves), in the names
        # the reference packs use; a pack of one voice says it in the library's name.
        cases = {("BD0075", ""): "kick", ("OH25", ""): "hat_open", ("CY0010", ""): "crash",
                 ("Hits/Clap [CP]/E808_CP-10", ""): "clap",
                 ("Hits/Tom Toms [LT-MT-HT]/E808_LT-01", ""): "tom_lo",
                 ("Hits/Tom Toms [LT-MT-HT]/E808_HT-03", ""): "tom_hi",
                 ("HHCD2", "robroy-tr909-chh"): "hat_closed", ("CSHD8", "robroy-tr909-csh"): "crash",
                 ("ST0T0S7", "robroy-tr909-sd"): "snare", ("HANDCLP2", ""): "clap",
                 ("RIM127", ""): "rim", ("Cowbell1_Muted_v3_rr1_Mid", ""): "cowbell",
                 ("susCymb1_hit_fff1", ""): "crash", ("Cassette808_Snr02", ""): "snare",
                 ("Cassette808_HH_01", ""): "hat_closed", ("Cassette808_CL_01", ""): None,
                 ("Cassette808_Cow01", ""): "cowbell", ("Cassette808_Tom02", ""): "tom_mid",
                 ("Cassette808_HHo_01", ""): "hat_open",
                 ("mid_snare_crossstick_vl16", ""): "rim"}
        for (name, library), role in cases.items():
            self.assertEqual(compare.role_of_file(name + ".wav", library), role, (name, library))


@NEEDS
class MessyLibraries(unittest.TestCase):
    """What downloaded libraries hold besides notes: macOS resource forks, files soundfile
    cannot open, tempo loops, thousands of files."""

    def test_forks_and_unreadable_files_are_skipped_and_named(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / "__MACOSX" / "Pads").mkdir(parents=True)
            (folder / "Pads").mkdir()
            sf.write(str(folder / "Pads" / "pad-C3.wav"), sine(130.81, 0.5, 0.5), RATE)
            (folder / "__MACOSX" / "Pads" / "._pad-C3.wav").write_bytes(b"\0\5\26\7" + b"\0" * 60)
            (folder / "Pads" / "._pad-D3.wav").write_bytes(b"\0\5\26\7" + b"\0" * 60)
            (folder / "Pads" / "broken.wav").write_bytes(b"RIFF\0\0\0\0WAVEjunk")
            skipped = []
            files = compare.audio_files(folder, skipped)
        self.assertEqual([p.name for p in files], ["pad-C3.wav"])
        why = {Path(e["file"]).name: e["why"] for e in skipped}
        self.assertEqual(sorted(why), ["._pad-C3.wav", "._pad-D3.wav", "broken.wav"])
        self.assertEqual(why["._pad-D3.wav"], "macOS resource fork")
        self.assertIn("Error", why["broken.wav"])

    def test_tempo_loops_are_not_notes(self):
        loops = ["Loops 85bpm/Mixes/AC_GuitarMix85C-04.wav",
                 "Nasty Leads/Nasty_Lead_Loops/NastyLead_Loop59(130BPM).wav",
                 "00s_Synths/Arps/110bpm/00s_Arp[110]-C.wav",
                 "Bass Synthesis/Basslines/Analogue/BassSynth_Moe-Dee_140_Emin.wav"]
        notes = ["samples/choir-FEMALE-sustain/chorus-female-b4-PB-loop.wav",
                 "Nasty Leads/Multis/Ripped_Squares/Ripped_Squares_F#1.wav",
                 "Bass Synthesis/Multisamples/Analogue/UniSun/BassSynth_UniSun_Multi_C3.wav"]
        self.assertEqual([compare.tempo_loop(p) for p in loops], [True] * 4)
        self.assertEqual([compare.tempo_loop(p) for p in notes], [False] * 3)

    def test_over_200_files_keep_one_per_semitone_nearest_in_tune(self):
        # 21 keys x 10 takes (210 files): take1 of each key is in tune, the others 15 to 40
        # cents off; one file per key stays, and it is take1.
        detune = [25, 0, 15, 30, 40, 20, 35, 15, 25, 30]
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            for key in range(48, 69):
                for take, cents in enumerate(detune):
                    freq = compare.key_frequency(key) * 2 ** (cents / 1200)
                    sf.write(str(folder / f"{compare.note_name(key)}_take{take}.wav"),
                             sine(freq, 0.4, 0.5), RATE)
            log = []
            chosen = compare.pitched_pack_files(compare.audio_files(folder), folder, log)
        self.assertEqual(len(chosen), 21)
        self.assertTrue(all(p.stem.endswith("_take1") for p in chosen), [p.name for p in chosen])
        self.assertTrue(any("one per semitone" in line for line in log), log)

    def test_a_one_drum_class_takes_that_drum_from_a_mixed_pack(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            for name in ("Hits/Bass Drum [BD]/E808_BD-01", "Hits/Bass Drum [BD]/E808_BD-02",
                         "Hits/Snare Drum [SD]/E808_SD-01", "Loops/95bpm/E808_Loop_BD_95-03",
                         "Hits/Claves [CL]/E808_CL-01"):
                (folder / name).parent.mkdir(parents=True, exist_ok=True)
                sf.write(str(folder / f"{name}.wav"), sine(60, 0.3, 0.5), RATE)
            log = []
            hits = compare.drum_pack_hits(compare.audio_files(folder), folder, "kick-808",
                                          "sampleradar-808", log)
            self.assertEqual([Path(h["name"]).name for h in hits], ["E808_BD-01", "E808_BD-02"])
            kit = compare.drum_pack_hits(compare.audio_files(folder), folder, "drums-electronic",
                                         "sampleradar-808", [])
            self.assertEqual(sorted(h["role"] for h in kit), ["kick", "kick", "snare"])
            with self.assertRaises(compare.kitlib.Problem):
                compare.drum_pack_hits(compare.audio_files(folder), folder, "cowbell", "x", [])


@NEEDS_DAWSOME
class HeardSeams(unittest.TestCase):
    """A loop of whole sine periods wraps unheard; the same loop one frame longer repeats a
    frame at every wrap and clicks there. Through Dawsome's render at 48 kHz, from files at
    48 kHz and at 44.1 kHz (resampled on load, so the wrap frames must follow its scaling)."""

    def held(self, rate, freq, start, end):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "set"
            (folder / "samples").mkdir(parents=True)
            sf.write(str(folder / "samples" / "a.wav"), sine(freq, 2.0, 0.5, rate), rate,
                     subtype="FLOAT")
            (folder / "set.sfz").write_text(
                "<region> sample=samples/a.wav lokey=0 hikey=127 pitch_keycenter=60 "
                f"loop_mode=loop_continuous loop_start={start} loop_end={end}\n")
            regions = compare.read_set(folder / "set.sfz")
            renderer = compare.Renderer(BINARY, Path(tmp) / "work")
            x, said = renderer.render(compare.project_json(
                [("instrument", folder / "set.sfz")], [(0, 3 * compare.BAR // 2, 1, 60, 100)], 2),
                "held", 2)
            self.assertEqual((said, renderer.close()), ([], []))
            return compare.heard_seams(x, regions, 60, 100, 0,
                                       compare.frames(3 * compare.BAR // 2))

    def test_at_48k(self):
        # A repeated frame on a sine of amplitude A turns one second difference into the step
        # A w (w = 2 pi f / rate) against the sine's own A w^2 / sqrt 2 RMS: sqrt 2 / w, 10.8
        # at 1 kHz; the loop starts at a zero crossing, where the step is whole.
        good = self.held(48000, 1000, 24000, 24000 + 48 * 250 - 1)
        bad = self.held(48000, 1000, 24000, 24000 + 48 * 250)
        w = 2 * math.pi * 1000 / 48000
        self.assertGreater(good["wraps"], 5)
        self.assertLess(good["click"], 2 * good["click_inside"] + 1)
        self.assertAlmostEqual(bad["click"], math.sqrt(2) / w, delta=0.15 * math.sqrt(2) / w)

    def test_at_44k1_resampled(self):
        # 441 Hz at 44.1 kHz, 100 frames a period. A loop of 14700 frames (147 periods) is
        # 16000 frames at 48 kHz exactly and wraps unheard; one frame longer repeats a frame.
        good = self.held(44100, 441, 22050, 22050 + 14700 - 1)
        bad = self.held(44100, 441, 22050, 22050 + 14700)
        self.assertGreater(good["wraps"], 5)
        self.assertLess(good["click"], 2 * good["click_inside"] + 1)
        self.assertGreater(bad["click"], 5 * bad["click_inside"])

    def test_a_loop_not_whole_at_48k_wraps_on_its_files_frames(self):
        # 12000 frames at 44.1 kHz are 13061.22 at 48 kHz. Dawsome reads such a loop at its
        # file's own rate (dawsome 7c11378; before, it played 13061 and each wrap slipped 0.2245
        # frame, a click of 5.5 here), so it wraps every 12000 file frames: 120 whole periods of
        # the 441 Hz sine, unheard, its first wrap (22050 + 12000) * 48000 / 44100 frames in.
        heard = self.held(44100, 441, 22050, 22050 + 12000 - 1)
        self.assertAlmostEqual(heard["first_wrap_s"], 34050 / 44100, places=9)
        self.assertAlmostEqual(heard["period_s"], 12000 / 44100, places=9)
        self.assertGreater(heard["wraps"], 5)
        self.assertLess(heard["click"], 2 * heard["click_inside"] + 1)


def odd_float_wav(path, m):
    """A mono 32-bit float WAV whose fmt chunk is 20 bytes (cbSize 0 and two more), as the
    vowel choir's files are."""
    data = np.asarray(m, dtype="<f4").tobytes()
    fmt = struct.pack("<HHIIHHH", 3, 1, RATE, RATE * 4, 4, 32, 0) + b"\0\0"
    body = b"WAVE" + b"fmt " + struct.pack("<I", len(fmt)) + fmt + b"data" + struct.pack(
        "<I", len(data)) + data
    Path(path).write_bytes(b"RIFF" + struct.pack("<I", len(body)) + body)


@NEEDS_DAWSOME
class OddWavs(unittest.TestCase):
    def test_a_20_byte_float_fmt_is_rewritten_and_sounds(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            odd = tmp / "odd.wav"
            odd_float_wav(odd, sine(261.6256, 1.0, 0.5))
            self.assertIsNone(compare.unreadable(odd))          # libsndfile reads it
            self.assertFalse(compare.canonical_wav(odd))
            region = dict(compare.REGION_DEFAULTS, sample=odd, pitch_keycenter=60)
            sfz = compare.mirror_set([region], tmp / "set", "test")
            self.assertTrue(compare.canonical_wav(tmp / "set" / "samples" / "0000.wav"))
            renderer = compare.Renderer(BINARY, tmp / "work")
            x, said = renderer.render(compare.project_json(
                [("instrument", sfz)], [(0, 96, 1, 60, 100)], 1), "odd", 1)
            # The file as it is: Dawsome played it as silence, why compare.py rewrites it; since
            # its e8a1406 (2026-10-27) it reads such a fmt chunk's first 16 bytes and plays it
            # as the rewritten file, sample for sample (the rewrite stays for older builds).
            (tmp / "direct.sfz").write_text("<region> sample=odd.wav pitch_keycenter=60\n")
            y, direct = renderer.render(compare.project_json(
                [("instrument", tmp / "direct.sfz")], [(0, 96, 1, 60, 100)], 1), "direct", 1)
            self.assertEqual(renderer.close(), [])
        self.assertEqual(said, [])
        self.assertGreater(float(np.abs(x).max()), 0.1)
        self.assertEqual(direct, [])
        self.assertEqual(float(np.abs(x - y).max()), 0.0)


@NEEDS_DAWSOME
class StandIn(unittest.TestCase):
    """Our piano read as a reference must render exactly as our piano does: the reference half
    reads an SFZ, writes the subset with absolute paths, and Dawsome plays it the same."""

    def test_flattened_piano_renders_as_the_shipped_piano(self):
        folder = ROOT / "sets" / "piano"
        regions = compare.read_set(folder / "piano.sfz")
        with tempfile.TemporaryDirectory() as tmp:
            flat = compare.flatten_sfz(folder / "piano.sfz", folder)
            self.assertEqual(flat["dropped"], {})
            self.assertEqual(flat["ignored"], {})
            generated = compare.mirror_set(flat["regions"], Path(tmp) / "set", "stand-in")
            # Hard links on one filesystem, copies across: the same bytes either way.
            mirrored = sorted(p.stat().st_size for p in (Path(tmp) / "set" / "samples").iterdir())
            self.assertEqual(mirrored, sorted(Path(r["sample"]).stat().st_size
                                              for r in {r["sample"]: r for r in regions}.values()))
            renderer = compare.Renderer(BINARY, Path(tmp) / "work")
            key = compare.root_key(regions)
            notes = [(t, length, 1, k, b) for t, length, k, b in compare.phrase_notes(key)]
            ours, said = renderer.render(compare.project_json([("instrument", folder / "piano.sfz")],
                                                           notes, compare.PHRASE_BARS),
                                      "ours", compare.PHRASE_BARS)
            theirs, said_too = renderer.render(compare.project_json([("instrument", generated)], notes,
                                                             compare.PHRASE_BARS),
                                        "theirs", compare.PHRASE_BARS)
            self.assertEqual(renderer.close(), [])
        self.assertEqual((said, said_too), ([], []))
        self.assertGreater(float(np.abs(ours).max()), 0.1)
        self.assertTrue(np.array_equal(ours, theirs))


@NEEDS_DAWSOME
class RefsRun(unittest.TestCase):
    """compare.py --refs end to end, on our own sets standing in for references: our piano as
    an SFZ library and the starter kit's WAVs as a drum pack play exactly as ours do."""

    def test_stand_in_references(self):
        with tempfile.TemporaryDirectory() as tmp:
            index = Path(tmp) / "index.json"
            index.write_text(json.dumps([
                {"class": "piano", "library": "standin", "folder": str(ROOT / "sets" / "piano"),
                 "format": "sfz", "licence": "CC0-1.0", "tag": "stand-in"},
                {"class": "starter-kit", "library": "pack",
                 "folder": str(compare.ROOT.parent / "dawsome" / "assets" / "sounds"),
                 "format": "wav pack", "licence": "CC0-1.0", "tag": "drums"}]))
            out = Path(tmp) / "out"
            self.assertEqual(compare.main(["--out", str(out), "--dawsome", str(BINARY),
                                           "--refs", str(index), "--only", "piano",
                                           "--only", "starter-kit"]), 0)
            report = json.loads((out / "compare-report.json").read_text())
            self.assertEqual([r["name"] for r in report["refs"]], ["piano/standin", "starter-kit/pack"])
            self.assertEqual(report["refs"][0]["flatten"]["dropped"], {})
            self.assertEqual(report["refs"][0]["pair"], "piano")

            def frames(path):
                return sf.read(str(path), always_2d=True)[0]

            self.assertTrue(np.array_equal(frames(out / "refs" / "piano" / "standin.wav"),
                                           frames(out / "ours" / "piano.wav")))
            self.assertTrue(np.array_equal(
                frames(out / "refs" / "starter-kit" / "pack" / "beat.wav"),
                frames(out / "ours" / "drums" / "starter-kit" / "beat.wav")))
            ours = {p.name[3:]: p for p in (out / "ours" / "drums" / "starter-kit").glob("?? *.wav")}
            refs = {p.name[3:]: p for p in (out / "refs" / "starter-kit" / "pack").glob("?? *.wav")}
            self.assertEqual(sorted(ours), sorted(refs))
            for name in ours:
                self.assertTrue(np.array_equal(frames(ours[name]), frames(refs[name])), name)
            for path in (out / "ours").rglob("*.wav"):
                self.assertAlmostEqual(compare.integrated_loudness(frames(path), RATE), -18.0,
                                       delta=0.01, msg=path)
            text = (out / "compare-report.md").read_text()
            self.assertIn("−18 LUFS", text)
            self.assertIn("### Drums beside theirs, role by role", text)


@NEEDS_DAWSOME
class FailingLibraries(unittest.TestCase):
    """A run over an index where libraries fail goes on: each failure is reported with its
    error, the rest render, and LISTEN.md pairs each class with ours by the index's own rows."""

    def test_failures_are_reported_and_the_run_goes_on(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            good, junk = tmp / "good", tmp / "junk"
            good.mkdir()
            (junk / "__MACOSX").mkdir(parents=True)
            for key in (48, 52, 55, 60):
                sf.write(str(good / f"pluck_{compare.note_name(key)}.wav"),
                         sine(compare.key_frequency(key), 1.0, 0.5)
                         * np.exp(-np.arange(RATE) / (0.3 * RATE)), RATE)
            (junk / "__MACOSX" / "._a.wav").write_bytes(b"\0\5\26\7" + b"\0" * 60)
            (junk / "bad.wav").write_bytes(b"RIFF\0\0\0\0WAVEjunk")
            index = tmp / "index.json"
            index.write_text(json.dumps([
                {"class": "pluck", "library": "dawsome", "status": "ours", "tag": "cc0",
                 "folder": str(compare.ROOT.parent / "dawsome" / "assets" / "instruments"
                               / "pluck-c4")},
                {"class": "pluck", "library": "junk", "status": "downloaded", "tag": "reference",
                 "folder": str(junk), "format": "WAV"},
                {"class": "pluck", "library": "gone", "status": "downloaded", "tag": "cc0",
                 "folder": str(tmp / "nowhere"), "format": "SFZ + WAV"},
                {"class": "pluck", "library": "never", "status": "skipped", "tag": "reference"},
                {"class": "pluck", "library": "good", "status": "downloaded", "tag": "cc0",
                 "folder": str(good), "format": "WAV"}]))
            out = tmp / "out"
            self.assertEqual(compare.main(["--out", str(out), "--dawsome", str(BINARY),
                                           "--refs", str(index), "--only", "pluck-c4",
                                           "--only", "pluck"]), 0)
            report = json.loads((out / "compare-report.json").read_text())
            refs = {r["library"]: r for r in report["refs"]}
            self.assertEqual(sorted(refs), ["gone", "good", "junk"])
            self.assertIn("no readable sound files", refs["junk"]["problem"])
            self.assertEqual(len(refs["junk"]["skipped"]), 2)
            self.assertIn("is not a folder", refs["gone"]["problem"])
            self.assertEqual(refs["good"]["pair"], "pluck-c4")
            self.assertTrue((out / "refs" / "pluck" / "good.wav").is_file())
            listen = (out / "LISTEN.md").read_text()
        self.assertIn("## pluck", listen)
        self.assertIn("ours: `ours/pluck-c4.wav`", listen)
        self.assertIn("- good (cc0): `refs/pluck/good.wav`", listen)
        self.assertIn("- junk (reference): not rendered", listen)
        self.assertIn("- not downloaded: never", listen)
        self.assertIn("**pluck**", listen)
        self.assertIn("pluck C4 (60)", listen)


if __name__ == "__main__":
    unittest.main()
