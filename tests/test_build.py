# SPDX-License-Identifier: GPL-3.0-or-later
"""Known answers for build.py's own sample work and SFZ writing. Expected values come from the
sources (Virtuosity Drums splits its four kick layers at velocities 31, 63 and 95), from
arithmetic done by hand, from signals whose correct result follows from their construction,
and from metaflac's listing of fixtures/audio/ramp.flac."""

import math
import shutil
import tempfile
import unittest
import wave
from array import array
from pathlib import Path

import build
import kitlib

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def sine(frames, period, channels=1, level=0.5):
    """`frames` frames of a sine of `period` frames, the same on every channel, interleaved."""
    return array("f", (level * math.sin(2 * math.pi * frame / period)
                       for frame in range(frames) for _ in range(channels)))


class Splits(unittest.TestCase):
    def test_velocity_splits_match_the_sources_own(self):
        # Virtuosity Drums' kick map: hivel=31, 32-63, 64-95, lovel=96.
        self.assertEqual(build.velocity_splits(4), [(1, 31), (32, 63), (64, 95), (96, 127)])
        self.assertEqual(build.velocity_splits(2), [(1, 63), (64, 127)])
        self.assertEqual(build.velocity_splits(1), [(1, 127)])

    def test_spread_gives_each_key_its_nearest_root(self):
        # Roots C2 (36) and E2 (40): 37 and 38 are nearer 36, 39 is nearer 40.
        self.assertEqual(build.spread([36, 40], 31, 54), [(31, 38), (39, 54)])
        self.assertEqual(build.spread([60], 60, 60), [(60, 60)])
        self.assertEqual(build.spread([36, 37, 38], 36, 38), [(36, 36), (37, 37), (38, 38)])


class Crossfade(unittest.TestCase):
    def test_identical_stretches_fade_linearly(self):
        for keep, take in build.crossfade_gains(100, 1.0):
            self.assertAlmostEqual(keep + take, 1.0)

    def test_unrelated_stretches_keep_their_power(self):
        for keep, take in build.crossfade_gains(100, 0.0):
            self.assertAlmostEqual(keep ** 2 + take ** 2, 1.0)

    def test_anti_correlated_stretches_count_as_unrelated(self):
        self.assertEqual(build.crossfade_gains(10, -0.7), build.crossfade_gains(10, 0.0))

    def test_correlation(self):
        signal = sine(400, 100)
        self.assertAlmostEqual(build.correlation(signal, signal), 1.0, places=6)
        self.assertAlmostEqual(build.correlation(signal, array("f", (-x for x in signal))),
                               -1.0, places=6)
        self.assertEqual(build.correlation(signal, array("f", [0.0] * 400)), 0.0)


class Shaped(unittest.TestCase):
    def test_a_loop_of_whole_periods_leaves_a_steady_sine_unchanged(self):
        # A 1000 Hz rate and a 100-frame period: the loop (whole periods) and its lead-in are
        # the same signal, so the crossfade changes nothing and the jump back is seamless. The
        # file runs 10 frames (0.01 s) past the loop's end.
        rate, period = 1000, 100
        samples = sine(6000, period, channels=2)
        out, (start, end) = build.shaped(samples, 2, rate,
                                         build.Shape(loop=(1.0, 3.0, 2.0, 0.5)))
        self.assertEqual((end + 1 - start) % period, 0)
        self.assertEqual(len(out), (end + 1 + 10) * 2)
        for index, value in enumerate(out):
            self.assertAlmostEqual(value, samples[index], places=5)
        self.assertAlmostEqual(out[end * 2], samples[(start - 1) * 2], places=5)

    def test_the_loop_runs_past_its_end_with_its_own_start(self):
        # Two sines whose sum never repeats within the sample: what follows the loop's end in the
        # recording differs from the loop's start, and the 10 frames (0.01 s at 1000 Hz) the file
        # runs past loop_end must be the loop's start, which is what playback reads next.
        rate = 1000
        samples = array("f", (0.3 * math.sin(0.37 * frame) + 0.2 * math.sin(1.91 * frame)
                              for frame in range(6000) for _ in range(2)))
        out, (start, end) = build.shaped(samples, 2, rate, build.Shape(loop=(1.0, 3.0, 2.0, 0.5)))
        self.assertEqual(len(out), (end + 1 + 10) * 2)
        self.assertEqual(out[(end + 1) * 2:], out[start * 2:(start + 10) * 2])
        self.assertEqual(out[start * 2:(start + 10) * 2], samples[start * 2:(start + 10) * 2])
        self.assertNotEqual(out[(end + 1) * 2:], samples[(end + 1) * 2:(end + 11) * 2])

    def test_the_loop_starts_where_the_level_holds(self):
        # A swell over the first 2.5 s, then a steady tone. Every window with any of the swell
        # in it varies, so the loop's lead-in (0.5 s) must start at 2.5 s at the earliest: the
        # first steady start is 3.0 s, frame 3000.
        rate = 1000
        swell = array("f", ((0.1 + 0.7 * frame / 2500) * math.sin(2 * math.pi * frame / 100)
                            for frame in range(2500)))
        steady = sine(3500, 100, level=0.4)
        out, (start, end) = build.shaped(swell + steady, 1, rate,
                                         build.Shape(loop=(1.0, 4.0, 1.0, 0.5)))
        self.assertEqual(start, 3000)
        self.assertEqual(end - start + 1, 1000)

    def test_a_loop_that_does_not_fit_is_refused(self):
        with self.assertRaises(kitlib.Problem):
            build.shaped(sine(2000, 100), 1, 1000, build.Shape(loop=(1.0, 1.0, 2.0, 0.5)))

    def test_the_tail_is_cut_after_the_last_loud_frame(self):
        # 1000 frames at 0.5, then 1000 at 1e-5 (94 dB under the peak): with a 60 dB floor the
        # sound ends at frame 1000, then a 10-frame fade (0.01 s at 1000 Hz) to silence.
        samples = array("f", [0.5] * 1000 + [1e-5] * 1000)
        out, loop = build.shaped(samples, 1, 1000, build.Shape(floor_db=-60, fade_seconds=0.01))
        self.assertIsNone(loop)
        self.assertEqual(len(out), 1010)
        self.assertEqual(out[999], 0.5)
        self.assertEqual(out[-1], 0.0)

    def test_a_cap_fades_out_to_its_end(self):
        samples = array("f", [0.5] * 2000)
        out, _ = build.shaped(samples, 1, 1000, build.Shape(max_seconds=1.0, fade_seconds=0.1))
        self.assertEqual(len(out), 1000)
        self.assertEqual(out[899], 0.5)
        self.assertAlmostEqual(out[900], 0.5 * 99 / 100)
        self.assertEqual(out[999], 0.0)

    def test_a_late_start_is_cut_with_a_short_fade_in(self):
        # 100 frames of noise at 1e-4 (74 dB under the 0.5 peak), then the note: the first frame
        # within 20 dB of the peak is frame 100, so with a 10-frame lead (0.01 s at 1000 Hz) the
        # sample starts at frame 90, fading in from 0 over frames 90..99.
        samples = array("f", [1e-4] * 100 + [0.5] * 900)
        out, loop = build.shaped(samples, 1, 1000, build.Shape(start_db=-20, start_seconds=0.01))
        self.assertIsNone(loop)
        self.assertEqual(len(out), 910)
        self.assertEqual(out[0], 0.0)
        self.assertAlmostEqual(out[5], 1e-4 * 5 / 10)
        self.assertAlmostEqual(out[9], 1e-4 * 9 / 10)
        self.assertEqual(out[10], 0.5)
        self.assertEqual(out[-1], 0.5)

    def test_a_start_already_within_the_lead_is_left_alone(self):
        samples = array("f", [1e-4] * 5 + [0.5] * 95)
        out, _ = build.shaped(samples, 1, 1000, build.Shape(start_db=-20, start_seconds=0.01))
        self.assertEqual(out, samples)

    def test_the_start_follows_the_loudest_channel(self):
        # Stereo: 50 silent frames, then the note on the right channel only (left stays 0).
        samples = array("f", [0.0, 0.0] * 50 + [0.0, 0.5] * 50)
        out, _ = build.shaped(samples, 2, 1000, build.Shape(start_db=-20, start_seconds=0.01))
        self.assertEqual(len(out), 60 * 2)
        self.assertEqual(out[10 * 2 + 1], 0.5)
        self.assertEqual(out[9 * 2 + 1], 0.0)

    def test_a_cap_counts_from_the_new_start(self):
        samples = array("f", [1e-4] * 100 + [0.5] * 2000)
        out, _ = build.shaped(samples, 1, 1000, build.Shape(
            start_db=-20, start_seconds=0.01, max_seconds=1.0, fade_seconds=0.1))
        self.assertEqual(len(out), 1000)
        self.assertEqual(out[10], 0.5)
        self.assertEqual(out[899], 0.5)
        self.assertEqual(out[999], 0.0)

    def test_a_start_cut_and_a_loop_are_refused_together(self):
        with self.assertRaises(kitlib.Problem):
            build.shaped(sine(5000, 100), 1, 1000,
                         build.Shape(start_db=-20, loop=(1.0, 2.0, 1.0, 0.5)))

    def test_a_short_fade_keeps_the_rest_of_the_lead(self):
        # As the late-start case, but the 10-frame lead fades in over its first 4 frames only:
        # frames 90..93 rise 0, 1/4, 2/4, 3/4 of the noise, 94..99 keep it as recorded.
        samples = array("f", [1e-4] * 100 + [0.5] * 900)
        out, _ = build.shaped(samples, 1, 1000, build.Shape(
            start_db=-20, start_seconds=0.01, start_fade_seconds=0.004))
        self.assertEqual(len(out), 910)
        self.assertEqual(out[0], 0.0)
        self.assertAlmostEqual(out[1], 1e-4 / 4)
        self.assertAlmostEqual(out[3], 1e-4 * 3 / 4)
        self.assertAlmostEqual(out[4], 1e-4)
        self.assertAlmostEqual(out[9], 1e-4)
        self.assertEqual(out[10], 0.5)

    @staticmethod
    def note_into_hiss(hiss):
        """1 s of a note (0.5, alternating sign: every 50 ms window's RMS is 0.5) at 1000 Hz,
        then 2 s of a constant `hiss` level the same way."""
        return array("f", [0.5 * (-1) ** frame for frame in range(1000)]
                     + [hiss * (-1) ** frame for frame in range(2000)])

    def test_a_hiss_tail_fades_into_the_cut(self):
        # The hiss, 0.0025, is 46.02 dB under the 0.5 peak (0.005 of it); the cut's -60 dB is 0.001
        # of it, 0.2 of the hiss. From the first window of hiss (frame 1000) to the end the gain
        # falls linearly in dB from 1 to 0.2: sqrt(0.2) halfway (frame 2000). The note before it
        # is untouched, and the sample keeps its length (the hiss never stays under -60 dB, so the
        # cut is at the end, its own 50 ms fade-out on top).
        samples = self.note_into_hiss(0.0025)
        out, _ = build.shaped(samples, 1, 1000, build.Shape(floor_db=-60, hiss_fade=True))
        self.assertEqual(len(out), 3000)
        self.assertEqual(out[:1000], samples[:1000])
        self.assertAlmostEqual(out[1000], samples[1000], places=7)
        self.assertAlmostEqual(out[2000], samples[2000] * math.sqrt(0.2), places=6)
        self.assertAlmostEqual(out[2949], samples[2949] * 0.2 ** (1949 / 2000), places=6)

    def test_a_tail_already_under_the_cuts_level_is_left_alone(self):
        # Hiss at 0.0002, 68 dB under the peak: under the cut's -60 dB, so no fade; the cut alone.
        samples = self.note_into_hiss(0.0002)
        plain, _ = build.shaped(samples, 1, 1000, build.Shape(floor_db=-60))
        faded, _ = build.shaped(samples, 1, 1000, build.Shape(floor_db=-60, hiss_fade=True))
        self.assertEqual(faded, plain)

    def test_a_hiss_fade_needs_a_floor(self):
        with self.assertRaises(kitlib.Problem):
            build.shaped(self.note_into_hiss(0.0025), 1, 1000, build.Shape(hiss_fade=True))

    def test_a_lead_of_silence_is_cut_to_its_last_frame_before_the_sound(self):
        # 50 frames of silence, 5 at 0.001 (54 dB under the 0.5 peak), then the note at frame 55.
        # The start cut keeps 15 frames before the note (from frame 40); the silence rule keeps at
        # most 1 frame before the first frame within 60 dB of the peak (frame 50), so the sample
        # starts at frame 49, that 1 frame faded in (0 at its start), the rest as recorded.
        samples = array("f", [0.0] * 50 + [0.001] * 5 + [0.5] * 945)
        out, _ = build.shaped(samples, 1, 1000, build.Shape(
            start_db=-20, start_seconds=0.015, start_fade_seconds=0.002, silence=(-60, 0.001)))
        self.assertEqual(len(out), 951)
        self.assertEqual(out[0], 0.0)
        self.assertEqual(list(out[1:6]), [samples[50]] * 5)
        self.assertEqual(out[6:], samples[55:])

    def test_a_lead_that_starts_on_the_sound_keeps_the_start_cut(self):
        # The 15-frame lead before the note holds no silence (0.001 from frame 0): the silence rule
        # moves nothing, and the start cut fades in over its first 2 frames as before.
        samples = array("f", [0.001] * 100 + [0.5] * 900)
        out, _ = build.shaped(samples, 1, 1000, build.Shape(
            start_db=-20, start_seconds=0.015, start_fade_seconds=0.002, silence=(-60, 0.001)))
        self.assertEqual(len(out), 915)
        self.assertEqual(out[0], 0.0)
        self.assertAlmostEqual(out[1], 0.001 / 2)
        self.assertEqual(out[2:], samples[87:])

    def test_a_silence_lead_without_a_start_cut_is_refused(self):
        with self.assertRaises(kitlib.Problem):
            build.shaped(sine(5000, 100), 1, 1000, build.Shape(silence=(-60, 0.001)))

    def test_a_note_within_its_lead_is_still_cut_to_its_silence(self):
        # The note 10 frames in, inside the 15-frame lead: the start cut leaves it alone, but the
        # 9 silent frames before its first loud frame are cut to 1, faded in.
        samples = array("f", [0.0] * 10 + [0.5] * 90)
        out, _ = build.shaped(samples, 1, 1000, build.Shape(
            start_db=-20, start_seconds=0.015, start_fade_seconds=0.002, silence=(-60, 0.001)))
        self.assertEqual(list(out), [0.0] + [0.5] * 90)

    def test_an_onset_fades_the_lead_in_and_puts_silence_before_it(self):
        # 100 frames at 0.04 (a source that starts mid-wave, 22 dB under the 0.5 peak), then the
        # note. At 1000 Hz: 5 frames of silence (0.005 s), then the 0.04 fades in over 20 frames
        # (0.02 s, the note being 100 frames away): 0, 1/20, ... 19/20 of it, then as recorded.
        samples = array("f", [0.04] * 100 + [0.5] * 900)
        out, loop = build.shaped(samples, 1, 1000, build.Shape(onset=(0.005, 0.02, -20)))
        self.assertIsNone(loop)
        self.assertEqual(len(out), 1005)
        self.assertEqual(list(out[:6]), [0.0] * 6)
        self.assertAlmostEqual(out[6], 0.04 / 20)
        self.assertAlmostEqual(out[15], 0.04 * 10 / 20)
        self.assertAlmostEqual(out[24], 0.04 * 19 / 20)
        self.assertEqual(out[25:105], samples[20:100])
        self.assertEqual(out[105:], samples[100:])

    def test_an_onset_fade_stops_where_the_note_comes_within_its_level(self):
        # The note 8 frames in: the fade covers those 8 frames only, the note is as recorded.
        samples = array("f", [-0.04] * 8 + [0.5] * 92)
        out, _ = build.shaped(samples, 1, 1000, build.Shape(onset=(0.005, 0.02, -20)))
        self.assertEqual(len(out), 105)
        self.assertEqual(list(out[:6]), [0.0] * 6)
        self.assertAlmostEqual(out[9], -0.04 * 4 / 8)
        self.assertAlmostEqual(out[12], -0.04 * 7 / 8)
        self.assertEqual(out[13:], samples[8:])

    def test_an_onset_on_a_sample_that_starts_on_its_note_is_refused(self):
        with self.assertRaises(kitlib.Problem):
            build.shaped(array("f", [0.5] * 100), 1, 1000, build.Shape(onset=(0.005, 0.02, -20)))

    def test_an_onset_with_a_loop_or_a_start_cut_is_refused(self):
        for shape in (build.Shape(onset=(0.005, 0.02, -20), loop=(1.0, 2.0, 1.0, 0.5)),
                      build.Shape(onset=(0.005, 0.02, -20), start_db=-20)):
            with self.subTest(shape), self.assertRaises(kitlib.Problem):
                build.shaped(sine(5000, 100), 1, 1000, shape)

    def test_a_tone_and_a_loop_are_refused_together(self):
        with self.assertRaises(kitlib.Problem):
            build.shaped(sine(5000, 100), 1, 1000, build.Shape(
                loop=(1.0, 2.0, 1.0, 0.5), tone=build.Tone("volume=0dB", "nothing")))


class Retuning(unittest.TestCase):
    def test_an_octave_doubles_or_halves_the_relabelled_rate(self):
        self.assertIsNone(build.retuning(44100, 0))
        self.assertTrue(build.retuning(44100, 1200).startswith("asetrate=88200,aresample=44100:"))
        self.assertTrue(build.retuning(44100, -1200).startswith("asetrate=22050,aresample=44100:"))

    def test_the_whole_number_rate_is_within_two_hundredths_of_a_cent(self):
        # 44100 * 2^(13/1200) = 44432.39..., relabeled 44432: 12.985 cents.
        self.assertTrue(build.retuning(44100, 13).startswith("asetrate=44432,"))
        for cents in range(-30, 31):
            if cents:
                rate = int(build.retuning(44100, cents).split(",")[0].split("=")[1])
                self.assertLess(abs(1200 * math.log2(rate / 44100) - cents), 0.02)


def write_wav(path, rate, samples, channels=1):
    """A 16-bit WAV of `samples` (floats in -1..1, interleaved when there are more channels)."""
    with wave.open(str(path), "wb") as file:
        file.setnchannels(channels)
        file.setsampwidth(2)
        file.setframerate(rate)
        file.writeframes(array("h", (round(value * 32767) for value in samples)).tobytes())


@unittest.skipUnless(shutil.which("ffmpeg"), "needs ffmpeg")
class WithFfmpeg(unittest.TestCase):
    def test_a_sine_raised_an_octave_has_half_its_period(self):
        # 441 Hz at 44100 Hz is a period of 100 frames; an octave up, 50: 2 s of it crosses zero
        # upwards 882 times before and 1764 times after (one per period, the ends aside).
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "sine.wav"
            write_wav(path, 44100, [0.5 * math.sin(2 * math.pi * 441 * n / 44100)
                                    for n in range(2 * 44100)])
            ffmpeg = shutil.which("ffmpeg")
            plain = build.decode(ffmpeg, path)
            raised = build.decode(ffmpeg, path, build.retuning(44100, 1200))

        def upward(samples):
            body = samples[2000:-2000]
            return sum(1 for a, b in zip(body, body[1:]) if a < 0 <= b)
        self.assertEqual(len(raised), len(plain) // 2)
        self.assertAlmostEqual(upward(plain), (88200 - 4000) / 100, delta=1)
        self.assertAlmostEqual(upward(raised), (44100 - 4000) / 50, delta=1)

    def test_a_16_bit_sample_with_only_an_onset_keeps_every_sample_after_the_fade(self):
        # Through make_sample and ffmpeg: a 16-bit WAV that starts mid-wave, 200 frames at 0.04
        # (22 dB under the note's 0.495 peak), then a decaying note. At 44.1 kHz the onset puts 88
        # frames (2 ms) of silence first and fades the first 88 frames in; every frame after those
        # is the WAV's own, bit for bit (rounded, not dithered).
        lead = [0.04] * 200
        note = [0.5 * math.sin(0.05 * n) * math.exp(-n / 3000) for n in range(4000)]
        with tempfile.TemporaryDirectory() as folder:
            source, target = Path(folder) / "in.wav", Path(folder) / "out.flac"
            write_wav(source, 44100, lead + note)
            zone = build.Zone("in.wav", {}, shape=build.Shape(onset=(0.002, 0.002, -20)))
            ffmpeg = shutil.which("ffmpeg")
            made, loop, peak = build.make_sample(ffmpeg, source, target, zone, None)
            original, out = build.decode(ffmpeg, source), build.decode(ffmpeg, target)
        self.assertEqual((made, loop, peak), ("processed", None, None))
        self.assertEqual(len(out), 88 + 4200)
        self.assertEqual(list(out[:89]), [0.0] * 89)
        self.assertEqual(out[88 + 88:], original[88:])
        self.assertAlmostEqual(out[88 + 44], original[44] / 2, delta=1 / 32768)

    def test_one_channel_of_a_stereo_sample_is_kept_as_a_mono_file(self):
        # hat_gnu's shape: the right channel is the left upside down and a little quieter, so the
        # two cancel when summed to mono. Keeping channel 0, with the onset the electronic drums
        # get (88 frames of silence, an 88-frame fade at 44.1 kHz), gives a mono file whose
        # every frame after the fade is the left channel's, bit for bit.
        left = [0.04] * 200 + [0.5 * math.sin(0.05 * n) * math.exp(-n / 3000) for n in range(4000)]
        stereo = [value for sample in left for value in (sample, -0.9 * sample)]
        with tempfile.TemporaryDirectory() as folder:
            source, target = Path(folder) / "in.wav", Path(folder) / "out.flac"
            write_wav(source, 44100, stereo, channels=2)
            zone = build.Zone("in.wav", {}, shape=build.Shape(onset=(0.002, 0.002, -20)),
                              channel=0)
            ffmpeg = shutil.which("ffmpeg")
            made, loop, peak = build.make_sample(ffmpeg, source, target, zone, None)
            original, out = build.decode(ffmpeg, source), build.decode(ffmpeg, target)
            info = kitlib.audio_info(target)
        self.assertEqual((made, loop, peak), ("processed", None, None))
        self.assertEqual((info.channels, info.frames), (1, 88 + 4200))
        self.assertEqual(list(out[:89]), [0.0] * 89)
        self.assertEqual(out[88 + 88:], original[0::2][88:])

    def test_a_channel_the_sample_lacks_is_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "in.wav"
            write_wav(source, 44100, [0.5] * 100)
            zone = build.Zone("in.wav", {}, channel=1)
            with self.assertRaises(kitlib.Problem):
                build.make_sample(shutil.which("ffmpeg"), source, Path(folder) / "out.flac", zone,
                                  None)

    def test_momentary_loudness_reads_a_sine_as_the_standard_says(self):
        # ITU-R BS.1770: a full-scale 997 Hz sine in one channel reads -3.01 LUFS, so one of
        # amplitude 0.5 reads 10 log10(0.5^2 / 2) = -9.03.
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "sine.wav"
            write_wav(path, 48000, [0.5 * math.sin(2 * math.pi * 997 * n / 48000)
                                    for n in range(2 * 48000)])
            measured = build.momentary_loudness(shutil.which("ffmpeg"), path)
        self.assertAlmostEqual(measured, -9.03, delta=0.1)


class SourceRegions(unittest.TestCase):
    def test_key_and_note_names_become_numbers(self):
        sample, opcodes, lowpass = build.subset_region(
            {"sample": "x.flac", "key": "c4", "lovel": "1"}, "test")
        self.assertEqual((sample, lowpass), ("x.flac", None))
        self.assertEqual(opcodes, {"lokey": "60", "hikey": "60", "pitch_keycenter": "60",
                                   "lovel": "1"})

    def test_a_fixed_lowpass_is_kept_to_bake_and_a_default_width_dropped(self):
        _, opcodes, lowpass = build.subset_region(
            {"sample": "x.flac", "lokey": "21", "fil_type": "lpf_2p", "cutoff": "2500",
             "width": "100"}, "test")
        self.assertEqual((opcodes, lowpass), ({"lokey": "21"}, 2500.0))

    def test_anything_else_outside_the_subset_is_refused(self):
        for region in ({"sample": "x", "width": "50"}, {"sample": "x", "amp_velcurve_1": "1"},
                       {"sample": "x", "fil_type": "hpf_2p", "cutoff": "100"}):
            with self.subTest(region), self.assertRaises(kitlib.Problem):
                build.subset_region(region, "test")


class LoopEnds(unittest.TestCase):
    """SFZ and the DAW read loop_end as a loop's last frame; FreePats writes it as the frame after
    the loop, and its files repeat the loop's first frames from there."""

    PERIOD = [0.5 * math.sin(2 * math.pi * frame / 100) for frame in range(100)]

    def test_an_end_written_as_the_frame_after_the_loop_repeats_its_start(self):
        # Eight copies of one 100-frame period, cut at frame 716: a loop of five periods from
        # frame 200 has frame 699 as its last, and frames 700 to 715 are frames 200 to 215 again,
        # as in FreePats' files. loop_end=700 plays frame 200 twice at every wrap.
        samples = array("f", (self.PERIOD * 8)[:716])
        self.assertTrue(build.loop_end_repeats_start(samples, 1, 200, 700))
        self.assertFalse(build.loop_end_repeats_start(samples, 1, 200, 699))

    def test_a_repeat_a_few_steps_off_still_counts(self):
        # The piano's and the choir pad's files repeat their loop's start to within a few 16-bit
        # steps, not bit for bit: two steps of noise on every frame from 700 on.
        samples = array("f", (self.PERIOD * 8)[:716])
        for frame in range(700, 716):
            samples[frame] += (2 if frame % 2 else -2) / 32768
        self.assertTrue(build.loop_end_repeats_start(samples, 1, 200, 700))
        self.assertFalse(build.loop_end_repeats_start(samples, 1, 200, 699))

    def test_a_loop_ending_on_the_files_last_frame(self):
        # Frame 700 is the file's last and frame 200 again: nothing follows it to compare.
        samples = array("f", (self.PERIOD * 8)[:701])
        self.assertTrue(build.loop_end_repeats_start(samples, 1, 200, 700))
        self.assertFalse(build.loop_end_repeats_start(samples, 1, 200, 699))

    def test_build_pys_own_loops_run_on_with_their_start(self):
        # The strings' loops: the file holds the loop's start after its end, so loop_end is the
        # loop's last frame and one frame more would be its first again.
        samples = array("f", (0.3 * math.sin(0.37 * frame) + 0.2 * math.sin(1.91 * frame)
                              for frame in range(6000) for _ in range(2)))
        out, (start, end) = build.shaped(samples, 2, 1000, build.Shape(loop=(1.0, 3.0, 2.0, 0.5)))
        self.assertFalse(build.loop_end_repeats_start(out, 2, start, end))
        self.assertTrue(build.loop_end_repeats_start(out, 2, start, end + 1))

    def test_a_source_that_writes_the_frame_after_has_its_ends_written_less_one(self):
        with tempfile.TemporaryDirectory() as folder:
            sfz = Path(folder) / "demo" / "extracted" / "Demo-1" / "demo.sfz"
            sfz.parent.mkdir(parents=True)
            sfz.write_text("<global> loop_mode=loop_continuous\n"
                           "<region> sample=samples/C4.wav key=60 loop_start=100 loop_end=900\n"
                           "<region> sample=samples/D4.wav key=62 loop_mode=no_loop\n")
            plain = build.from_source_sfz(Path(folder), "demo", "Demo", "demo")
            after = build.from_source_sfz(Path(folder), "demo", "Demo", "demo",
                                          exclusive_loop_end=True)
        self.assertEqual((plain.zones[0].opcodes["loop_start"], plain.zones[0].opcodes["loop_end"]),
                         ("100", "900"))
        self.assertEqual((after.zones[0].opcodes["loop_start"], after.zones[0].opcodes["loop_end"]),
                         ("100", "899"))
        self.assertNotIn("loop_end", after.zones[1].opcodes)


class OrganVelocity(unittest.TestCase):
    """A drawbar organ plays at one level however hard a key is struck. Dawsome's gain for a
    velocity byte b at amp_veltrack t percent is 1 - t/100 (1 - (b/127)^2) (its arc.md, the
    instrument zones): at t = 100, velocity 40 to 120 swings 20 log10(9) = 19.08 dB."""

    @staticmethod
    def gain(byte, veltrack):
        return 1 - veltrack / 100 * (1 - (byte / 127) ** 2)

    def test_the_formula_gives_the_swing_measured_before(self):
        self.assertAlmostEqual(20 * math.log10(self.gain(120, 100) / self.gain(40, 100)), 19.08,
                               places=2)

    def test_the_shipped_organ_swings_at_most_6_db_from_velocity_40_to_120(self):
        sfz = kitlib.read_sfz(kitlib.SETS / "organ" / "organ.sfz")
        for region in sfz.regions:
            veltrack = float(region.get("amp_veltrack", 100))
            swing = 20 * math.log10(self.gain(120, veltrack) / self.gain(40, veltrack))
            self.assertLessEqual(swing, 6.0, region["sample"])

    def test_the_shipped_organ_plays_c3_as_loud_as_the_other_sets_median(self):
        # compare.py, 2026-10-27, C3 velocity 100 held 4 s through dawsome render: the 18 other
        # pitched sets Dawsome ships (these and its electronic sets) have a median of -18.6 LUFS
        # (-18.71 and -18.52 either side of it); the organ, at full level since amp_veltrack=0,
        # played -9.2.
        compare = _compare_or_skip(self)
        sfz = kitlib.SETS / "organ" / "organ.sfz"
        with tempfile.TemporaryDirectory() as folder:
            renderer = compare.Renderer(compare.default_binary(), Path(folder))
            try:
                notes = [(0, compare.HOLD_BARS * compare.BAR, 1, 48, compare.MATCH_VELOCITY)]
                x, _ = renderer.render(compare.project_json([("instrument", sfz)], notes,
                                                            compare.PROBE_SECTION_BARS),
                                       "organ", compare.PROBE_SECTION_BARS)
            finally:
                renderer.close()
        self.assertAlmostEqual(compare.integrated_loudness(x, compare.RATE), -18.6, delta=0.2)


def shipped_samples(name):
    """{sample file name: its MANIFEST.json entry} of the shipped set `name`."""
    prefix = f"sets/{name}/samples/"
    return {entry["path"][len(prefix):]: entry
            for entry in kitlib.read_json(kitlib.MANIFEST)["files"]
            if entry["path"].startswith(prefix)}


def first_frame_steps(path):
    """The first frame's largest channel, in 16-bit steps."""
    samples = build.decode(shutil.which("ffmpeg"), path)
    channels = kitlib.audio_info(path).channels
    return max(abs(value) for value in samples[:channels]) * 32768


class ElectricPiano(unittest.TestCase):
    """The ep set is FreePats' FM Piano 1 as its own .sfz maps it ("FM-Piano1 20190916.sfz": per
    layer 12 regions rooted 30 to 96, six keys apart, each from 3 keys under its root to 2 over;
    the layers hivel=69, then lovel=70 hivel=109, then lovel=110), every sample traced to the
    archive FreePats' page links (the SHA-256 fetch.py pinned, the one the reference research's
    own download had on 2026-10-27), each starting from silence as the source's do (their first
    frame is 0; dither to 16 bits adds at most 2 steps)."""

    ARCHIVE = ("https://github.com/freepats/fm-piano1/releases/download/2019-09-16/"
               "FM-Piano1-SFZ+FLAC-20190916.7z")
    SHA256 = "fa308cead617211b29f500db2f962f34e18c9cd55a94ae9176e1d63babeab083"
    FOLDER = kitlib.SETS / "ep"

    def test_the_mapping_is_freepats_own(self):
        layers = ((None, "69", "v60"), ("70", "109", "v80"), ("110", None, "v100"))
        expected = [(str(root - 3), str(root + 2), str(root), low, high,
                     f"{'C' if root % 12 == 0 else 'F#'}{root // 12 - 1}{suffix}.flac")
                    for low, high, suffix in layers for root in range(30, 97, 6)]
        sfz = kitlib.read_sfz(self.FOLDER / "ep.sfz")
        found = [(r["lokey"], r["hikey"], r["pitch_keycenter"], r.get("lovel"), r.get("hivel"),
                  r["sample"]) for r in sfz.regions]
        self.assertEqual(found, expected)
        self.assertEqual({r.get("ampeg_release") for r in sfz.regions}, {"0.4"})
        self.assertEqual({r.get("amp_veltrack") for r in sfz.regions}, {None})

    def test_the_set_passes_the_kits_checks(self):
        import check
        self.assertEqual(check.check_set(self.FOLDER, kitlib.ROOT), [])
        about = kitlib.read_json(self.FOLDER / "set.json")
        self.assertEqual((about["name"], about["category"], about["keys"]),
                         ("Electric Piano", "Keys", [27, 98]))

    def test_every_sample_traces_to_the_pinned_archive(self):
        entries = shipped_samples("ep")
        self.assertEqual(len(entries), 36)
        for name, entry in entries.items():
            self.assertEqual((entry["origin"], entry["origin_sha256"], entry["member"],
                              entry["licence"]),
                             (self.ARCHIVE, self.SHA256,
                              f"FM-Piano1 SFZ+FLAC-20190916/samples/{name}", "CC0-1.0"), name)

    @unittest.skipUnless(shutil.which("ffmpeg"), "needs ffmpeg")
    def test_every_sample_starts_from_silence(self):
        for path in sorted((self.FOLDER / "samples").iterdir()):
            self.assertLessEqual(first_frame_steps(path), 2, path.name)


class BassGuitar(unittest.TestCase):
    """The bass guitar is Karoryfer's Fashionbass as its clean mapping places it
    (fashionbass_clean.sfz: 19 notes, each over the keys its group gives, five layers split at
    velocities 40, 65, 85 and 110, three round robins in thirds of lorand/hirand), keeping the id,
    name, category and keys 21-67 the Black And Blue set had; every sample traced to its file at
    the pinned commit; each starting from silence (the source's first frames sit up to 42 dB under
    their peak, a step from silence the onset fades in)."""

    COMMIT = "0703973b300a6bacb837602be9b588ab64732a72"
    FOLDER = kitlib.SETS / "bass-guitar"
    # (note, lokey, hikey, pitch_keycenter), from fashionbass_clean.sfz's <group> headers.
    NOTES = (("gb0", 12, 19, 18), ("a0", 20, 22, 21), ("c1", 23, 24, 24), ("d1", 25, 27, 26),
             ("f1", 28, 30, 29), ("ab1", 31, 33, 32), ("b1", 34, 36, 35), ("d2", 37, 39, 38),
             ("f2", 40, 42, 41), ("ab2", 43, 45, 44), ("b2", 46, 48, 47), ("d3", 49, 51, 50),
             ("f3", 52, 54, 53), ("ab3", 55, 57, 56), ("b3", 58, 60, 59), ("d4", 61, 63, 62),
             ("f4", 64, 66, 65), ("ab4", 67, 68, 68), ("a4", 69, 72, 69))
    LAYERS = (("pp", None, "40"), ("p", "41", "65"), ("mf", "66", "85"), ("f", "86", "110"),
              ("ff", "111", None))

    def test_the_mapping_is_fashionbass_own(self):
        expected = sorted(((str(low), str(high), str(root), lovel, hivel, "3", str(robin),
                            f"{note}_{layer}_rr{robin}.flac")
                           for note, low, high, root in self.NOTES
                           for layer, lovel, hivel in self.LAYERS for robin in (1, 2, 3)),
                          key=lambda region: region[-1])
        sfz = kitlib.read_sfz(self.FOLDER / "bass-guitar.sfz")
        found = sorted(((r["lokey"], r["hikey"], r["pitch_keycenter"], r.get("lovel"),
                         r.get("hivel"), r["seq_length"], r["seq_position"], r["sample"])
                        for r in sfz.regions), key=lambda region: region[-1])
        self.assertEqual(found, expected)
        self.assertEqual({(r["amp_veltrack"], r["ampeg_release"]) for r in sfz.regions},
                         {("80", "0.4")})

    def test_the_set_keeps_its_place_and_passes_the_kits_checks(self):
        import check
        self.assertEqual(check.check_set(self.FOLDER, kitlib.ROOT), [])
        about = kitlib.read_json(self.FOLDER / "set.json")
        self.assertEqual((about["name"], about["category"]), ("Bass Guitar", "Bass"))
        low, high = about["keys"]
        self.assertTrue(low <= 21 and high >= 67, about["keys"])

    def test_every_sample_traces_to_its_file_at_the_pinned_commit(self):
        pins = {entry["path"]: entry["sha256"] for entry in
                kitlib.read_json(kitlib.SOURCES)["sources"]["karoryfer-fashionbass"]["files"]}
        entries = shipped_samples("bass-guitar")
        self.assertEqual(len(entries), 285)
        for name, entry in entries.items():
            member = f"notes/{name[:-len('.flac')]}.wav"
            self.assertEqual((entry["member"], entry["origin"], entry["origin_sha256"]),
                             (member, kitlib.raw_url("sfzinstruments/karoryfer.fashionbass",
                                                     self.COMMIT, member), pins[member]), name)

    @unittest.skipUnless(shutil.which("ffmpeg"), "needs ffmpeg")
    def test_every_sample_starts_from_silence(self):
        for path in sorted((self.FOLDER / "samples").iterdir()):
            self.assertLessEqual(first_frame_steps(path), 2, path.name)


class Mixing(unittest.TestCase):
    """A multi-microphone hit mixed into one sample: Dawsome's pan law for a mono zone (engine
    dsp/laws.rs: each side sqrt(2) sin((1 + towards) pi/4), so 1 and 1 at the center, 0 and
    sqrt(2) hard over), each file at its gain."""

    def test_pan_gains_are_dawsomes_law(self):
        self.assertEqual(build.pan_gains(0.0), (1.0, 1.0))
        left, right = build.pan_gains(1.0)
        self.assertAlmostEqual(left, 0.0)
        self.assertAlmostEqual(right, math.sqrt(2))
        self.assertAlmostEqual(build.pan_gains(-1.0)[0], math.sqrt(2))
        for pan in (-0.68, -0.25, 0.375, 0.53):
            left, right = build.pan_gains(pan)
            self.assertAlmostEqual(left ** 2 + right ** 2, 2.0)

    @unittest.skipUnless(shutil.which("ffmpeg"), "needs ffmpeg")
    def test_a_panned_mono_mic_and_a_stereo_one_mix_by_hand(self):
        # A mono close mic, every frame 0.4, at gain 0.5 panned hard right; a stereo overhead,
        # left 0.2 and right -0.1, at gain 0.25: left 0.4 x 0.5 x 0 + 0.2 x 0.25 = 0.05, right
        # 0.4 x 0.5 x sqrt(2) - 0.1 x 0.25 = 0.25784 (16-bit files: within 1e-4).
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            write_wav(root / "cl.wav", 44100, [0.4] * 100)
            write_wav(root / "oh.wav", 44100, [0.2, -0.1] * 100, channels=2)
            samples, channels = build.mixed(shutil.which("ffmpeg"), root, (
                build.Mic("cl.wav", 0.5, 1.0), build.Mic("oh.wav", 0.25)))
        self.assertEqual((channels, len(samples)), (2, 200))
        for frame in range(100):
            self.assertAlmostEqual(samples[2 * frame], 0.05, delta=1e-4)
            self.assertAlmostEqual(samples[2 * frame + 1], 0.25784, delta=1e-4)

    @unittest.skipUnless(shutil.which("ffmpeg"), "needs ffmpeg")
    def test_centred_mono_mics_stay_mono_and_takes_must_match(self):
        # 0.4 at gain 0.5 plus -0.2 at gain 0.25: 0.15, one channel.
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            write_wav(root / "a.wav", 44100, [0.4] * 100)
            write_wav(root / "b.wav", 44100, [-0.2] * 100)
            write_wav(root / "short.wav", 44100, [0.1] * 99)
            ffmpeg = shutil.which("ffmpeg")
            samples, channels = build.mixed(ffmpeg, root, (build.Mic("a.wav", 0.5),
                                                           build.Mic("b.wav", 0.25)))
            with self.assertRaises(kitlib.Problem):
                build.mixed(ffmpeg, root, (build.Mic("a.wav", 0.5), build.Mic("short.wav", 1.0)))
        self.assertEqual((channels, len(samples)), (1, 100))
        self.assertAlmostEqual(samples[50], 0.15, delta=1e-4)


class AcousticDrums(unittest.TestCase):
    """The acoustic drums are Karoryfer's Big Rusty Drums on the keys the set had (Virtuosity's
    .sfz: 36 kick, 37 cross-stick, 38 snare, 40 rimshot, 41 low tom, 42 closed hi-hat, 44 pedal
    hi-hat, 46 open hi-hat, 48 high tom, 49 crash, 51 ride, 53 ride bell), stick hits only, at
    most four velocity layers and two round robins a drum, each hit its microphones mixed at
    Programs/02-basic.sfz's defaults (amplitude_ccN=share at set_ccN, pan through its curve 9: -1
    at 0, 0 at 63, 1 at 127), every file traced, every hit starting from silence."""

    COMMIT = "f07ce00df34a46b6b08375be56fe116cf15782bc"
    FOLDER = kitlib.SETS / "drums-acoustic"
    # key -> Big Rusty's file stem for it (its keymap and 02-basic.sfz's mappings).
    HITS = {36: "k_vl", 37: "sn_ss_vl", 38: "sn_center_vl", 40: "sn_rims_vl", 41: "t18_vl",
            42: "ht_tc_vl", 44: "ht_chik_vl", 46: "ht_open_vl", 48: "t14_vl", 49: "cr_vl",
            51: "rd_vl", 53: "rd_bl_vl"}
    # The mix at 02-basic.sfz's defaults, read off its text: (gain in dB, pan) per microphone, in
    # the order build.py mixes them. kick: amplitude_cc70=100 at set_cc70=100 (the overhead's
    # CC71 and the snare mic's CC72 are not set, so 0); snare: btm, top, oh at cc80=100, cc81=40,
    # cc82=60, pan_cc88 at 63; toms: cl at cc34=100 / cc42=100 panned by cc36=87 / cc44=20, oh at
    # cc35=70 / cc43=70; hi-hat: amplitude 70 at cc15=100 and cc16=100, pan cc17=63; crash: 70 at
    # cc100=40 (pan cc102=30) and cc101=100; ride and bell: 60 at cc104=40 (pan cc106=97) and
    # cc105=100.
    MIX = {
        36: [(20 * math.log10(100 / 127), 0.0)],
        37: [(20 * math.log10(100 / 127), 0.0), (20 * math.log10(40 / 127), 0.0),
             (20 * math.log10(60 / 127), 0.0)],
        41: [(20 * math.log10(100 / 127), (87 - 63) / 64), (20 * math.log10(70 / 127), 0.0)],
        42: [(20 * math.log10(0.7 * 100 / 127), 0.0)] * 2,
        48: [(20 * math.log10(100 / 127), (20 - 63) / 63), (20 * math.log10(70 / 127), 0.0)],
        49: [(20 * math.log10(0.7 * 40 / 127), (30 - 63) / 63),
             (20 * math.log10(0.7 * 100 / 127), 0.0)],
        51: [(20 * math.log10(0.6 * 40 / 127), (97 - 63) / 64),
             (20 * math.log10(0.6 * 100 / 127), 0.0)],
    }
    MIX.update({38: MIX[37], 40: MIX[37], 44: MIX[42], 46: MIX[42], 53: MIX[51]})

    def regions(self):
        return kitlib.read_sfz(self.FOLDER / "drums-acoustic.sfz").regions

    def test_the_keys_play_the_drums_they_played(self):
        by_key = {}
        for region in self.regions():
            self.assertEqual(region["lokey"], region["hikey"])
            by_key.setdefault(int(region["lokey"]), []).append(region)
        self.assertEqual(sorted(by_key), sorted(self.HITS))
        for key, regions in by_key.items():
            self.assertTrue(all(r["sample"].startswith(self.HITS[key]) for r in regions), key)
            layers = {(r["lovel"], r["hivel"]) for r in regions}
            self.assertLessEqual(len(layers), 4, key)
            self.assertEqual(min(int(low) for low, _ in layers), 1)
            self.assertEqual(max(int(high) for _, high in layers), 127)
            self.assertLessEqual(len(regions) / len(layers), 2, key)
            self.assertTrue(all(int(r.get("seq_length", 1)) <= 2 for r in regions), key)

    def test_the_set_keeps_its_place_and_passes_the_kits_checks(self):
        import check
        self.assertEqual(check.check_set(self.FOLDER, kitlib.ROOT), [])
        about = kitlib.read_json(self.FOLDER / "set.json")
        self.assertEqual((about["name"], about["keys"], "category" in about),
                         ("Acoustic Drums", [36, 53], False))

    def test_every_hit_is_the_programs_own_mix_of_pinned_files(self):
        pins = {entry["path"]: entry["sha256"] for entry in
                kitlib.read_json(kitlib.SOURCES)["sources"]["karoryfer-big-rusty-drums"]["files"]}
        entries = shipped_samples("drums-acoustic")
        keys = {region["sample"]: int(region["lokey"]) for region in self.regions()}
        self.assertEqual(sorted(entries), sorted(keys))
        for name, entry in entries.items():
            mics = entry["mixed_from"]
            self.assertEqual(len(mics), len(self.MIX[keys[name]]), name)
            for mic, (gain_db, pan) in zip(mics, self.MIX[keys[name]]):
                self.assertTrue(mic["member"].endswith(f"/{name[:-len('.flac')]}.flac"), name)
                self.assertEqual((mic["origin"], mic["origin_sha256"]),
                                 (kitlib.raw_url("sfzinstruments/karoryfer.big-rusty-drums",
                                                 self.COMMIT, mic["member"]),
                                  pins[mic["member"]]), name)
                self.assertAlmostEqual(mic["gain_db"], gain_db, delta=0.006)
                self.assertAlmostEqual(mic["pan"], pan, delta=0.00006)

    @unittest.skipUnless(shutil.which("ffmpeg"), "needs ffmpeg")
    def test_every_hit_starts_from_silence(self):
        for path in sorted((self.FOLDER / "samples").iterdir()):
            self.assertLessEqual(first_frame_steps(path), 2, path.name)


def _compare_or_skip(test):
    """compare.py, for a test that renders through dawsome; skips without numpy, scipy,
    soundfile or a built dawsome."""
    import compare
    if compare.IMPORT_ERROR is not None:
        test.skipTest("needs numpy, scipy and soundfile")
    if not compare.default_binary().is_file():
        test.skipTest(f"needs a built dawsome at {compare.default_binary()}")
    return compare


class Writing(unittest.TestCase):
    def test_the_sfz_reads_back_as_its_zones(self):
        zones = [
            build.Zone("a/C4 soft.wav", {"lokey": "60", "hikey": "61", "pitch_keycenter": "60",
                                         "ampeg_release": "0.5", "loop_mode": "one_shot"}),
            build.Zone("a/D4.wav", {"lokey": "62", "hikey": "63", "pitch_keycenter": "62",
                                    "ampeg_release": "0.5", "loop_mode": "no_loop"}),
        ]
        plan = build.Plan("demo", "Demo", "demo-source", zones)
        text = build.format_sfz(plan, {"title": "Demo source"}, zones)
        self.assertIn("<global>\nampeg_release=0.5\n", text)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "demo.sfz"
            path.write_text(text)
            sfz = kitlib.read_sfz(path)
        self.assertEqual([sfz.sample_path(region) for region in sfz.regions],
                         ["samples/C4 soft.flac", "samples/D4.flac"])
        for region, zone in zip(sfz.regions, zones):
            self.assertEqual({k: v for k, v in region.items() if k != "sample"}, zone.opcodes)
        self.assertLessEqual(sfz.used, kitlib.SFZ_OPCODES | kitlib.SFZ_DIRECTIVES
                             | {f"<{header}>" for header in kitlib.SFZ_HEADERS})

    def test_strip_padding_removes_only_the_padding(self):
        # metaflac: STREAMINFO, SEEKTABLE, VORBIS_COMMENT, then PADDING of 8192 bytes.
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "ramp.flac"
            shutil.copyfile(FIXTURES / "audio" / "ramp.flac", path)
            before = path.read_bytes()
            build.strip_padding(path)
            after = path.read_bytes()
            info = kitlib.audio_info(path)
        self.assertEqual(len(before) - len(after), 4 + 8192)
        self.assertEqual((info.rate, info.channels, info.frames), (22050, 1, 1000))
        blocks, position = [], 4
        while True:
            header = after[position]
            blocks.append(header & 0x7F)
            position += 4 + int.from_bytes(after[position + 1:position + 4], "big")
            if header & 0x80:
                break
        self.assertEqual(blocks, [0, 3, 4])
        self.assertEqual(after[position:], before[-(len(after) - position):])


if __name__ == "__main__":
    unittest.main()
