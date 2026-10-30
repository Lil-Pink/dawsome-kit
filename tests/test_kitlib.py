# SPDX-License-Identifier: GPL-3.0-or-later
"""Known answers for kitlib: SFZ key names, the SFZ reader, and audio headers. Expected values
come from the SFZ format (c4 = 60), from fixtures/sfz/main.sfz worked through by hand, and from
metaflac's listing of fixtures/audio/ramp.flac (see its make.sh)."""

import struct
import tempfile
import unittest
import wave
from pathlib import Path

import kitlib

FIXTURES = Path(__file__).resolve().parent / "fixtures"


class NoteNumber(unittest.TestCase):
    def test_names_and_numbers(self):
        cases = {"c4": 60, "C4": 60, "c#4": 61, "db4": 61, "a0": 21, "c-1": 0, "g9": 127,
                 "60": 60, " 36 ": 36, "b3": 59}
        for value, expected in cases.items():
            with self.subTest(value):
                self.assertEqual(kitlib.note_number(value), expected)

    def test_refuses_what_is_not_a_key(self):
        for value in ("h4", "c10", "128", "-1", "", "c#"):
            with self.subTest(value), self.assertRaises(kitlib.Problem):
                kitlib.note_number(value)


class ReadSfz(unittest.TestCase):
    def test_regions_inherit_and_variables_and_includes_resolve(self):
        sfz = kitlib.read_sfz(FIXTURES / "sfz" / "main.sfz")
        self.assertEqual(sfz.control, {"default_path": "samples/"})
        self.assertEqual(sfz.regions, [
            {"ampeg_release": "0.5", "volume": "-3", "lovel": "1", "hivel": "63",
             "sample": "soft kick.wav", "key": "36"},
            {"ampeg_release": "0.5", "volume": "-6", "lovel": "1", "hivel": "63",
             "sample": "soft snare.wav", "lokey": "c#2", "hikey": "d2",
             "pitch_keycenter": "38"},
            {"ampeg_release": "0.5", "volume": "-3", "lovel": "64",
             "sample": "loud kick.wav", "key": "36", "cutoff": "500"},
            # A second <global> starts over: the earlier global and group no longer apply.
            {"sample": "low.wav", "lokey": "24", "hikey": "36"},
        ])
        self.assertEqual(sfz.sample_path(sfz.regions[0]), "samples/soft kick.wav")

    def test_lists_everything_used(self):
        sfz = kitlib.read_sfz(FIXTURES / "sfz" / "main.sfz")
        self.assertEqual(sfz.used, {
            "#define", "#include", "<control>", "<global>", "<group>", "<region>",
            "default_path", "ampeg_release", "volume", "lovel", "hivel", "sample", "key",
            "lokey", "hikey", "pitch_keycenter", "cutoff"})

    def test_an_include_loop_is_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "self.sfz"
            path.write_text('#include "self.sfz"\n')
            with self.assertRaisesRegex(kitlib.Problem, "deeper"):
                kitlib.read_sfz(path)


class AudioInfo(unittest.TestCase):
    def test_flac_header(self):
        info = kitlib.audio_info(FIXTURES / "audio" / "ramp.flac")
        self.assertEqual((info.rate, info.channels, info.bits, info.frames), (22050, 1, 16, 1000))
        self.assertEqual(info.decoded_size, 4000)

    def test_wav_header(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "stereo.wav"
            with wave.open(str(path), "wb") as out:
                out.setnchannels(2)
                out.setsampwidth(3)
                out.setframerate(48000)
                out.writeframes(b"\0" * 6 * 777)
            info = kitlib.audio_info(path)
        self.assertEqual((info.rate, info.channels, info.bits, info.frames), (48000, 2, 24, 777))
        self.assertEqual(info.decoded_size, 777 * 2 * 4)

    def test_other_files_are_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "note.txt"
            path.write_bytes(struct.pack("<4s", b"OggS") + b"\0" * 40)
            with self.assertRaisesRegex(kitlib.Problem, "neither FLAC nor WAV"):
                kitlib.audio_info(path)


if __name__ == "__main__":
    unittest.main()
