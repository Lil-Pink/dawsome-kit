# SPDX-License-Identifier: GPL-3.0-or-later
"""check.py against a small kit made here: it must pass as made, and fail, naming the problem,
for each way a shipped kit can go wrong."""

import hashlib
import json
import tempfile
import unittest
import wave
from pathlib import Path

import check

SFZ = """<control>
default_path=samples/
<global>
ampeg_release=0.5
<region> sample=a.wav lokey=60 hikey=61 pitch_keycenter=60
<region> sample=b.wav lokey=62 hikey=62 pitch_keycenter=62 loop_mode=loop_continuous loop_start=10 loop_end=199
"""
FRAMES = {"a.wav": 100, "b.wav": 200}
PAGE = "https://example.org/licence"


def write_wav(path, frames):
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(44100)
        out.writeframes(b"\1\0" * frames)


def make_kit(root, sfz=SFZ, frames=FRAMES):
    """A one-set kit under `root` whose set.json and MANIFEST.json match its files."""
    folder = root / "sets" / "demo"
    (folder / "samples").mkdir(parents=True)
    for name, count in frames.items():
        write_wav(folder / "samples" / name, count)
    (folder / "demo.sfz").write_text(sfz)
    shipped = [folder / "samples" / name for name in frames] + [folder / "demo.sfz"]
    about = {
        "name": "Demo", "kind": "instrument", "maker": "test", "licence": "CC0-1.0",
        "licence_page": PAGE, "licence_quote": "a quote", "source": "test",
        "source_url": PAGE, "sfz": "demo.sfz", "zones": 2, "keys": [60, 62], "peak_db": -1.0,
        "files": len(frames), "size": sum(path.stat().st_size for path in shipped),
        "decoded_size": sum(frames.values()) * 4, "sample_rates": [44100], "channels": [1],
    }
    (folder / "set.json").write_text(json.dumps(about))
    write_manifest(root)


def write_manifest(root, change=None):
    """MANIFEST.json for every file now under sets/; `change(path, entry)` may alter entries."""
    entries = []
    for path in sorted((root / "sets").rglob("*")):
        if path.is_dir():
            continue
        relative = path.relative_to(root).as_posix()
        entry = {"path": relative, "size": path.stat().st_size,
                 "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                 "licence": "CC0-1.0", "licence_page": PAGE, "source": "test"}
        if path.suffix == ".wav":
            entry.update(made_by="copied", origin=PAGE, origin_sha256="0" * 64,
                         member=f"kit/{path.name}")
        else:
            entry["made_by"] = "written"
        if change:
            change(relative, entry)
        entries.append(entry)
    downloads = [{"set": "big", "name": "Big", "url": PAGE, "sha256": "1" * 64, "size": 5,
                  "licence": "CC0-1.0", "licence_page": PAGE, "licence_quote": "a quote"}]
    (root / "MANIFEST.json").write_text(json.dumps({"files": entries, "downloads": downloads}))


class Check(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)

    def problems(self):
        return check.check_manifest(self.root) + check.check_sets(self.root)

    def assertProblem(self, text):
        found = self.problems()
        self.assertTrue(any(text in problem for problem in found), found)

    def test_the_kit_as_made_passes(self):
        make_kit(self.root)
        self.assertEqual(self.problems(), [])

    def test_a_file_the_manifest_does_not_list(self):
        make_kit(self.root)
        (self.root / "sets" / "demo" / "notes.txt").write_text("hello")
        self.assertProblem("sets/demo/notes.txt: in sets/ but not in the manifest")

    def test_a_changed_byte(self):
        make_kit(self.root)
        path = self.root / "sets" / "demo" / "samples" / "a.wav"
        data = bytearray(path.read_bytes())
        data[-1] ^= 0xFF
        path.write_bytes(bytes(data))
        self.assertProblem("a.wav: SHA-256 differs")

    def test_a_licence_other_than_cc0(self):
        make_kit(self.root)
        write_manifest(self.root, lambda path, entry: entry.update(licence="CC-BY-4.0")
                       if path.endswith("a.wav") else None)
        self.assertProblem("license 'CC-BY-4.0'")

    def test_a_sample_without_its_origin(self):
        make_kit(self.root)
        write_manifest(self.root, lambda path, entry: entry.pop("origin", None))
        self.assertProblem("no origin URL")

    def mixed(self, change=None):
        """The kit with a.wav mixed from two microphones' files (its own and an overhead's),
        `change(mics)` applied to their list."""
        make_kit(self.root)

        def mix(path, entry):
            if path.endswith("a.wav"):
                mics = [{"member": entry["member"], "origin": entry["origin"],
                         "origin_sha256": entry["origin_sha256"], "gain_db": -2.08, "pan": 0.375},
                        {"member": "kit/oh/a.wav", "origin": PAGE, "origin_sha256": "3" * 64,
                         "gain_db": -5.17, "pan": 0.0}]
                if change:
                    change(mics)
                entry["mixed_from"] = mics
        write_manifest(self.root, mix)

    def test_a_sample_mixed_from_every_microphones_file_passes(self):
        self.mixed()
        self.assertEqual(self.problems(), [])

    def test_a_mixed_microphone_without_its_hash(self):
        self.mixed(lambda mics: mics[1].pop("origin_sha256"))
        self.assertProblem("mixed_from needs each microphone's")

    def test_a_mix_that_does_not_start_with_the_samples_own_file(self):
        self.mixed(lambda mics: mics.reverse())
        self.assertProblem("mixed_from does not start with the entry's own member")

    def test_a_generated_sample_names_its_script_and_hash(self):
        make_kit(self.root)
        script = self.root / "gen.py"
        script.write_text("print('a generator')\n")
        digest = hashlib.sha256(script.read_bytes()).hexdigest()

        def generated(sha256):
            def change(path, entry):
                if path.endswith(".wav"):
                    for name in ("origin", "origin_sha256"):
                        entry.pop(name)
                    entry.update(made_by="generated", generator="gen.py", generator_sha256=sha256)
            return change
        write_manifest(self.root, generated(digest))
        self.assertEqual(self.problems(), [])
        write_manifest(self.root, generated("0" * 64))
        self.assertProblem("has changed since it was made")

    def generated_with(self, **fields):
        """The kit with its WAVs generated by gen.py, which uses lib.py, plus `fields`."""
        make_kit(self.root)
        for name, text in (("gen.py", "print('a generator')\n"), ("lib.py", "x = 1\n")):
            (self.root / name).write_text(text)

        def change(path, entry):
            if path.endswith(".wav"):
                for name in ("origin", "origin_sha256"):
                    entry.pop(name)
                entry.update(made_by="generated", generator="gen.py",
                             generator_sha256=hashlib.sha256(b"print('a generator')\n")
                             .hexdigest(), **fields)
        write_manifest(self.root, change)

    def test_a_generator_module_that_changed_since(self):
        self.generated_with(generator_uses=[{
            "path": "lib.py", "sha256": hashlib.sha256(b"x = 1\n").hexdigest()}])
        self.assertEqual(self.problems(), [])
        (self.root / "lib.py").write_text("x = 2\n")
        self.assertProblem("lib.py has changed since it was made")

    def test_a_generator_module_without_its_hash(self):
        self.generated_with(generator_uses=[{"path": "lib.py"}])
        self.assertProblem("generator_uses needs")

    def write_sources(self, licence="CC0-1.0", files=("Samples/a_1.wav",)):
        (self.root / "sources.json").write_text(json.dumps({"sources": {"rec": {
            "licence": licence, "licence_page": PAGE,
            "files": [{"path": path, "sha256": "2" * 64} for path in files]}}}))

    def made_from(self, members=("Samples/a_1.wav",)):
        return {"source": "rec", "origin": PAGE, "licence_page": PAGE, "members": list(members)}

    def test_a_sample_made_from_pinned_recordings_passes(self):
        self.write_sources()
        self.generated_with(made_from=self.made_from())
        self.assertEqual(self.problems(), [])

    def made_with(self, **fields):
        """The kit with its WAVs committed as made (no script in the repository), plus `fields`."""
        make_kit(self.root)

        def change(path, entry):
            if path.endswith(".wav"):
                for name in ("origin", "origin_sha256"):
                    entry.pop(name)
                entry.update(made_by="made", **fields)
        write_manifest(self.root, change)

    def test_a_made_sample_traced_to_pinned_recordings_passes(self):
        self.write_sources()
        self.made_with(made_from=self.made_from())
        self.assertEqual(self.problems(), [])

    def test_a_made_sample_without_made_from(self):
        self.write_sources()
        self.made_with()
        self.assertProblem("made, but no made_from to trace it by")

    def test_a_made_sample_made_from_a_file_the_source_does_not_pin(self):
        self.write_sources()
        self.made_with(made_from=self.made_from(("Samples/b_1.wav",)))
        self.assertProblem("Samples/b_1.wav is not a file sources.json pins for rec")

    def test_made_from_a_file_the_source_does_not_pin(self):
        self.write_sources()
        self.generated_with(made_from=self.made_from(("Samples/a_1.wav", "Samples/b_1.wav")))
        self.assertProblem("Samples/b_1.wav is not a file sources.json pins for rec")

    def test_made_from_a_source_that_is_not_cc0(self):
        self.write_sources(licence="CC-BY-4.0")
        self.generated_with(made_from=self.made_from())
        self.assertProblem("made_from rec is licensed CC-BY-4.0")

    def test_made_from_without_members(self):
        self.write_sources()
        self.generated_with(made_from=self.made_from(()))
        self.assertProblem("made_from needs")

    def test_made_from_without_sources_json(self):
        self.generated_with(made_from=self.made_from())
        self.assertProblem("made_from rec is not in sources.json")

    def test_an_opcode_outside_the_subset(self):
        make_kit(self.root, SFZ.replace("pitch_keycenter=60", "pitch_keycenter=60 cutoff=500"))
        self.assertProblem("outside the SFZ subset: cutoff")

    def test_a_note_name_where_the_kit_writes_numbers(self):
        make_kit(self.root, SFZ.replace("lokey=60", "lokey=c4"))
        self.assertProblem("lokey=c4")

    def test_a_sample_no_region_uses(self):
        make_kit(self.root, frames={**FRAMES, "c.wav": 50})
        self.assertProblem("samples/c.wav is used by no region")

    def test_a_missing_sample(self):
        make_kit(self.root, SFZ.replace("sample=a.wav", "sample=gone.wav"))
        self.assertProblem("sample 'samples/gone.wav' is not a file of the set")

    def test_a_loop_without_points_or_past_the_end(self):
        make_kit(self.root, SFZ.replace(" loop_start=10 loop_end=199", ""))
        self.assertProblem("loops without loop_start and loop_end")
        make_kit(self.root / "second", SFZ.replace("loop_end=199", "loop_end=200"))
        self.root = self.root / "second"
        self.assertProblem("loop 10..200 does not fit 200 frames")

    def test_set_json_must_match_its_files(self):
        make_kit(self.root)
        path = self.root / "sets" / "demo" / "set.json"
        about = json.loads(path.read_text())
        about["decoded_size"] = 1
        path.write_text(json.dumps(about))
        write_manifest(self.root)
        self.assertProblem("decoded_size is 1")

    def test_an_instrument_louder_than_the_limit(self):
        make_kit(self.root)
        path = self.root / "sets" / "demo" / "set.json"
        about = json.loads(path.read_text())
        about["peak_db"] = -0.5
        path.write_text(json.dumps(about))
        write_manifest(self.root)
        self.assertProblem("peak_db -0.5")

    def test_a_download_set_needs_a_hash(self):
        make_kit(self.root)
        manifest = json.loads((self.root / "MANIFEST.json").read_text())
        manifest["downloads"][0]["sha256"] = "not a hash"
        (self.root / "MANIFEST.json").write_text(json.dumps(manifest))
        self.assertProblem("sha256 is not a SHA-256")


if __name__ == "__main__":
    unittest.main()
