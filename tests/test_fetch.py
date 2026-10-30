# SPDX-License-Identifier: GPL-3.0-or-later
"""fetch.py's guards on untrusted archives: member names that climb out or are absolute, links,
duplicate names, and an interrupted extraction. The archives are made here with Python's own
tarfile and zipfile, so what each holds is known."""

import io
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path, PurePosixPath

import fetch
import kitlib


def tar_with(path, members):
    """A .tar.gz at `path` holding (name, bytes) files and (name, tarfile type, link target)
    entries (a type is itself bytes, so the two are told apart by length)."""
    with tarfile.open(path, "w:gz") as archive:
        for member in members:
            info = tarfile.TarInfo(member[0])
            if len(member) == 2:
                info.size = len(member[1])
                archive.addfile(info, io.BytesIO(member[1]))
            else:
                info.type, info.linkname = member[1], member[2]
                archive.addfile(info)


class SafeRelative(unittest.TestCase):
    def test_plain_paths_pass_and_are_tidied(self):
        self.assertEqual(fetch.safe_relative("a/b.wav"), PurePosixPath("a/b.wav"))
        self.assertEqual(fetch.safe_relative("./a//./b.wav"), PurePosixPath("a/b.wav"))
        self.assertEqual(fetch.safe_relative("a\\b.wav"), PurePosixPath("a/b.wav"))
        self.assertIsNone(fetch.safe_relative("./"))

    def test_escapes_are_refused(self):
        for name in ("../evil", "a/../../evil", "/etc/passwd", "\\evil", "C:\\evil", "a\0b"):
            with self.subTest(name), self.assertRaises(kitlib.Problem):
                fetch.safe_relative(name)

    def test_archive_names_come_from_the_url(self):
        self.assertEqual(fetch.archive_name("https://h/x/Kit%20One-SFZ+FLAC.7z"),
                         "Kit One-SFZ+FLAC.7z")
        for url in ("https://h/x/", "https://h/x/kit.exe", "https://h/x/.hidden.zip"):
            with self.subTest(url), self.assertRaises(kitlib.Problem):
                fetch.archive_name(url)


class Extract(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        self.target = self.root / "source" / "extracted"
        self.target.parent.mkdir()

    def written(self):
        """Every file under the temporary root, as relative paths."""
        return sorted(path.relative_to(self.root).as_posix()
                      for path in self.root.rglob("*") if path.is_file())

    def test_a_plain_archive_is_extracted(self):
        archive = self.root / "source" / "kit.tar.gz"
        tar_with(archive, [("kit/a.wav", b"aa"), ("kit/sub/b.wav", b"bbb")])
        fetch.extract(archive, self.target)
        self.assertEqual((self.target / "kit/a.wav").read_bytes(), b"aa")
        self.assertEqual((self.target / "kit/sub/b.wav").read_bytes(), b"bbb")

    def test_a_climbing_member_stops_everything_before_a_byte_is_written(self):
        archive = self.root / "source" / "kit.tar.gz"
        tar_with(archive, [("kit/a.wav", b"aa"), ("../../evil.txt", b"x")])
        with self.assertRaisesRegex(kitlib.Problem, "climbs out"):
            fetch.extract(archive, self.target)
        self.assertEqual(self.written(), ["source/kit.tar.gz"])

    def test_links_are_refused(self):
        archive = self.root / "source" / "kit.tar.gz"
        tar_with(archive, [("kit/link", tarfile.SYMTYPE, "/etc/passwd")])
        with self.assertRaisesRegex(kitlib.Problem, "link"):
            fetch.extract(archive, self.target)
        zipped = self.root / "source" / "kit.zip"
        with zipfile.ZipFile(zipped, "w") as archive:
            info = zipfile.ZipInfo("kit/link")
            info.external_attr = 0o120777 << 16
            archive.writestr(info, "/etc/passwd")
        with self.assertRaisesRegex(kitlib.Problem, "link"):
            fetch.extract(zipped, self.target)
        self.assertFalse(self.target.exists())

    def test_a_second_member_of_the_same_name_is_refused(self):
        archive = self.root / "source" / "kit.zip"
        with zipfile.ZipFile(archive, "w") as zipped, self.assertWarns(UserWarning):
            zipped.writestr("kit/a.wav", "first")
            zipped.writestr("kit/a.wav", "second")
        with self.assertRaises(FileExistsError):
            fetch.extract(archive, self.target)
        self.assertFalse(self.target.exists())

    def test_an_interrupted_extraction_is_redone_from_scratch(self):
        staging = self.target.with_name("extracted.partial")
        staging.mkdir()
        (staging / "half.wav").write_bytes(b"left over")
        archive = self.root / "source" / "kit.tar.gz"
        tar_with(archive, [("kit/a.wav", b"aa")])
        fetch.extract(archive, self.target)
        self.assertEqual(self.written(), ["source/extracted/kit/a.wav", "source/kit.tar.gz"])


class Sources(unittest.TestCase):
    def test_a_source_without_a_cc0_licence_and_quote_is_not_ready(self):
        sources = {"bad": {"licence": "CC-BY-4.0", "licence_page": "http://x",
                           "archive": {"url": "https://x/a.7z", "sha256": None}}}
        found = fetch.validate_sources(sources, pinning=False)
        self.assertEqual(len(found), 4)
        self.assertTrue(any("CC-BY-4.0" in problem for problem in found))
        self.assertTrue(any("unpinned" in problem for problem in found))
        self.assertEqual(len(fetch.validate_sources(sources, pinning=True)), 3)

    def test_the_kits_own_sources_are_ready(self):
        sources = kitlib.read_json(kitlib.SOURCES)["sources"]
        self.assertEqual(fetch.validate_sources(sources, pinning=False), [])


if __name__ == "__main__":
    unittest.main()
