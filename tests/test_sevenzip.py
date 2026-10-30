# SPDX-License-Identifier: GPL-3.0-or-later
"""Known answers for sevenzip.py: every archive in fixtures/7z was made by the 7-Zip program
from fixtures/7z/content (make.sh says how), so a correct reader gives back exactly those files.
Damaged copies of an archive must be refused, never extracted wrongly."""

import tempfile
import unittest
from pathlib import Path

import sevenzip

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "7z"
CONTENT = FIXTURES / "content"
# Git keeps no empty folders, so make.sh adds this one to the archives itself.
EXPECTED_DIRS = {"dir", "emptydir"}


def expected_files():
    return {
        path.relative_to(CONTENT).as_posix(): path.read_bytes()
        for path in CONTENT.rglob("*")
        if path.is_file()
    }


def extract(path):
    """(name -> bytes of each file, names of the folders) as the reader gives them."""
    files, dirs = {}, set()
    with sevenzip.SevenZipArchive(path) as archive:
        for member, chunks in archive.contents():
            if member.is_dir:
                dirs.add(member.name)
            else:
                files[member.name] = b"".join(chunks)
    return files, dirs


class Extract(unittest.TestCase):
    def test_every_supported_method_gives_the_original_files(self):
        for name in ("lzma2.7z", "lzma.7z", "copy.7z", "delta.7z", "bcj.7z", "nonsolid.7z",
                     "deflate.7z", "bzip2.7z"):
            with self.subTest(name):
                files, dirs = extract(FIXTURES / name)
                self.assertEqual(files, expected_files())
                self.assertEqual(dirs, EXPECTED_DIRS)

    def test_members_are_listed_before_any_data_is_read(self):
        with sevenzip.SevenZipArchive(FIXTURES / "lzma2.7z") as archive:
            sizes = {member.name: member.size for member in archive.members if not member.is_dir}
        self.assertEqual(sizes, {name: len(data) for name, data in expected_files().items()})

    def test_a_member_left_unread_does_not_shift_the_next_one(self):
        with sevenzip.SevenZipArchive(FIXTURES / "lzma2.7z") as archive:
            files = {}
            for member, chunks in archive.contents():
                if member.name != "dir/notes.txt" and not member.is_dir:
                    files[member.name] = b"".join(chunks)
        expected = expected_files()
        del expected["dir/notes.txt"]
        self.assertEqual(files, expected)

    def test_an_unsupported_method_is_named(self):
        with self.assertRaisesRegex(sevenzip.SevenZipError, "PPMd"):
            extract(FIXTURES / "ppmd.7z")

    def test_bcj_in_front_of_lzma1_is_refused_not_cut_short(self):
        with self.assertRaisesRegex(sevenzip.SevenZipError, "BCJ"):
            extract(FIXTURES / "bcj-lzma.7z")

    def test_an_encrypted_archive_is_refused(self):
        with self.assertRaisesRegex(sevenzip.SevenZipError, "AES"):
            extract(FIXTURES / "encrypted.7z")


class Damage(unittest.TestCase):
    def damaged(self, change):
        """A temporary copy of copy.7z (stored, header not compressed) with `change` applied."""
        data = bytearray((FIXTURES / "copy.7z").read_bytes())
        change(data)
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        path = Path(folder.name) / "damaged.7z"
        path.write_bytes(bytes(data))
        return path

    def test_a_changed_data_byte_fails_the_crc(self):
        # Byte 40 is inside the first stored file, right after the 32-byte start header.
        path = self.damaged(lambda data: data.__setitem__(40, data[40] ^ 0xFF))
        with self.assertRaisesRegex(sevenzip.SevenZipError, "CRC"):
            extract(path)

    def test_a_truncated_archive_is_refused(self):
        path = self.damaged(lambda data: data.__delitem__(slice(-20, None)))
        with self.assertRaisesRegex(sevenzip.SevenZipError, "truncated"):
            extract(path)

    def test_a_changed_header_byte_is_refused(self):
        path = self.damaged(lambda data: data.__setitem__(-5, data[-5] ^ 0xFF))
        with self.assertRaisesRegex(sevenzip.SevenZipError, "header is damaged"):
            extract(path)

    def test_a_file_that_is_not_7z_is_refused(self):
        path = self.damaged(lambda data: data.__setitem__(0, ord("8")))
        with self.assertRaisesRegex(sevenzip.SevenZipError, "not a 7z archive"):
            extract(path)


if __name__ == "__main__":
    unittest.main()
