# SPDX-License-Identifier: GPL-3.0-or-later
"""What fetch.py, build.py and check.py share: where things are, hashing, JSON files, and the
SFZ subset Dawsome reads.

The SFZ subset is defined once, here, so the files build.py writes and the files check.py
accepts can never disagree. It is the list in the DAW's multisample-instruments spec
(2026-10-22); the DAW ignores everything else.
"""

import hashlib
import json
import os
import re
import struct
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SETS = ROOT / "sets"
SOURCES = ROOT / "sources.json"
MANIFEST = ROOT / "MANIFEST.json"
DOWNLOADS = ROOT / "downloads"
LICENCE = "CC0-1.0"
# The most a full-velocity note of an instrument set may peak at, and where a normalized set's
# samples peak: 1 dB under full scale (the DAW engine's request, 2026-10-22).
PEAK_DB = -1.0
CC0_DEED = "https://creativecommons.org/publicdomain/zero/1.0/"

SFZ_HEADERS = frozenset({"control", "global", "group", "region"})
SFZ_DIRECTIVES = frozenset({"#define", "#include"})
SFZ_OPCODES = frozenset({
    "default_path", "sample", "lokey", "hikey", "key", "pitch_keycenter", "lovel", "hivel",
    "loop_mode", "loop_start", "loop_end", "ampeg_attack", "ampeg_release", "amp_veltrack",
    "volume", "tune", "transpose", "pan", "offset", "end", "seq_length", "seq_position",
})
LOOP_MODES = frozenset({"no_loop", "one_shot", "loop_continuous", "loop_sustain"})
NOTE_NAMES = {"c": 0, "d": 2, "e": 4, "f": 5, "g": 7, "a": 9, "b": 11}

# Two tokens on an SFZ line: a header, or the start of an opcode (its value runs to the next
# token, so a sample path may hold spaces).
_TOKEN = re.compile(r"<(\w+)>|(\w+)=")
_DEFINE = re.compile(r"\s*#define\s+(\$\w+)\s+(.*?)\s*$")
_INCLUDE = re.compile(r'\s*#include\s+"([^"]+)"\s*$')
_VARIABLE = re.compile(r"\$\w+")
_INCLUDE_DEPTH = 16


class Problem(Exception):
    """A check fails or a step cannot run; the message says which and why."""


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        while chunk := file.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    try:
        with open(path, encoding="utf-8") as file:
            return json.load(file)
    except (OSError, json.JSONDecodeError) as error:
        raise Problem(f"cannot read {path}: {error}") from None


def write_json(path, data):
    """Writes `data` as indented UTF-8 JSON with a final newline, atomically (temp file and
    rename), so an interrupted run never leaves half a manifest."""
    path = Path(path)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def relative(path):
    """`path` as the repository names it: relative to the root, with forward slashes."""
    return Path(path).resolve().relative_to(ROOT).as_posix()


@dataclass(frozen=True)
class AudioInfo:
    rate: int
    channels: int
    bits: int
    frames: int

    @property
    def decoded_size(self):
        """Bytes once decoded to 32-bit float at its own rate, the way the DAW preloads zones
        (resampling to a 48 kHz project adds 48/44.1, about 9 %, for 44.1 kHz files)."""
        return self.frames * self.channels * 4


def audio_info(path):
    """Rate, channels, bits and length in frames of a FLAC or WAV file, from its header."""
    with open(path, "rb") as file:
        head = file.read(12)
        if head[:4] == b"fLaC":
            file.seek(4)
            while True:
                block = file.read(4)
                if len(block) < 4:
                    break
                kind, length = block[0] & 0x7F, int.from_bytes(block[1:4], "big")
                body = file.read(length)
                if kind == 0 and len(body) >= 18:
                    # STREAMINFO: rate (20 bits), channels - 1 (3), bits - 1 (5), frames (36).
                    packed = int.from_bytes(body[10:18], "big")
                    return AudioInfo(packed >> 44, ((packed >> 41) & 0x7) + 1,
                                     ((packed >> 36) & 0x1F) + 1, packed & 0xFFFFFFFFF)
                if block[0] & 0x80:
                    break
            raise Problem(f"{path}: a FLAC file without STREAMINFO")
        if head[:4] == b"RIFF" and head[8:12] == b"WAVE":
            chunks = dict(wav_chunks(file))
            if b"fmt " not in chunks or b"data" not in chunks:
                raise Problem(f"{path}: a WAV file without fmt or data")
            channels, rate = struct.unpack("<HI", chunks[b"fmt "][0][2:8])
            bits = struct.unpack("<H", chunks[b"fmt "][0][14:16])[0]
            frame_size = channels * ((bits + 7) // 8)
            return AudioInfo(rate, channels, bits, chunks[b"data"][1] // frame_size)
    raise Problem(f"{path}: neither FLAC nor WAV")


def wav_chunks(file):
    """Yields (id, (body, size)) for the chunks of the RIFF WAVE file open at `file`; the body
    is read for every chunk but the sample data, whose size alone is given."""
    file.seek(12)
    while len(header := file.read(8)) == 8:
        chunk, size = header[:4], struct.unpack("<I", header[4:])[0]
        if chunk == b"data":
            yield chunk, (b"", size)
            file.seek(size + (size & 1), os.SEEK_CUR)
        else:
            body = file.read(size)
            file.read(size & 1)
            yield chunk, (body, size)


def raw_url(repository, commit, path):
    """The URL of one file of a GitHub repository at a commit."""
    return (f"https://raw.githubusercontent.com/{repository}/{commit}/"
            f"{urllib.parse.quote(path)}")


def note_number(value):
    """A MIDI key from an SFZ key value: a number, or a name such as c4, c#4 or db4, where
    c4 is 60 (SFZ's convention, and Dawsome's: its typing keyboard calls key 60 C4)."""
    text = value.strip().lower()
    if re.fullmatch(r"-?\d+", text):
        number = int(text)
    else:
        match = re.fullmatch(r"([a-g])([#b]?)(-?\d+)", text)
        if match is None:
            raise Problem(f"{value!r} is neither a key number nor a note name")
        letter, accidental, octave = match.groups()
        number = (int(octave) + 1) * 12 + NOTE_NAMES[letter] + {"#": 1, "b": -1, "": 0}[accidental]
    if not 0 <= number <= 127:
        raise Problem(f"key {value!r} is outside 0..127")
    return number


@dataclass
class Sfz:
    """An SFZ file as its regions see it. `regions` holds each region's own opcodes merged over
    its <global>, <master> and <group> (in that order, the later winning), with values as
    written. `used` names every header, directive and opcode the file and its includes use, so
    a check can compare it with the subset."""

    control: dict = field(default_factory=dict)
    regions: list = field(default_factory=list)
    used: set = field(default_factory=set)

    def sample_path(self, region):
        """The region's sample path relative to the .sfz file's folder."""
        return self.control.get("default_path", "") + region["sample"]


def read_sfz(path):
    """Reads the SFZ file at `path` and the files it includes. #define variables are replaced
    and #include paths are read relative to the first file's folder, as SFZ players do."""
    path = Path(path)
    sfz = Sfz()
    scopes = {"global": {}, "master": {}, "group": {}}
    state = {"header": None, "region": None, "defines": {}}

    def close_region():
        if state["region"] is not None:
            merged = {**scopes["global"], **scopes["master"], **scopes["group"]}
            sfz.regions.append({**merged, **state["region"]})
            state["region"] = None

    def header(name):
        close_region()
        sfz.used.add(f"<{name}>")
        state["header"] = name
        if name == "region":
            state["region"] = {}
        elif name in scopes:
            # A <global> starts over its masters and groups, a <master> its groups.
            for scope in ("global", "master", "group")[("global", "master", "group").index(name):]:
                scopes[scope] = {}

    def opcode(name, value):
        sfz.used.add(name)
        if state["region"] is not None:
            state["region"][name] = value
        elif state["header"] == "control":
            sfz.control[name] = value
        elif state["header"] in scopes:
            scopes[state["header"]][name] = value
        # Opcodes under other headers (<curve>, <effect>, ...) do not reach regions.

    def read(file, depth):
        if depth > _INCLUDE_DEPTH:
            raise Problem(f"{path}: #include goes deeper than {_INCLUDE_DEPTH} files")
        try:
            raw = file.read_bytes()
        except OSError as error:
            raise Problem(f"{path}: cannot read {file}: {error}") from None
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            text = raw.decode("latin-1")
        text = re.sub(r"/\*.*?\*/", " ", text, flags=re.DOTALL)
        for line in text.splitlines():
            line = line.split("//", 1)[0]
            if match := _DEFINE.match(line):
                sfz.used.add("#define")
                state["defines"][match[1]] = match[2]
                continue
            # Variables before #include too: libraries write #include "$DIR/$DYN.txt".
            line = _VARIABLE.sub(lambda m: state["defines"].get(m[0], m[0]), line)
            if match := _INCLUDE.match(line):
                sfz.used.add("#include")
                read(path.parent / match[1], depth + 1)
                continue
            tokens = list(_TOKEN.finditer(line))
            for token, after in zip(tokens, tokens[1:] + [None]):
                if token[1]:
                    header(token[1])
                else:
                    end = after.start() if after else len(line)
                    opcode(token[2], line[token.end():end].strip())

    read(path, 0)
    close_region()
    return sfz
