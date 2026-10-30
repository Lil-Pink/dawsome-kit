# SPDX-License-Identifier: GPL-3.0-or-later
"""Reads 7z archives with the Python standard library only, for fetch.py.

FreePats publishes most of its sound banks as .7z files, and Python has no 7z module. This
reads the part of the format that 7-Zip and p7zip write for plain files: the LZMA and LZMA2
methods with or without the Delta or x86 BCJ filter in front (through the standard `lzma`
module), Deflate (`zlib`), BZip2 (`bz2`) and Copy, solid or not, with the header compressed
or not. Anything else (another method,
encryption, coders with several streams, data kept outside the header) raises SevenZipError
naming what it found instead of guessing. Each file's size and CRC-32 are checked as it is
read. The format is described in DOC/7zFormat.txt of the 7-Zip sources.

Member names come back exactly as stored: deciding where, and whether, to write them is the
caller's job.

    with SevenZipArchive(path) as archive:
        archive.members                      # every member, before any data is read
        for member, chunks in archive.contents():
            ...                              # chunks yields the member's bytes
"""

import bz2
import lzma
import os
import stat
import struct
import zlib
from dataclasses import dataclass, field

SIGNATURE = b"7z\xbc\xaf\x27\x1c"
START_HEADER_SIZE = 32
CHUNK = 1 << 20
# A compressed header bigger than this is damage or a trap, not a list of sound files.
MAX_HEADER = 1 << 26

# Property ids, from 7zFormat.txt.
K_END = 0x00
K_HEADER = 0x01
K_ARCHIVE_PROPERTIES = 0x02
K_ADDITIONAL_STREAMS_INFO = 0x03
K_MAIN_STREAMS_INFO = 0x04
K_FILES_INFO = 0x05
K_PACK_INFO = 0x06
K_UNPACK_INFO = 0x07
K_SUBSTREAMS_INFO = 0x08
K_SIZE = 0x09
K_CRC = 0x0A
K_FOLDER = 0x0B
K_CODERS_UNPACK_SIZE = 0x0C
K_NUM_UNPACK_STREAM = 0x0D
K_EMPTY_STREAM = 0x0E
K_EMPTY_FILE = 0x0F
K_ANTI = 0x10
K_NAME = 0x11
K_WIN_ATTRIBUTES = 0x15
K_ENCODED_HEADER = 0x17

# Method ids.
COPY = b"\x00"
DELTA = b"\x03"
LZMA = b"\x03\x01\x01"
LZMA2 = b"\x21"
BCJ_X86 = b"\x03\x03\x01\x03"
DEFLATE = b"\x04\x01\x08"
BZIP2 = b"\x04\x02\x02"
# Method names, so an error says what the archive needs.
METHOD_NAMES = {
    COPY: "Copy", DELTA: "Delta", LZMA: "LZMA", LZMA2: "LZMA2", BCJ_X86: "BCJ (x86)",
    DEFLATE: "Deflate", BZIP2: "BZip2",
    b"\x04\x01\x09": "Deflate64",
    b"\x03\x04\x01": "PPMd",
    b"\x03\x03\x01\x1b": "BCJ2",
    b"\x06\xf1\x07\x01": "AES (the archive is encrypted)",
}

FILE_ATTRIBUTE_REPARSE_POINT = 0x400
# Set by p7zip and 7-Zip on Unix: the high 16 bits then hold the file's st_mode.
FILE_ATTRIBUTE_UNIX_EXTENSION = 0x8000


class SevenZipError(Exception):
    """The archive is damaged, or uses a part of the format this reader does not handle."""


@dataclass(frozen=True)
class Member:
    """One file or folder of the archive. `folder` is the index of the solid block holding its
    data, None when it has none (folders and empty files). `is_link` marks a symbolic link or
    reparse point, which the caller should refuse."""

    name: str
    is_dir: bool
    size: int
    crc: int | None
    is_link: bool
    folder: int | None


@dataclass
class _Coder:
    method: bytes
    properties: bytes
    in_streams: int
    out_streams: int


@dataclass
class _Folder:
    """A solid block: coders joined by bind pairs, fed by one or more packed streams."""

    coders: list
    bind_pairs: list  # (in stream index, out stream index): that output feeds that input
    packed: list  # the in stream indexes read from packed streams
    unpack_sizes: list = field(default_factory=list)  # one per out stream
    crc: int | None = None

    def main_out(self):
        """The out stream no bind pair consumes: the folder's unpacked data."""
        bound = {out_index for _, out_index in self.bind_pairs}
        free = [index for index in range(len(self.unpack_sizes)) if index not in bound]
        if len(free) != 1:
            raise SevenZipError("a folder's coders do not end in one output")
        return free[0]

    @property
    def size(self):
        return self.unpack_sizes[self.main_out()]


@dataclass
class _Streams:
    pack_pos: int = 0
    pack_sizes: list = field(default_factory=list)
    folders: list = field(default_factory=list)
    # Per folder, the (size, crc) of each file stream in it, in order.
    substreams: list = field(default_factory=list)


class _Bytes:
    """A cursor over header bytes."""

    def __init__(self, data):
        self.data = data
        self.pos = 0

    def take(self, count):
        if count < 0 or self.pos + count > len(self.data):
            raise SevenZipError("the header ends early")
        part = self.data[self.pos:self.pos + count]
        self.pos += count
        return part

    def rest(self):
        return self.take(len(self.data) - self.pos)

    def byte(self):
        return self.take(1)[0]

    def number(self):
        """7z's variable-length number: the leading 1 bits of the first byte count the bytes
        that follow it (least significant first); its remaining bits are the top bits."""
        first = self.byte()
        value = 0
        for index in range(8):
            mask = 0x80 >> index
            if not first & mask:
                return value | ((first & (mask - 1)) << (8 * index))
            value |= self.byte() << (8 * index)
        return value

    def count(self):
        """A number used as a count or index. Never more than the header has bytes, so a
        damaged header cannot ask for a huge list."""
        value = self.number()
        if value > len(self.data):
            raise SevenZipError(f"the header names {value} items in {len(self.data)} bytes")
        return value

    def uint32(self):
        return struct.unpack("<I", self.take(4))[0]

    def bits(self, count):
        """`count` booleans packed most significant bit first."""
        data = self.take((count + 7) // 8)
        return [bool(data[index // 8] & (0x80 >> (index % 8))) for index in range(count)]

    def defined(self, count):
        """`count` booleans after an "all are defined" byte."""
        return [True] * count if self.byte() else self.bits(count)

    def digests(self, count):
        """`count` CRC-32s, None where the archive gives none."""
        return [self.uint32() if is_defined else None for is_defined in self.defined(count)]

    def expect(self, property_id, where):
        found = self.byte()
        if found != property_id:
            raise SevenZipError(f"{where}: property {found:#x} where {property_id:#x} belongs")


def _method_name(method):
    return METHOD_NAMES.get(method, f"with id {method.hex() or '(empty)'}")


def _read_pack_info(header):
    pack_pos = header.number()
    count = header.count()
    sizes = None
    while (prop := header.byte()) != K_END:
        if prop == K_SIZE:
            sizes = [header.number() for _ in range(count)]
        elif prop == K_CRC:
            header.digests(count)  # the unpacked files' own CRCs are checked instead
        else:
            raise SevenZipError(f"pack info: unexpected property {prop:#x}")
    if sizes is None:
        raise SevenZipError("pack info gives no sizes")
    return pack_pos, sizes


def _read_folder(header):
    coders = []
    for _ in range(header.count()):
        flags = header.byte()
        if flags & 0xC0:
            raise SevenZipError("a coder uses alternative methods, which 7z no longer writes")
        method = header.take(flags & 0x0F)
        in_streams, out_streams = (header.count(), header.count()) if flags & 0x10 else (1, 1)
        properties = header.take(header.count()) if flags & 0x20 else b""
        coders.append(_Coder(method, properties, in_streams, out_streams))
    total_in = sum(coder.in_streams for coder in coders)
    total_out = sum(coder.out_streams for coder in coders)
    if total_out < 1:
        raise SevenZipError("a folder has no coders")
    bind_pairs = [(header.count(), header.count()) for _ in range(total_out - 1)]
    packed_count = total_in - len(bind_pairs)
    if packed_count < 1:
        raise SevenZipError("a folder reads no packed stream")
    if packed_count == 1:
        bound = {in_index for in_index, _ in bind_pairs}
        packed = [index for index in range(total_in) if index not in bound]
    else:
        packed = [header.count() for _ in range(packed_count)]
    return _Folder(coders, bind_pairs, packed, [0] * total_out)


def _read_unpack_info(header):
    header.expect(K_FOLDER, "unpack info")
    count = header.count()
    if header.byte():
        raise SevenZipError("folders kept in another stream are not supported")
    folders = [_read_folder(header) for _ in range(count)]
    header.expect(K_CODERS_UNPACK_SIZE, "unpack info")
    for folder in folders:
        folder.unpack_sizes = [header.number() for _ in folder.unpack_sizes]
    while (prop := header.byte()) != K_END:
        if prop != K_CRC:
            raise SevenZipError(f"unpack info: unexpected property {prop:#x}")
        for folder, crc in zip(folders, header.digests(len(folders))):
            folder.crc = crc
    return folders


def _read_substreams_info(header, folders):
    counts = [1] * len(folders)
    prop = header.byte()
    if prop == K_NUM_UNPACK_STREAM:
        counts = [header.count() for _ in folders]
        prop = header.byte()
    sizes = []
    for folder, count in zip(folders, counts):
        known = []
        if count > 1:
            if prop != K_SIZE:
                raise SevenZipError("a solid block holds several files but gives no sizes")
            known = [header.number() for _ in range(count - 1)]
        last = folder.size - sum(known)
        if last < 0:
            raise SevenZipError("the files of a solid block are bigger than the block")
        sizes.append(known + [last] if count else [])
    if prop == K_SIZE:
        prop = header.byte()
    # A block holding one file with a known CRC lends that CRC to the file; the rest are listed.
    lent = [count == 1 and folder.crc is not None for folder, count in zip(folders, counts)]
    crcs = [[folder.crc] if lends else [None] * count
            for folder, count, lends in zip(folders, counts, lent)]
    while prop != K_END:
        if prop != K_CRC:
            raise SevenZipError(f"substreams info: unexpected property {prop:#x}")
        listed = iter(header.digests(sum(c for c, lends in zip(counts, lent) if not lends)))
        crcs = [crc if lends else [next(listed) for _ in crc] for crc, lends in zip(crcs, lent)]
        prop = header.byte()
    return [list(zip(size, crc)) for size, crc in zip(sizes, crcs)]


def _read_streams_info(header):
    streams = _Streams()
    prop = header.byte()
    if prop == K_PACK_INFO:
        streams.pack_pos, streams.pack_sizes = _read_pack_info(header)
        prop = header.byte()
    if prop == K_UNPACK_INFO:
        streams.folders = _read_unpack_info(header)
        prop = header.byte()
    if prop == K_SUBSTREAMS_INFO:
        streams.substreams = _read_substreams_info(header, streams.folders)
        prop = header.byte()
    else:
        streams.substreams = [[(folder.size, folder.crc)] for folder in streams.folders]
    if prop != K_END:
        raise SevenZipError(f"streams info: unexpected property {prop:#x}")
    if sum(len(folder.packed) for folder in streams.folders) != len(streams.pack_sizes):
        raise SevenZipError("the folders and the packed streams do not match")
    return streams


def _is_link(attributes):
    if attributes is None:
        return False
    if attributes & FILE_ATTRIBUTE_REPARSE_POINT:
        return True
    return bool(attributes & FILE_ATTRIBUTE_UNIX_EXTENSION) and stat.S_ISLNK(attributes >> 16)


def _read_files_info(header, substreams):
    count = header.count()
    empty_stream = [False] * count
    empty_file = None
    names = None
    attributes = [None] * count
    while (prop := header.byte()) != K_END:
        body = _Bytes(header.take(header.count()))
        if prop == K_EMPTY_STREAM:
            empty_stream = body.bits(count)
        elif prop == K_EMPTY_FILE:
            empty_file = body.bits(sum(empty_stream))
        elif prop == K_ANTI:
            if any(body.bits(sum(empty_stream))):
                raise SevenZipError("deletion markers (anti-items) are not supported")
        elif prop == K_NAME:
            if body.byte():
                raise SevenZipError("names kept in another stream are not supported")
            try:
                names = body.rest().decode("utf-16-le").split("\0")
            except UnicodeDecodeError:
                raise SevenZipError("the file names are not valid UTF-16") from None
            if len(names) != count + 1 or names.pop():
                raise SevenZipError("the file names do not match the file count")
        elif prop == K_WIN_ATTRIBUTES:
            defined = body.defined(count)
            if body.byte():
                raise SevenZipError("attributes kept in another stream are not supported")
            attributes = [body.uint32() if is_defined else None for is_defined in defined]
        # Times, start positions and padding are not needed to extract the files.
    if names is None:
        raise SevenZipError("the archive gives no file names")
    streams = iter([(index, size, crc)
                    for index, files in enumerate(substreams) for size, crc in files])
    empty_files = iter(empty_file or [])
    members = []
    for name, is_empty, attribute in zip(names, empty_stream, attributes):
        if is_empty:
            is_file = next(empty_files, False)
            members.append(Member(name, not is_file, 0, None, _is_link(attribute), None))
            continue
        folder, size, crc = next(streams, (None, None, None))
        if folder is None:
            raise SevenZipError("the archive lists more files than it has data for")
        members.append(Member(name, False, size, crc, _is_link(attribute), folder))
    if next(streams, None) is not None:
        raise SevenZipError("the archive has data for more files than it lists")
    return members


def _lzma2_dict_size(code):
    if code > 40:
        raise SevenZipError(f"LZMA2 dictionary code {code} is out of range")
    return 0xFFFFFFFF if code == 40 else (2 | (code & 1)) << (code // 2 + 11)


def _filter(coder):
    """The `lzma` module filter that undoes `coder`."""
    method, properties = coder.method, coder.properties
    if method == LZMA2 and len(properties) == 1:
        return {"id": lzma.FILTER_LZMA2, "dict_size": _lzma2_dict_size(properties[0])}
    if method == LZMA and len(properties) == 5 and properties[0] < 9 * 5 * 5:
        lc, lp, pb = properties[0] % 9, properties[0] // 9 % 5, properties[0] // 45
        # 7-Zip shrinks the dictionary to the data size; liblzma refuses one under 4 KiB, and a
        # bigger dictionary decodes the same stream.
        dict_size = max(struct.unpack("<I", properties[1:])[0], 4096)
        return {"id": lzma.FILTER_LZMA1, "lc": lc, "lp": lp, "pb": pb, "dict_size": dict_size}
    if method == DELTA and len(properties) == 1:
        return {"id": lzma.FILTER_DELTA, "dist": properties[0] + 1}
    if method == BCJ_X86 and not properties:
        return {"id": lzma.FILTER_X86}
    if method in (LZMA2, LZMA, DELTA, BCJ_X86):
        raise SevenZipError(f"a coder has properties {properties.hex()!r} its method cannot have")
    raise SevenZipError(f"the compression method {_method_name(method)} is not supported")


def _chain(folder):
    """The folder's coders in filter order: from the one giving the folder's output down to the
    one reading the packed stream. Only chains of one-input, one-output coders are supported;
    with those, coder i owns in stream i and out stream i."""
    for coder in folder.coders:
        if coder.in_streams != 1 or coder.out_streams != 1:
            raise SevenZipError(f"the method {_method_name(coder.method)} uses several streams,"
                                " which is not supported")
    fed_by = {in_index: out_index for in_index, out_index in folder.bind_pairs}
    order = [folder.main_out()]
    while order[-1] in fed_by:
        order.append(fed_by[order[-1]])
        if len(order) > len(folder.coders):
            raise SevenZipError("a folder's coders loop")
    if len(order) != len(folder.coders) or folder.packed != [order[-1]]:
        raise SevenZipError("a folder's coders do not form one chain")
    return [folder.coders[index] for index in order]


class _Inflate:
    """zlib's raw Deflate decoder with the interface of lzma's and bz2's decompressors."""

    def __init__(self):
        self._zlib = zlib.decompressobj(-zlib.MAX_WBITS)

    @property
    def needs_input(self):
        return not self._zlib.unconsumed_tail

    @property
    def eof(self):
        return self._zlib.eof

    def decompress(self, data, max_length):
        return self._zlib.decompress(self._zlib.unconsumed_tail + data, max_length)


def _decompressor(folder):
    """A decompressor for the folder's coders, or None when they only copy."""
    chain = [coder for coder in _chain(folder) if coder.method != COPY]
    if not chain:
        return None
    if len(chain) == 1 and chain[0].method == DEFLATE:
        return _Inflate()
    if len(chain) == 1 and chain[0].method == BZIP2:
        return bz2.BZ2Decompressor()
    filters = [_filter(coder) for coder in chain]
    if filters[-1]["id"] not in (lzma.FILTER_LZMA1, lzma.FILTER_LZMA2):
        raise SevenZipError("a filter is used without LZMA or LZMA2 behind it")
    # The x86 filter holds back its last bytes until the stream says it has ended, and
    # LZMA streams in 7z carry no end marker (LZMA2 streams do), so those bytes never come.
    if filters[-1]["id"] == lzma.FILTER_LZMA1 and any(
            step["id"] == lzma.FILTER_X86 for step in filters):
        raise SevenZipError("the x86 BCJ filter in front of LZMA (not LZMA2) is not supported")
    return lzma.LZMADecompressor(lzma.FORMAT_RAW, filters=filters)


class _Stream:
    """Reads a bounded number of bytes at a time from a generator of chunks."""

    def __init__(self, chunks):
        self._chunks = chunks
        self._buffer = b""

    def read(self, limit):
        """Up to `limit` bytes; empty only at the end of the data."""
        if not self._buffer:
            self._buffer = next(self._chunks, b"")
        part, self._buffer = self._buffer[:limit], self._buffer[limit:]
        return part


class SevenZipArchive:
    """An open 7z archive; `members` is read on opening, data only through `contents()`."""

    def __init__(self, path):
        self._file = open(path, "rb")
        try:
            self._size = os.fstat(self._file.fileno()).st_size
            self._streams, self.members = self._read_headers()
        except BaseException:
            self._file.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()

    def close(self):
        self._file.close()

    def _read_at(self, offset, count):
        if offset + count > self._size:
            raise SevenZipError("the archive is truncated")
        self._file.seek(offset)
        return self._file.read(count)

    def _read_headers(self):
        start = self._read_at(0, START_HEADER_SIZE) if self._size >= START_HEADER_SIZE else b""
        if start[:6] != SIGNATURE:
            raise SevenZipError("not a 7z archive")
        if start[6] != 0:
            raise SevenZipError(f"7z format version {start[6]}.{start[7]} is not supported")
        if zlib.crc32(start[12:]) != struct.unpack("<I", start[8:12])[0]:
            raise SevenZipError("the start header is damaged")
        offset, size, crc = struct.unpack("<QQI", start[12:])
        if size == 0:
            return _Streams(), []
        data = self._read_at(START_HEADER_SIZE + offset, size)
        if zlib.crc32(data) != crc:
            raise SevenZipError("the header is damaged")
        header = _Bytes(data)
        prop = header.byte()
        while prop == K_ENCODED_HEADER:
            header = _Bytes(self._decode_header(_read_streams_info(header)))
            prop = header.byte()
        if prop != K_HEADER:
            raise SevenZipError(f"the header starts with property {prop:#x}")
        prop = header.byte()
        if prop == K_ARCHIVE_PROPERTIES:
            while header.byte() != K_END:
                header.take(header.count())
            prop = header.byte()
        if prop == K_ADDITIONAL_STREAMS_INFO:
            raise SevenZipError("additional streams are not supported")
        streams, members = _Streams(), []
        if prop == K_MAIN_STREAMS_INFO:
            streams = _read_streams_info(header)
            prop = header.byte()
        if prop == K_FILES_INFO:
            members = _read_files_info(header, streams.substreams)
            prop = header.byte()
        if prop != K_END:
            raise SevenZipError(f"the header ends with property {prop:#x}")
        return streams, members

    def _decode_header(self, streams):
        """The bytes of a compressed header, every folder of `streams` in turn."""
        if sum(folder.size for folder in streams.folders) > MAX_HEADER:
            raise SevenZipError("the compressed header claims an implausible size")
        data = b""
        for index, folder in enumerate(streams.folders):
            part = b"".join(self._folder_chunks(streams, index))
            if folder.crc is not None and zlib.crc32(part) != folder.crc:
                raise SevenZipError("the header is damaged (CRC mismatch)")
            data += part
        return data

    def _folder_chunks(self, streams, index):
        """The unpacked bytes of folder `index`, exactly its size, in chunks."""
        folder = streams.folders[index]
        first = sum(len(earlier.packed) for earlier in streams.folders[:index])
        position = START_HEADER_SIZE + streams.pack_pos + sum(streams.pack_sizes[:first])
        left_in = streams.pack_sizes[first]
        if position + left_in > self._size:
            raise SevenZipError("the archive is truncated")
        decompressor = _decompressor(folder)
        wanted = folder.size
        if decompressor is None:
            if left_in != wanted:
                raise SevenZipError("a stored file's packed and unpacked sizes differ")
            while left_in:
                data = self._read_at(position, min(CHUNK, left_in))
                position += len(data)
                left_in -= len(data)
                yield data
            return
        produced = 0
        while produced < wanted:
            data = b""
            starving = decompressor.needs_input
            if starving and left_in:
                data = self._read_at(position, min(CHUNK, left_in))
                position += len(data)
                left_in -= len(data)
            try:
                out = decompressor.decompress(data, max_length=min(CHUNK, wanted - produced))
            except (lzma.LZMAError, zlib.error, OSError) as error:
                raise SevenZipError(f"the packed data is damaged: {error}") from None
            # Fed nothing because nothing was left, or at the stream's end: no output now
            # means none will come.
            if not out and not data and (starving or decompressor.eof):
                raise SevenZipError("the packed data ends early: the archive is damaged")
            produced += len(out)
            if out:
                yield out

    def contents(self):
        """Yields (member, chunks) for every member in archive order; `chunks` yields the
        member's bytes. A member's size and CRC-32 are checked when its last chunk is read; a
        member the caller leaves unread is read (and checked) before the next one."""
        stream, folder = None, None
        for member in self.members:
            if member.folder is None:
                yield member, iter(())
                continue
            if member.folder != folder:
                folder = member.folder
                stream = _Stream(self._folder_chunks(self._streams, folder))
            chunks = _member_chunks(stream, member)
            yield member, chunks
            for _ in chunks:
                pass


def _member_chunks(stream, member):
    left, crc = member.size, 0
    while left:
        part = stream.read(min(left, CHUNK))
        if not part:
            raise SevenZipError(f"{member.name}: the data ends early")
        crc = zlib.crc32(part, crc)
        left -= len(part)
        yield part
    if member.crc is not None and crc != member.crc:
        raise SevenZipError(f"{member.name}: CRC mismatch, the archive is damaged")
