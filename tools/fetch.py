#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Downloads the kit's sources, checks each against its pinned SHA-256, and extracts them.

usage: python3 -I tools/fetch.py [--downloads DIR] [--check | --pin] [SOURCE ...]
       --downloads DIR: where downloads go (default: downloads/ in the repository)
       --check: download nothing; exit 1 unless every pinned file is on disk with its hash
       --pin: download the entries of sources.json that have no hash yet and record the
              hash seen (trust on first use: read the source's license page first)
       SOURCE: only these sources of sources.json (default: all)

Each source gets its own folder, downloads/<source>/. An archive source is saved there under
its own file name, checked, and extracted into downloads/<source>/extracted/; a repository
source (files of a Git repository at a pinned commit, fetched one by one so a subset of a
6 GB library costs only what it uses) lands file by file in that same extracted/ folder.
Running it again redoes nothing that is already there and right, and stops at anything on
disk that does not match its pin rather than overwriting it.

Downloads are untrusted data: nothing in them is run or imported. Archives are read with
Python's own modules (and sevenzip.py for .7z), every member name is checked before anything
is written (no absolute paths, no "..", only plain files and folders), and members are written
into a new folder that is renamed into place once complete.

Exit 0 when everything asked for is done or checked; 1 when a check fails; 2 when the check
cannot run (unreadable sources.json, network failure). Standard library only.
"""

import argparse
import datetime
import hashlib
import os
import shutil
import stat
import sys
import tarfile
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

# -I keeps the script's own folder off sys.path (so a planted module in the working folder
# is never imported); this adds back exactly the repository's folder for kitlib and sevenzip.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import kitlib  # noqa: E402
import sevenzip  # noqa: E402
from kitlib import Problem  # noqa: E402

CHUNK = 1 << 20
TIMEOUT = 60
ATTEMPTS = 3
USER_AGENT = "dawsome-kit-fetch/1 (+https://github.com/Lil-Pink/dawsome-kit)"
ARCHIVE_SUFFIXES = (".7z", ".zip", ".tar", ".tar.gz", ".tgz", ".tar.xz", ".tar.bz2")
EXTRACTED = "extracted"
# Written beside extracted/ once an extraction is complete: the SHA-256 of the archive it came
# from, so a later run knows the folder is whole and from the pinned archive.
EXTRACTED_MARK = "extracted.sha256"


class Mismatch(Problem):
    """Something on disk or downloaded differs from its pin."""


def archive_name(url):
    """The file name an archive is saved under: the last part of its URL, checked to be a
    plain name with an archive suffix fetch.py can open."""
    name = urllib.parse.unquote(PurePosixPath(urllib.parse.urlsplit(url).path).name)
    if not name or name.startswith(".") or "/" in name or "\\" in name:
        raise Problem(f"{url}: no usable file name at the end of the URL")
    if not name.lower().endswith(ARCHIVE_SUFFIXES):
        raise Problem(f"{url}: {name} is not one of {', '.join(ARCHIVE_SUFFIXES)}")
    return name


def safe_relative(name):
    """The relative path an archive member or repository path may be written to, or None for
    the archive's own root ("./"). Refuses absolute paths, drive letters, ".." and NUL."""
    text = name.replace("\\", "/")
    if "\0" in text or text.startswith("/") or (len(text) > 1 and text[1] == ":"):
        raise Problem(f"{name!r} is an absolute path")
    parts = [part for part in text.split("/") if part not in ("", ".")]
    if ".." in parts:
        raise Problem(f"{name!r} climbs out of its folder")
    return PurePosixPath(*parts) if parts else None


def validate_sources(sources, pinning):
    """Problems with sources.json itself: every source needs a CC0 license, its license page
    and a quote from it, and exactly one of an archive or a repository with files; every
    entry needs a hash unless `pinning`."""
    found = []
    for name, source in sources.items():
        where = f"sources.json: {name}"
        if source.get("licence") != kitlib.LICENCE:
            found.append(f"{where}: license is {source.get('licence')!r}, not {kitlib.LICENCE}")
        if not str(source.get("licence_page", "")).startswith("https://"):
            found.append(f"{where}: licence_page is not an https URL")
        if not source.get("licence_quote"):
            found.append(f"{where}: no licence_quote from the license page")
        has_archive, has_files = "archive" in source, "files" in source
        if has_archive == has_files:
            found.append(f"{where}: give exactly one of archive or repository + files")
            continue
        entries = [source["archive"]] if has_archive else source["files"]
        if has_files and not ("repository" in source and "commit" in source):
            found.append(f"{where}: files need a repository and a commit")
        if not pinning and any(not entry.get("sha256") for entry in entries):
            found.append(f"{where}: unpinned entries; read the license page, then run --pin")
    return found


def download(url, target, expected=None):
    """Downloads `url` to `target` through `target`.part and returns (sha256, size). With
    `expected`, a different hash removes the partial file and raises Problem."""
    if not url.startswith("https://"):
        raise Problem(f"{url}: only https downloads are allowed")
    partial = target.with_name(target.name + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(1, ATTEMPTS + 1):
        try:
            with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
                if not response.url.startswith("https://"):
                    raise Problem(f"{url}: redirected to {response.url}, which is not https")
                digest, size = hashlib.sha256(), 0
                with open(partial, "wb") as out:
                    while chunk := response.read(CHUNK):
                        out.write(chunk)
                        digest.update(chunk)
                        size += len(chunk)
            break
        except urllib.error.HTTPError as error:
            if error.code < 500 or attempt == ATTEMPTS:
                raise Problem(f"{url}: HTTP {error.code} {error.reason}") from None
        except (urllib.error.URLError, OSError) as error:
            if attempt == ATTEMPTS:
                raise Problem(f"{url}: {error}") from None
        time.sleep(2 ** attempt)
    sha256 = digest.hexdigest()
    if expected and sha256 != expected:
        partial.unlink()
        raise Mismatch(f"{url}: SHA-256 {sha256}, but the pin is {expected}")
    os.replace(partial, target)
    return sha256, size


def _tar_kind(member):
    return "dir" if member.isdir() else "file" if member.isfile() else "link or device"


def _zip_kind(info):
    if stat.S_ISLNK(info.external_attr >> 16):
        return "link or device"
    return "dir" if info.is_dir() else "file"


def _chunks(file):
    while chunk := file.read(CHUNK):
        yield chunk


def archive_members(path):
    """(name, kind) of every member, kind being "file", "dir" or what else it is."""
    lower = path.name.lower()
    if lower.endswith(".7z"):
        with sevenzip.SevenZipArchive(path) as archive:
            return [(member.name, "link or device" if member.is_link else
                     "dir" if member.is_dir else "file") for member in archive.members]
    if lower.endswith(".zip"):
        with zipfile.ZipFile(path) as archive:
            return [(info.filename, _zip_kind(info)) for info in archive.infolist()]
    with tarfile.open(path, "r|*") as archive:
        return [(member.name, _tar_kind(member)) for member in archive]


def archive_contents(path):
    """Yields (name, kind, chunks) for every member in archive order."""
    lower = path.name.lower()
    if lower.endswith(".7z"):
        with sevenzip.SevenZipArchive(path) as archive:
            for member, chunks in archive.contents():
                yield member.name, "dir" if member.is_dir else "file", chunks
    elif lower.endswith(".zip"):
        with zipfile.ZipFile(path) as archive:
            for info in archive.infolist():
                with archive.open(info) as file:
                    yield info.filename, _zip_kind(info), _chunks(file)
    else:
        with tarfile.open(path, "r|*") as archive:
            for member in archive:
                file = archive.extractfile(member) if member.isfile() else None
                yield member.name, _tar_kind(member), _chunks(file) if file else iter(())


def extract(archive, destination):
    """Extracts `archive` into the folder `destination`, which must not exist. Every member is
    checked first; the files are written into `destination`.partial (removed first if an
    interrupted run left it) and renamed into place when all are written."""
    for name, kind in archive_members(archive):
        if kind not in ("file", "dir"):
            raise Problem(f"{archive.name}: {name!r} is a {kind}; only files and folders are"
                          " extracted")
        if safe_relative(name) is None and kind == "file":
            raise Problem(f"{archive.name}: a file named {name!r}")
    staging = destination.with_name(destination.name + ".partial")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir()
    for name, kind, chunks in archive_contents(archive):
        relative = safe_relative(name)
        if relative is None:
            continue
        target = staging.joinpath(relative)
        if kind == "dir":
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        # "x": a second member with the same name is refused, never written over the first.
        with open(target, "xb") as out:
            for chunk in chunks:
                out.write(chunk)
    os.replace(staging, destination)


def pin_entry(entry, sha256, path):
    """Records the hash seen for an unpinned entry; True when it did. validate_sources lets an
    entry through unpinned only with --pin, so this pins only when asked to."""
    if entry.get("sha256"):
        return False
    entry.update(sha256=sha256, size=path.stat().st_size)
    return True


def verified(path, entry):
    """The SHA-256 of `path`, which must equal the entry's pin when it has one."""
    sha256 = kitlib.sha256_file(path)
    if entry.get("sha256") and sha256 != entry["sha256"]:
        raise Mismatch(f"{path}: SHA-256 {sha256}, but the pin is {entry['sha256']};"
                       " remove the file to download it again")
    return sha256


def fetch_archive(folder, entry):
    """Downloads, checks and extracts one archive source into `folder`; True when it pinned
    the archive."""
    archive = folder / archive_name(entry["url"])
    if archive.exists():
        sha256 = verified(archive, entry)
        print(f"fetch: {archive.name} already downloaded")
    else:
        print(f"fetch: downloading {entry['url']}")
        sha256, _ = download(entry["url"], archive, entry.get("sha256"))
    pinned = pin_entry(entry, sha256, archive)
    mark = folder / EXTRACTED_MARK
    extracted = folder / EXTRACTED
    if extracted.exists():
        if mark.exists() and mark.read_text().strip() == entry["sha256"]:
            return pinned
        raise Mismatch(f"{extracted} is not marked as extracted from the pinned archive;"
                       " remove it to extract again")
    print(f"fetch: extracting {archive.name}")
    try:
        extract(archive, extracted)
    except (sevenzip.SevenZipError, tarfile.TarError, zipfile.BadZipFile, OSError) as error:
        raise Problem(f"{archive}: {error}") from None
    mark.write_text(entry["sha256"] + "\n")
    return pinned


def repository_file(folder, entry):
    """Where a repository source's file lives under `folder`."""
    relative = safe_relative(entry["path"])
    if relative is None:
        raise Problem(f"sources.json: {entry['path']!r} names no file")
    return folder / EXTRACTED / relative


def fetch_files(folder, source):
    """Downloads and checks each file of a repository source into `folder`/extracted; True
    when it pinned any."""
    fetched, pinned = 0, False
    for entry in source["files"]:
        target = repository_file(folder, entry)
        if target.exists():
            sha256 = verified(target, entry)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            url = kitlib.raw_url(source["repository"], source["commit"], entry["path"])
            sha256, _ = download(url, target, entry.get("sha256"))
            fetched += 1
        pinned |= pin_entry(entry, sha256, target)
    print(f"fetch: {len(source['files'])} files, {fetched} downloaded now")
    return pinned


def check_source(folder, source):
    """Problems found comparing what is on disk for one source with its pins."""
    if "archive" in source:
        entry = source["archive"]
        archive = folder / archive_name(entry["url"])
        if not archive.exists():
            return [f"{archive}: not downloaded"]
        found = []
        if kitlib.sha256_file(archive) != entry["sha256"]:
            found.append(f"{archive}: SHA-256 differs from the pin")
        mark = folder / EXTRACTED_MARK
        if not mark.exists() or mark.read_text().strip() != entry["sha256"]:
            found.append(f"{folder / EXTRACTED}: not extracted from the pinned archive")
        return found
    found = []
    for entry in source["files"]:
        target = repository_file(folder, entry)
        if not target.exists():
            found.append(f"{target}: not downloaded")
        elif kitlib.sha256_file(target) != entry["sha256"]:
            found.append(f"{target}: SHA-256 differs from the pin")
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--downloads", type=Path, default=kitlib.DOWNLOADS)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--pin", action="store_true")
    parser.add_argument("source", nargs="*")
    args = parser.parse_args()
    try:
        document = kitlib.read_json(kitlib.SOURCES)
        sources = document["sources"]
        unknown = sorted(set(args.source) - set(sources))
        if unknown:
            raise Problem(f"not in sources.json: {', '.join(unknown)}")
        found = validate_sources(sources, args.pin)
        if found:
            print("fetch: sources.json is not ready:", *found, sep="\n  ", file=sys.stderr)
            return 1
        chosen = args.source or list(sources)
        if args.check:
            found = [problem for name in chosen
                     for problem in check_source(args.downloads / name, sources[name])]
            if found:
                print("fetch: the downloads do not match the pins:", *found, sep="\n  ",
                      file=sys.stderr)
                return 1
            print(f"fetch: {len(chosen)} sources match their pins")
            return 0
        for name in chosen:
            source = sources[name]
            folder = args.downloads / name
            folder.mkdir(parents=True, exist_ok=True)
            print(f"fetch: {name}")
            if "archive" in source:
                pinned = fetch_archive(folder, source["archive"])
            else:
                pinned = fetch_files(folder, source)
            if pinned:
                source["pinned"] = datetime.date.today().isoformat()
                kitlib.write_json(kitlib.SOURCES, document)
    except Mismatch as error:
        print(f"fetch: {error}", file=sys.stderr)
        return 1
    except Problem as error:
        print(f"fetch: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
