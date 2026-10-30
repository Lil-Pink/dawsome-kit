#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Checks the kit as it would ship: the files under sets/ against MANIFEST.json, and each set
against the SFZ subset Dawsome reads.

usage: python3 -I tools/check.py [--manifest] [--sfz]
       --manifest: every file under sets/ is in MANIFEST.json with its size, its SHA-256, a
                   CC0-1.0 license and where it came from (a sample mixed from several
                   microphones' files, each of them), and nothing else is there; every
                   download set has a URL, a hash and a CC0-1.0 license
       --sfz: each set has a set.json that matches its files, and an instrument set one .sfz
              whose headers, directives and opcodes are all in kitlib's subset, with values
              in range, every region's sample present and every sample used
       (no option: both)

Exit 0 when everything passes; 1 when something does not, with every problem listed; 2 when
the check cannot run. Standard library only; nothing is downloaded.
"""

import argparse
import re
import sys
from pathlib import Path

# -I keeps the script's own folder off sys.path (so a planted module in the working folder
# is never imported); this adds back exactly the repository's folder for kitlib.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import kitlib  # noqa: E402
from kitlib import Problem  # noqa: E402

SHA256 = re.compile(r"[0-9a-f]{64}")
MADE_FROM_SOURCE = {"copied", "converted", "processed"}
# "generated": made by a script in this repository (its path and SHA-256 are recorded, and those
# of the scripts it imports under generator_uses), then converted like a downloaded source; one
# made from recordings names them under made_from (files sources.json pins for a CC0 source).
# "made": made outside this repository and committed as made; it names the recordings it was
# made from under made_from, held to sources.json the same way.
MADE_BY = MADE_FROM_SOURCE | {"written", "generated", "made"}
SET_FIELDS = ("name", "kind", "maker", "licence", "licence_page", "licence_quote", "source",
              "source_url", "size", "decoded_size", "files", "sample_rates", "channels")
DOWNLOAD_FIELDS = ("set", "name", "url", "sha256", "size", "licence", "licence_page",
                   "licence_quote")
# Allowed values: whole numbers in a range, or numbers in a range.
INTEGER_RANGES = {
    "lokey": (0, 127), "hikey": (0, 127), "key": (0, 127), "pitch_keycenter": (0, 127),
    "lovel": (0, 127), "hivel": (0, 127), "loop_start": (0, None), "loop_end": (0, None),
    "offset": (0, None), "end": (0, None), "seq_length": (1, 100), "seq_position": (1, 100),
    "transpose": (-127, 127),
}
NUMBER_RANGES = {
    "volume": (-144, 6), "pan": (-100, 100), "amp_veltrack": (-100, 100), "tune": (-100, 100),
    "ampeg_attack": (0, 100), "ampeg_release": (0, 100),
}


def https(value):
    return isinstance(value, str) and value.startswith("https://")


def files_under(folder, root):
    """Repository paths of everything under `folder` that is not a folder, links included."""
    return {path.relative_to(root).as_posix() for path in folder.rglob("*")
            if path.is_symlink() or not path.is_dir()}


def generator_use_problems(path, uses, root):
    """A generated sample's generator_uses: the other scripts its generator imports, each with
    the SHA-256 it had when the sample was made."""
    if uses is None:
        return []
    if not isinstance(uses, list) or not all(
            isinstance(use, dict) and use.get("path") and SHA256.fullmatch(str(use.get("sha256")))
            for use in uses):
        return [f"{path}: generator_uses needs a path and a SHA-256 for each script"]
    found = []
    for use in uses:
        script = root / use["path"]
        if not script.is_file():
            found.append(f"{path}: its generator uses {use['path']}, which is not in the"
                         " repository")
        elif kitlib.sha256_file(script) != use["sha256"]:
            found.append(f"{path}: {use['path']} has changed since it was made; rebuild the set")
    return found


def made_from_problems(path, made_from, root):
    """A generated sample's made_from: the recordings it was made from, all of them files that
    sources.json pins for a CC0 source, so the sample can be traced to them and removed with
    them."""
    if made_from is None:
        return []
    if not (isinstance(made_from, dict) and made_from.get("source") and https(made_from.get(
            "origin")) and https(made_from.get("licence_page")) and isinstance(
            made_from.get("members"), list) and made_from["members"]):
        return [f"{path}: made_from needs a source, an origin URL, a license page and members"]
    name = made_from["source"]
    listing = root / "sources.json"
    sources = kitlib.read_json(listing)["sources"] if listing.is_file() else {}
    if name not in sources:
        return [f"{path}: made_from {name} is not in sources.json"]
    if sources[name].get("licence") != kitlib.LICENCE:
        return [f"{path}: made_from {name} is licensed {sources[name].get('licence')}, not"
                f" {kitlib.LICENCE}"]
    pinned = {entry.get("path") for entry in sources[name].get("files", [])
              if SHA256.fullmatch(str(entry.get("sha256")))}
    return [f"{path}: made_from {member} is not a file sources.json pins for {name}"
            for member in made_from["members"] if member not in pinned]


def mixed_from_problems(path, entry):
    """A sample mixed from several microphones' files (a drum hit): mixed_from names each one's
    file with its origin URL, SHA-256 and share of the mix (gain in dB, pan -1 to 1), the first of
    them the entry's own member, so the sample can be traced to every file it holds."""
    mics = entry["mixed_from"]
    if not (isinstance(mics, list) and mics and all(
            isinstance(mic, dict) and mic.get("member") and https(mic.get("origin"))
            and SHA256.fullmatch(str(mic.get("origin_sha256")))
            and isinstance(mic.get("gain_db"), (int, float))
            and isinstance(mic.get("pan"), (int, float)) and -1 <= mic["pan"] <= 1
            for mic in mics)):
        return [f"{path}: mixed_from needs each microphone's member, origin URL, SHA-256, gain_db"
                " and pan"]
    first = mics[0]
    if (first["member"], first["origin"], first["origin_sha256"]) != (
            entry.get("member"), entry.get("origin"), entry.get("origin_sha256")):
        return [f"{path}: mixed_from does not start with the entry's own member"]
    return []


def check_manifest(root):
    """Problems with sets/ against MANIFEST.json under `root`."""
    manifest = kitlib.read_json(root / "MANIFEST.json")
    found, listed = [], {}
    for entry in manifest.get("files", []):
        path = entry.get("path", "")
        if path in listed:
            found.append(f"{path}: listed twice")
        listed[path] = entry
        if not path.startswith("sets/"):
            found.append(f"{path}: outside sets/")
            continue
        target = root / path
        if target.is_symlink() or not target.is_file():
            found.append(f"{path}: listed, but not a file in sets/")
            continue
        if entry.get("licence") != kitlib.LICENCE:
            found.append(f"{path}: license {entry.get('licence')!r}, not {kitlib.LICENCE}")
        if not https(entry.get("licence_page")):
            found.append(f"{path}: no https licence_page")
        if not entry.get("source"):
            found.append(f"{path}: no source")
        if entry.get("made_by") not in MADE_BY:
            found.append(f"{path}: made_by {entry.get('made_by')!r}")
        if entry.get("made_by") in MADE_FROM_SOURCE and not (
                https(entry.get("origin")) and SHA256.fullmatch(str(entry.get("origin_sha256")))
                and entry.get("member")):
            found.append(f"{path}: no origin URL, origin SHA-256 and member to trace it by")
        if "mixed_from" in entry:
            found += mixed_from_problems(path, entry)
        if entry.get("made_by") == "generated":
            script = root / str(entry.get("generator", ""))
            if not (entry.get("generator") and script.is_file() and entry.get("member")):
                found.append(f"{path}: generated, but no generator script in the repository and"
                             " member to trace it by")
            elif entry.get("generator_sha256") != kitlib.sha256_file(script):
                found.append(f"{path}: its generator {entry['generator']} has changed since it"
                             " was made; rebuild the set")
            found += generator_use_problems(path, entry.get("generator_uses"), root)
            found += made_from_problems(path, entry.get("made_from"), root)
        if entry.get("made_by") == "made":
            if entry.get("made_from") is None:
                found.append(f"{path}: made, but no made_from to trace it by")
            found += made_from_problems(path, entry.get("made_from"), root)
        if entry.get("size") != target.stat().st_size:
            found.append(f"{path}: {target.stat().st_size} bytes, the manifest says"
                         f" {entry.get('size')}")
        elif entry.get("sha256") != kitlib.sha256_file(target):
            found.append(f"{path}: SHA-256 differs from the manifest")
    sets = root / "sets"
    on_disk = files_under(sets, root) if sets.is_dir() else set()
    found += [f"{path}: in sets/ but not in the manifest" for path in sorted(on_disk - set(listed))]
    for download in manifest.get("downloads", []):
        where = f"download {download.get('set')!r}"
        missing = [name for name in DOWNLOAD_FIELDS if download.get(name) in (None, "")]
        if missing:
            found.append(f"{where}: no {', '.join(missing)}")
            continue
        if download["licence"] != kitlib.LICENCE:
            found.append(f"{where}: license {download['licence']!r}, not {kitlib.LICENCE}")
        if not (https(download["url"]) and https(download["licence_page"])):
            found.append(f"{where}: url and licence_page must be https")
        if not SHA256.fullmatch(str(download["sha256"])):
            found.append(f"{where}: sha256 is not a SHA-256")
    return found


def value_problems(name, value, where):
    """Problems with one opcode's value."""
    if name == "loop_mode":
        return [] if value in kitlib.LOOP_MODES else [f"{where}: loop_mode={value}"]
    if name in INTEGER_RANGES:
        low, high = INTEGER_RANGES[name]
        if not re.fullmatch(r"-?\d+", value) or int(value) < low or (
                high is not None and int(value) > high):
            return [f"{where}: {name}={value} (a whole number from {low}"
                    f"{f' to {high}' if high is not None else ' up'} is needed)"]
    if name in NUMBER_RANGES:
        low, high = NUMBER_RANGES[name]
        try:
            number = float(value)
        except ValueError:
            return [f"{where}: {name}={value} is not a number"]
        if not low <= number <= high:
            return [f"{where}: {name}={value} outside {low}..{high}"]
    return []


def region_problems(region, number, sfz, folder, samples):
    """Problems with one region of the .sfz in `folder`."""
    where = f"{folder.name}, region {number}"
    found = []
    for name, value in region.items():
        found += value_problems(name, value, where)
    if "sample" not in region:
        return found + [f"{where}: no sample"]
    path = (folder / sfz.sample_path(region)).resolve()
    if not path.is_relative_to(folder.resolve()) or not path.is_file():
        return found + [f"{where}: sample {sfz.sample_path(region)!r} is not a file of the set"]
    if found:
        return found
    for low, high in (("lokey", "hikey"), ("lovel", "hivel")):
        if low in region and high in region and int(region[low]) > int(region[high]):
            found.append(f"{where}: {low} above {high}")
    frames = samples[path].frames
    if region.get("loop_mode") in ("loop_continuous", "loop_sustain") and not (
            "loop_start" in region and "loop_end" in region):
        # Dawsome reads loop points from the .sfz only, never from a WAV's smpl chunk.
        found.append(f"{where}: loops without loop_start and loop_end")
    if "loop_start" in region or "loop_end" in region:
        start, end = int(region.get("loop_start", 0)), int(region.get("loop_end", frames - 1))
        if not start < end < frames:
            found.append(f"{where}: loop {start}..{end} does not fit {frames} frames")
    for name in ("offset", "end"):
        if name in region and int(region[name]) >= frames:
            found.append(f"{where}: {name}={region[name]} past the sample's {frames} frames")
    if "seq_position" in region and int(region["seq_position"]) > int(
            region.get("seq_length", 1)):
        found.append(f"{where}: seq_position past seq_length")
    return found


def check_set(folder, root):
    """Problems with one set folder."""
    name = folder.name
    found = []
    if not (folder / "set.json").is_file():
        return [f"{name}: no set.json"]
    about = kitlib.read_json(folder / "set.json")
    missing = [field for field in SET_FIELDS if field not in about]
    if missing:
        return [f"{name}/set.json: no {', '.join(missing)}"]
    if about["licence"] != kitlib.LICENCE or not about["licence_quote"]:
        found.append(f"{name}/set.json: not {kitlib.LICENCE} with a license quote")
    sample_folder = folder / "samples"
    sample_paths = sorted(sample_folder.iterdir()) if sample_folder.is_dir() else []
    try:
        samples = {path.resolve(): kitlib.audio_info(path) for path in sample_paths}
    except (Problem, OSError) as error:
        return found + [f"{name}: a sample cannot be read: {error}"]
    shipped = files_under(folder, root) - {(folder / "set.json").relative_to(root).as_posix()}
    size = sum((root / path).stat().st_size for path in shipped)
    expected = {
        "files": len(samples),
        "size": size,
        "decoded_size": sum(info.decoded_size for info in samples.values()),
        "sample_rates": sorted({info.rate for info in samples.values()}),
        "channels": sorted({info.channels for info in samples.values()}),
    }
    found += [f"{name}/set.json: {field} is {about[field]!r}, the files say {value!r}"
              for field, value in expected.items() if about[field] != value]
    sfz_files = sorted(folder.glob("*.sfz"))
    if about["kind"] == "samples":
        return found + ([f"{name}: a samples set with a .sfz"] if sfz_files else [])
    if about["kind"] != "instrument":
        return found + [f"{name}/set.json: kind {about['kind']!r}"]
    if [path.name for path in sfz_files] != [about.get("sfz")]:
        return found + [f"{name}: needs exactly the one .sfz set.json names"]
    sfz = kitlib.read_sfz(sfz_files[0])
    allowed = (kitlib.SFZ_OPCODES | kitlib.SFZ_DIRECTIVES
               | {f"<{header}>" for header in kitlib.SFZ_HEADERS})
    outside = sorted(sfz.used - allowed)
    if outside:
        found.append(f"{name}: outside the SFZ subset: {', '.join(outside)}")
    for number, region in enumerate(sfz.regions, 1):
        found += region_problems(region, number, sfz, folder, samples)
    used = {(folder / sfz.sample_path(region)).resolve() for region in sfz.regions
            if "sample" in region}
    found += [f"{name}: samples/{path.name} is used by no region"
              for path in sorted(set(samples) - used)]
    if not isinstance(about.get("peak_db"), (int, float)) or about["peak_db"] > kitlib.PEAK_DB:
        found.append(f"{name}/set.json: peak_db {about.get('peak_db')!r}; a full-velocity note"
                     f" must peak at {kitlib.PEAK_DB:g} dBFS or lower")
    if about.get("zones") != len(sfz.regions):
        found.append(f"{name}/set.json: zones is {about.get('zones')}, the .sfz has"
                     f" {len(sfz.regions)}")
    keys = [int(region[name]) for region in sfz.regions for name in ("lokey", "hikey")
            if re.fullmatch(r"\d+", region.get(name, ""))]
    if keys and about.get("keys") != [min(keys), max(keys)]:
        found.append(f"{name}/set.json: keys is {about.get('keys')}, the .sfz spans"
                     f" {[min(keys), max(keys)]}")
    return found


def check_sets(root):
    """Problems with every set folder under `root`/sets."""
    sets = root / "sets"
    if not sets.is_dir():
        return ["no sets/ folder"]
    found = [f"sets/{path.name}: not a set folder" for path in sorted(sets.iterdir())
             if not path.is_dir() or path.name.startswith(".")]
    for folder in sorted(path for path in sets.iterdir()
                         if path.is_dir() and not path.name.startswith(".")):
        found += check_set(folder, root)
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", action="store_true")
    parser.add_argument("--sfz", action="store_true")
    args = parser.parse_args()
    both = not (args.manifest or args.sfz)
    try:
        found = []
        if args.manifest or both:
            found += check_manifest(kitlib.ROOT)
        if args.sfz or both:
            found += check_sets(kitlib.ROOT)
    except Problem as error:
        print(f"check: {error}", file=sys.stderr)
        return 2
    if found:
        print("check: the kit does not pass:", *found, sep="\n  ", file=sys.stderr)
        return 1
    print("check: the kit passes" + ("" if both else
                                     f" ({'--manifest' if args.manifest else '--sfz'})"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
