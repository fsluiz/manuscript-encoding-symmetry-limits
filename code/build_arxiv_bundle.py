#!/usr/bin/env python3
"""Build and verify a minimal, deterministic arXiv source bundle.

The bundle is intentionally distinct from the reproducibility archive.  It
contains only the TeX dependency closure of ``main_rewrite.tex`` together with
the generated bibliography and the exact journal class/style required to
compile it.  Code, data, local notes, and the compiled PDF are excluded.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import os
from pathlib import Path
import re
import shutil
import subprocess
import tarfile
import tempfile


ROOT = Path(__file__).resolve().parents[1]
MAIN = Path("main_rewrite.tex")
DEFAULT_OUTPUT = Path("dist/arxiv_source_candidate.tar.gz")
DEFAULT_EPOCH = 1_786_449_600  # 2026-08-11 12:00:00 UTC

INPUT_RE = re.compile(r"\\(?:input|include)\{([^}]+)\}")
GRAPHICS_RE = re.compile(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}")
BIBLIOGRAPHY_RE = re.compile(r"\\bibliography\{([^}]+)\}")
BIBSTYLE_RE = re.compile(r"\\bibliographystyle\{([^}]+)\}")
DOCUMENTCLASS_RE = re.compile(r"\\documentclass(?:\[[^]]*\])?\{([^}]+)\}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def strip_comments(text: str) -> str:
    """Remove TeX comments while preserving escaped percent signs."""

    cleaned: list[str] = []
    for line in text.splitlines():
        cut = len(line)
        for index, char in enumerate(line):
            if char != "%":
                continue
            backslashes = 0
            cursor = index - 1
            while cursor >= 0 and line[cursor] == "\\":
                backslashes += 1
                cursor -= 1
            if backslashes % 2 == 0:
                cut = index
                break
        cleaned.append(line[:cut])
    return "\n".join(cleaned)


def safe_relative(candidate: Path) -> Path:
    absolute = (ROOT / candidate).resolve()
    try:
        relative = absolute.relative_to(ROOT.resolve())
    except ValueError as exc:
        raise ValueError(f"dependency escapes repository root: {candidate}") from exc
    if not absolute.is_file():
        raise FileNotFoundError(f"missing manuscript dependency: {relative}")
    if (ROOT / candidate).is_symlink():
        raise ValueError(f"arXiv dependency must not be a symlink: {relative}")
    return relative


def tex_path(name: str, parent: Path) -> Path:
    raw = Path(name.strip())
    candidates = [raw, parent / raw]
    if not raw.suffix:
        candidates = [raw.with_suffix(".tex"), (parent / raw).with_suffix(".tex")]
    for candidate in candidates:
        if (ROOT / candidate).is_file():
            return safe_relative(candidate)
    raise FileNotFoundError(f"cannot resolve TeX input {name!r} from {parent}")


def graphic_path(name: str, parent: Path) -> Path:
    raw = Path(name.strip())
    candidates = [raw, parent / raw]
    if not raw.suffix:
        candidates = [
            candidate.with_suffix(extension)
            for candidate in candidates
            for extension in (".pdf", ".png", ".jpg", ".jpeg", ".eps")
        ]
    for candidate in candidates:
        if (ROOT / candidate).is_file():
            return safe_relative(candidate)
    raise FileNotFoundError(f"cannot resolve graphic {name!r} from {parent}")


def dependency_closure() -> list[Path]:
    main = safe_relative(MAIN)
    files: set[Path] = {main}
    queue = [main]

    while queue:
        current = queue.pop()
        text = strip_comments((ROOT / current).read_text(encoding="utf-8"))

        for name in INPUT_RE.findall(text):
            dependency = tex_path(name, current.parent)
            if dependency not in files:
                files.add(dependency)
                queue.append(dependency)

        for name in GRAPHICS_RE.findall(text):
            files.add(graphic_path(name, current.parent))

        for group in BIBLIOGRAPHY_RE.findall(text):
            for name in group.split(","):
                files.add(safe_relative(Path(name.strip()).with_suffix(".bib")))

        for name in BIBSTYLE_RE.findall(text):
            files.add(safe_relative(Path(name.strip()).with_suffix(".bst")))

        for name in DOCUMENTCLASS_RE.findall(text):
            local_class = Path(name.strip()).with_suffix(".cls")
            if (ROOT / local_class).is_file():
                files.add(safe_relative(local_class))

    # arXiv should not depend on running BibTeX.  The generated bibliography
    # is included in addition to the .bib and .bst provenance files.
    files.add(safe_relative(Path("main_rewrite.bbl")))
    return sorted(files, key=lambda path: path.as_posix())


def build_archive(output: Path, epoch: int) -> tuple[list[Path], str]:
    files = dependency_closure()
    output = output if output.is_absolute() else ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=epoch) as zipped:
            with tarfile.open(fileobj=zipped, mode="w", format=tarfile.USTAR_FORMAT) as archive:
                for relative in files:
                    payload = (ROOT / relative).read_bytes()
                    info = tarfile.TarInfo(relative.as_posix())
                    info.size = len(payload)
                    info.mtime = epoch
                    info.uid = 0
                    info.gid = 0
                    info.uname = "root"
                    info.gname = "root"
                    info.mode = 0o644
                    archive.addfile(info, fileobj=io.BytesIO(payload))

    return files, sha256(output)


def verify_archive(output: Path, expected_pdf: Path, epoch: int) -> str:
    output = output if output.is_absolute() else ROOT / output
    expected_pdf = expected_pdf if expected_pdf.is_absolute() else ROOT / expected_pdf
    if not expected_pdf.is_file():
        raise FileNotFoundError(f"missing reference PDF: {expected_pdf}")

    with tempfile.TemporaryDirectory(prefix="set-cover-arxiv-") as temporary:
        work = Path(temporary)
        with tarfile.open(output, "r:gz") as archive:
            archive.extractall(work)

        env = os.environ.copy()
        env.update(
            {
                "SOURCE_DATE_EPOCH": str(epoch),
                "FORCE_SOURCE_DATE": env.get("FORCE_SOURCE_DATE", "1"),
                "TZ": "UTC",
            }
        )
        command = ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", MAIN.name]
        for pass_number in range(1, 4):
            result = subprocess.run(
                command,
                cwd=work,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )
            (work / f"verification-pass-{pass_number}.log").write_text(
                result.stdout, encoding="utf-8"
            )
            if result.returncode:
                raise RuntimeError(
                    f"clean arXiv compilation failed on pass {pass_number}:\n"
                    f"{result.stdout[-4000:]}"
                )

        log = (work / "main_rewrite.log").read_text(encoding="utf-8", errors="replace")
        forbidden = (
            "There were undefined citations",
            "There were undefined references",
            "Citation(s) may have changed",
            "Label(s) may have changed",
            "Rerun to get cross-references right",
            "Fatal error",
            "Overfull \\hbox",
            "Overfull \\vbox",
        )
        present = [marker for marker in forbidden if marker in log]
        if present:
            raise RuntimeError(f"final TeX log contains forbidden diagnostics: {present}")

        rebuilt = work / "main_rewrite.pdf"
        rebuilt_hash = sha256(rebuilt)
        expected_hash = sha256(expected_pdf)
        if rebuilt_hash != expected_hash:
            raise RuntimeError(
                "clean arXiv PDF differs from the release PDF: "
                f"rebuilt={rebuilt_hash}, expected={expected_hash}"
            )

        included_bbl_hash = sha256(work / "main_rewrite.bbl")
        bibtex = subprocess.run(
            ["bibtex", MAIN.stem],
            cwd=work,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        if bibtex.returncode:
            raise RuntimeError(f"BibTeX provenance check failed:\n{bibtex.stdout[-4000:]}")
        regenerated_bbl_hash = sha256(work / "main_rewrite.bbl")
        if regenerated_bbl_hash != included_bbl_hash:
            raise RuntimeError(
                "included main_rewrite.bbl is stale relative to refs.bib/quantum.bst: "
                f"included={included_bbl_hash}, regenerated={regenerated_bbl_hash}"
            )
        return rebuilt_hash


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"archive path relative to the repository (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=int(os.environ.get("SOURCE_DATE_EPOCH", DEFAULT_EPOCH)),
        help="timestamp used for every tar member and the gzip header",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="compile the archive in a clean temporary directory and compare its PDF",
    )
    parser.add_argument(
        "--reference-pdf",
        type=Path,
        default=Path("main_rewrite.pdf"),
        help="release PDF used by --verify",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    files, archive_hash = build_archive(args.output, args.epoch)
    output = args.output if args.output.is_absolute() else ROOT / args.output
    print(f"Wrote {output.relative_to(ROOT) if output.is_relative_to(ROOT) else output}")
    print(f"Files: {len(files)}")
    for path in files:
        print(f"  {path.as_posix()}")
    print(f"Archive SHA-256: {archive_hash}")
    if args.verify:
        pdf_hash = verify_archive(args.output, args.reference_pdf, args.epoch)
        print(f"Clean compilation PDF SHA-256: {pdf_hash}")
        print("arXiv bundle verification passed")


if __name__ == "__main__":
    main()
