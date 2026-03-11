#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Turn a JSONL file into a quoted-lines text file.

User-requested behavior: take each *raw JSONL line* and convert it to:
    "<raw line>"
Optionally append a trailing comma.

Notes:
- Uses ASCII double quotes (half-width) and escapes backslashes/quotes so the
  output can be pasted into JSON/Python string arrays safely.
- Does NOT parse JSON; it treats each input line as plain text.

Examples:
  python turn.py --input AI_data.jsonl --output quoted_lines.txt
  python turn.py --input AI_data.jsonl --output quoted_lines_with_comma.txt --comma
"""

from __future__ import annotations

import argparse
from pathlib import Path


def escape_for_double_quoted_string(text: str) -> str:
    text = text.replace("\\", "\\\\")
    text = text.replace('"', '\\"')
    text = text.replace("\r\n", "\\n").replace("\n", "\\n").replace("\r", "\\n")
    return text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input JSONL file path")
    parser.add_argument("--output", required=True, help="Output text file path")
    parser.add_argument(
        "--comma",
        action="store_true",
        help="Append a trailing comma after the closing quote.",
    )
    parser.add_argument(
        "--no-escape",
        action="store_true",
        help=(
            "Do not escape backslashes/quotes; only wrap with double quotes. "
            "(Not recommended if you need the output to be machine-parsable.)"
        ),
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise SystemExit(f"Input not found: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    suffix = "," if args.comma else ""

    with input_path.open("r", encoding="utf-8") as fin, output_path.open(
        "w", encoding="utf-8", newline="\n"
    ) as fout:
        for raw in fin:
            line = raw.rstrip("\n")
            if line.strip() == "":
                continue
            content = line if args.no_escape else escape_for_double_quoted_string(line)
            fout.write(f'"{content}"{suffix}\n')


if __name__ == "__main__":
    main()
