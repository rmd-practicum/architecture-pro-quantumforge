import json
import re
import sys
from collections import Counter
from pathlib import Path


def build_pattern(keys):
    ordered = sorted(keys, key=len, reverse=True)
    alternation = "|".join(re.escape(key) for key in ordered)
    return re.compile(rf"(?<![A-Za-z0-9])({alternation})(s?)(?![A-Za-z0-9])")


def slugify(title):
    name = re.sub(r"\s*\([^)]*\)", "", title)
    name = name.replace("'", "").replace("’", "")
    name = re.sub(r"[^A-Za-z0-9]+", " ", name)
    return "".join(word[0].upper() + word[1:] for word in name.split())


def main():
    in_path = Path("../0_raw")
    out_path = Path("out")
    map_file = Path("replacements.json")

    mapping = json.loads(map_file.read_text(encoding="utf-8"))
    sources = sorted(in_path.glob("*.txt"))
    if not sources:
        sys.exit(f"no .txt files in {in_path}")

    hits = Counter()

    def substitute(match):
        hits[match.group(1)] += 1
        return mapping[match.group(1)] + match.group(2)

    out_path.mkdir(parents=True, exist_ok=True)
    stale = list(out_path.glob("*.txt"))
    for path in stale:
        path.unlink()

    pattern = build_pattern(mapping)

    written = {}
    for source in sources:
        text = pattern.sub(substitute, source.read_text(encoding="utf-8"))
        title = text.split("\n", 1)[0].lstrip("# ").strip()
        name = f"{slugify(title)}.txt"
        if name in written:
            sys.exit(f"name collision: {source.name} and {written[name]} -> {name}")
        written[name] = source.name
        (out_path / name).write_text(text, encoding="utf-8")

    print(f"cleared {len(stale)} stale file(s).")
    print(f"processed {len(sources)} files -> {out_path}")
    print(f"applied {sum(hits.values())} replacements over {len(hits)} entities.")
    print(
        f"renamed {sum(1 for k, v in written.items() if k != v)} of {len(written)} files."
    )

    unused = sorted(set(mapping) - set(hits))
    if unused:
        print(f"unused keys ({len(unused)}): {', '.join(unused)}")


if __name__ == "__main__":
    main()
