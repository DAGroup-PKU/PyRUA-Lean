#!/bin/sh
# Freeze DIR/out (a finished session of run_worker.sh) as read-only DIR/library-v1 (lessons/*.md,
# skills/*.py; no __pycache__) with DIR/library-v1.sha256, and copy library-v1, BRIEF.md,
# out/NOTES.md and whichever of SOURCES.json, MANIFEST.json DIR has to DEST (a new directory).
#     sh transfer/freeze.sh DIR DEST
# DEST/library-v1 is what an evaluation is given as its library; DEST is also what xbench/prep.py
# takes as a source library.
set -eu
[ $# -eq 2 ] || { echo "usage: sh freeze.sh DIR DEST" >&2; exit 2; }
d=$(cd "$1" && pwd)
t=$2
[ -f "$d/out/NOTES.md" ] || { echo "$d/out/NOTES.md missing" >&2; exit 1; }
[ ! -e "$d/library-v1" ] && [ ! -e "$t" ] || { echo "$d/library-v1 or $t exists" >&2; exit 1; }
mkdir -p "$d/library-v1"
for part in lessons skills; do
    if [ -d "$d/out/$part" ]; then
        mkdir "$d/library-v1/$part"
        find "$d/out/$part" -maxdepth 1 -type f \( -name '*.md' -o -name '*.py' \) -exec cp {} "$d/library-v1/$part/" \;
    fi
done
(cd "$d/library-v1" && find . -type f | sort | xargs sha256sum) > "$d/library-v1.sha256"
chmod -R a-w "$d/library-v1"
mkdir -p "$t"
cp -r "$d/library-v1" "$t/library-v1"
chmod -R u+w "$t/library-v1"
cp "$d/BRIEF.md" "$t/"
for name in SOURCES.json MANIFEST.json; do
    if [ -f "$d/$name" ]; then cp "$d/$name" "$t/"; fi
done
cp "$d/out/NOTES.md" "$t/NOTES.md"
(cd "$t/library-v1" && find . -type f | sort | xargs sha256sum) | cmp - "$d/library-v1.sha256"
echo "$t $(cut -c1-8 "$d/library-v1.sha256" | tr '\n' ' ')"
