#!/usr/bin/env bash
# Build an immutable RoboDojo source root judged by the instruction-faithful criteria, for the
# "source_root" of a learn campaign or an evaluation:
#   criteria_v2/build_source.sh <tag>          ->  criteria_v2/build/<tag>/
#   OUT=<dir> criteria_v2/build_source.sh <tag> ->  <dir> (a scratch build for tests)
# The base is the RoboDojo checkout at cb2a37d ($ROBODOJO_SOURCE_ROOT, see tools/env.sh).
# env/ and task/ are copied from it and overlay/ is laid over them; the other parts the
# evaluation client reads are links to the checkout.  A built tag is never rebuilt: runs pin
# the sha256 of the benchmark they were judged by.  The paper's build is tag v2-b9.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
base="${ROBODOJO_SOURCE_ROOT:?set ROBODOJO_SOURCE_ROOT (source tools/env.sh)}"
tag="${1:?usage: build_source.sh <tag>}"
out="${OUT:-$here/build/$tag}"
if [ -e "$out" ]; then echo "exists: $out (tags are immutable; use a new tag)" >&2; exit 1; fi
mkdir -p "$out"
for part in env task; do
  cp -a "$base/$part" "$out/$part"
done
find "$out" -name __pycache__ -prune -exec rm -rf {} +
for part in src utils scripts env_cfg XPolicyLab Assets third_party; do
  ln -s "$(readlink -f "$base/$part")" "$out/$part"
done
(cd "$here/overlay" && find . -type f ! -name '*.pyc' ! -path '*/__pycache__/*' | sort) | while read -r f; do
  f="${f#./}"
  mkdir -p "$out/$(dirname "$f")"
  cp "$here/overlay/$f" "$out/$f"
done
# the cb2a37d file of every overlaid task, for the shadow judge (RewardManager.attach_v1_shadow)
mkdir -p "$out/task/RoboDojo/tasks/criteria_v1"
for f in "$here"/overlay/task/RoboDojo/tasks/*.py; do
  cp "$base/task/RoboDojo/tasks/$(basename "$f")" "$out/task/RoboDojo/tasks/criteria_v1/"
done
{
  echo "tag: $tag"
  echo "built: $(date -Is)"
  echo "base: $(readlink -f "$base") (RoboDojo cb2a37d)"
  echo "overlay files (sha256, relative to the root):"
  (cd "$out" && cd "$here/overlay" && find . -type f ! -name '*.pyc' ! -path '*/__pycache__/*' | sort | sed 's#^\./##') | while read -r f; do
    echo "  $(sha256sum "$out/$f" | cut -c1-64)  $f$( [ -e "$base/$f" ] || echo '  (new)')"
  done
} > "$out/MANIFEST.txt"
(cd "$here/overlay" && find . -type f -name '*.py' | sort | sed 's#^\./##') | while read -r f; do
  if [ -e "$base/$f" ]; then diff -u "$base/$f" "$out/$f" --label "cb2a37d/$f" --label "$tag/$f" || true; fi
done > "$out/overlay.diff"
chmod -R a-w "$out/env" "$out/task" "$out/MANIFEST.txt" "$out/overlay.diff"
echo "built $out"
cat "$out/MANIFEST.txt"
