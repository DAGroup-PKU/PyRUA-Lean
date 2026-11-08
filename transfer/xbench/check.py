"""Static acceptance checks of a cross-benchmark library (DIR/out, or any directory with lessons/ and
skills/ given with --lib) against the LIBERO-PRO libero_rgb API or, with --target robocasa, the
RoboCasa365 robocasa_rgb API (both from this repository's src/):

    python -I transfer/xbench/check.py [--target T] DIR [DIR ...]          # checks DIR/out, target prompt DIR/inputs/target/
    python -I transfer/xbench/check.py [--target T] --lib LIB --prompt P   # checks any library (e.g. a frozen library-v1)

- the BRIEF's limits and card fields;
- API: every `robo.<name>` in skills (AST) and in lessons (text) exists on LiberoColourRobot, and
  every call binds to its signature; `state().<field>` names a field of libero_rgb.State; camera
  names are the target's; no RoboDojo-only words (home(, path=, arm=, cam_*) in a card's do:;
- names: no RoboDojo task name anywhere; no LIBERO task object in do:/applies:/skills (a term that
  the target prompt itself uses is only a warning);
- every library.<module>[.<name>] a card names is a skill module that imports and defines it.
robocasa_rgb keeps RoboDojo's argument names (arm=, path=, home()), so those are allowed there; what
is not: an arm other than "right", RoboDojo or LIBERO camera names, the absent members (done,
world_map), and in do:/applies:/skills a RoboCasa Target50 task name (kitchen words only warn).
Exit status 1 when any PROBLEM is found.
"""

import argparse
import ast
import dataclasses
import importlib.util
import inspect
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.dont_write_bytecode = True  # importing a library's skills leaves no __pycache__ in it

ap = argparse.ArgumentParser()
ap.add_argument("dirs", nargs="*", type=Path)
ap.add_argument("--lib", type=Path)
ap.add_argument("--prompt", type=Path)
ap.add_argument("--target", choices=("libero", "robocasa"), default="libero")
a = ap.parse_args()
if a.target == "libero":
    from pyrualean.libero_rgb import LiberoColourRobot as Real  # noqa: E402
    from pyrualean.libero_rgb import State

    ROBOT, BENCH = "libero_rgb", "LIBERO"
else:
    from pyrualean.robocasa_rgb import RobocasaColourRobot as Real  # noqa: E402
    from pyrualean.robocasa_rgb import State

    ROBOT, BENCH = "robocasa_rgb", "RoboCasa"
ABSENT = set(getattr(Real, "ABSENT_MEMBERS", ()))

FIELDS = ("stage", "symptom", "applies", "do", "evidence")
STATE_FIELDS = {f.name for f in dataclasses.fields(State)}
CAMERAS = set(Real.CAMERAS)
# RoboDojo's tasks (task/RoboDojo/tasks/*.py at commit cb2a37d, without the _random suffix)
ROBODOJO_TASKS = [
    "align_blocks",
    "arrange_largest_number",
    "build_tower",
    "classify_objects",
    "classify_objects_by_language",
    "cover_blocks",
    "deposit_coin",
    "fasten_screws",
    "fill_egg_holder",
    "fill_pen_holder",
    "fold_clothes",
    "general_pickup",
    "hang_mugs",
    "imitate_sorting_sequence",
    "insert_key",
    "insert_tubes",
    "make_kong",
    "make_toast",
    "match_and_pick_from_conveyor",
    "organize_table",
    "pack_objects_into_box",
    "pick_from_conveyor_by_image",
    "play_Xylophone",
    "play_stacking_toy",
    "play_tic_tac_toe",
    "plug_in_charger",
    "pour_balls_into_vase",
    "pour_by_language",
    "pour_liquid_into_cup",
    "press_by_number",
    "push_T",
    "put_bottles_into_dustbin",
    "solve_equation",
    "sort_nesting_dolls_by_size",
    "stack_blocks",
    "stack_blocks_by_language",
    "stack_bowls",
    "store_laptop_and_headphones",
    "store_tools_in_toolbox",
    "swap_T",
    "swap_blocks",
    "sweep_blocks",
]
# LIBERO-10 objects and task words (from the libero_10_swap task cards); the evaluated suite's names
LIBERO_HARD = [
    "alphabet_soup",
    "tomato_sauce",
    "cream_cheese",
    "orange_juice",
    "chefmate",
    "frypan",
    "moka",
    "akita",
    "wine_bottle",
    "porcelain_mug",
    "red_coffee_mug",
    "white_yellow_mug",
    "black_book",
    "chocolate_pudding",
    "alphabet soup",
    "tomato sauce",
    "cream cheese",
    "chocolate pudding",
    "caddy",
    "ketchup",
    "butter",
    "yellow and white mug",
    "white mug",
    "black bowl",
]
LIBERO_SOFT = [
    "basket",
    "microwave",
    "stove",
    "drawer",
    "cabinet",
    "plate",
    "mug",
    "book",
    "bowl",
    "milk",
]
SOURCE_ONLY = [
    (r"\bhome\s*\(", "home()"),
    (r"\bpath\s*=", "path="),
    (r"\barm\s*=", "arm="),
    (r"\bcam_[a-z_]+", "a RoboDojo camera"),
    (r"\.arm\s*\(", "state().arm()"),
]
OTHER_CAMERAS = {"cam_head", "head", "left_wrist", "right_wrist"}
if a.target == "robocasa":
    # the Target50 tasks (RoboCasa365's 18 atomic + 32 composite target tasks); kitchen words only warn
    LIBERO_HARD = [
        "ArrangeBreadBasket",
        "ArrangeTea",
        "BreadSelection",
        "CategorizeCondiments",
        "CloseBlenderLid",
        "CloseFridge",
        "CloseToasterOvenDoor",
        "CoffeeSetupMug",
        "CuttingToolSelection",
        "DeliverStraw",
        "GarnishPancake",
        "GatherTableware",
        "GetToastedBread",
        "HeatKebabSandwich",
        "KettleBoiling",
        "LoadDishwasher",
        "MakeIceLemonade",
        "NavigateKitchen",
        "OpenCabinet",
        "OpenDrawer",
        "OpenStandMixerHead",
        "PackIdenticalLunches",
        "PanTransfer",
        "PickPlaceCounterToCabinet",
        "PickPlaceCounterToStove",
        "PickPlaceDrawerToCounter",
        "PickPlaceSinkToCounter",
        "PickPlaceToasterToCounter",
        "PortionHotDogs",
        "PreSoakPan",
        "PrepareCoffee",
        "RecycleBottlesByType",
        "RinseSinkBasin",
        "ScrubCuttingBoard",
        "SearingMeat",
        "SeparateFreezerRack",
        "SetUpCuttingStation",
        "SlideDishwasherRack",
        "StackBowlsCabinet",
        "SteamInMicrowave",
        "StirVegetables",
        "StoreLeftoversInBowl",
        "TurnOffStove",
        "TurnOnElectricKettle",
        "TurnOnMicrowave",
        "TurnOnSinkFaucet",
        "WaffleReheat",
        "WashFruitColander",
        "WashLettuce",
        "WeighIngredients",
    ]
    LIBERO_SOFT = [
        "drawer",
        "cabinet",
        "fridge",
        "microwave",
        "sink",
        "faucet",
        "stove",
        "burner",
        "oven",
        "toaster",
        "dishwasher",
        "kettle",
        "blender",
        "mixer",
        "counter",
        "mug",
        "bowl",
        "plate",
        "pan",
    ]
    SOURCE_ONLY = [
        (r"\bcam_[a-z_]+", "a RoboDojo camera"),
        (r"""arm\s*=\s*["'](?!right)""", "an arm other than right"),
        (r"""\.arm\s*\(\s*["'](?!right)""", "an arm other than right"),
        (r"\bstate\(\)\.left\b", "state().left"),
    ]
    OTHER_CAMERAS |= {"agentview", "wrist"}


def cards_of(lessons):
    out = []
    for p in lessons:
        for block in re.split(r"(?m)^(?=## )", p.read_text())[1:]:
            cid = block.splitlines()[0][3:].strip()
            f = {}
            for line in block.splitlines()[1:]:
                k, sep, v = line.partition(":")
                if sep and not k.startswith(" ") and k.strip().lower() in FIELDS:
                    f[k.strip().lower()] = v.strip()
            out.append((cid, f, block))
    return out


def bind(name, args, kwargs):
    """None when robo.<name>(...) with these argument shapes binds to the real method, else why not."""
    if not hasattr(Real, name) or name in ABSENT:
        return f"robo.{name} does not exist on {ROBOT}"
    if name in Real.VLA_PRIMITIVES:
        return f"robo.{name} is a VLA primitive, absent in the no-vla evaluation"
    member = getattr(Real, name)
    if not callable(member):
        return f"robo.{name} is not callable"
    try:
        inspect.signature(member).bind(None, *args, **kwargs)
    except TypeError as e:
        return f"robo.{name}(...): {e}"
    return None


def calls_in(tree):
    """(name, n_positional, keyword names, starred) of every robo.<name>(...) call, and plain robo.<name>."""
    calls, attrs = [], []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "robo"
        ):
            starred = any(isinstance(a, ast.Starred) for a in node.args) or any(
                k.arg is None for k in node.keywords
            )
            calls.append(
                (
                    node.func.attr,
                    len(node.args),
                    [k.arg for k in node.keywords if k.arg],
                    starred,
                    node.lineno,
                )
            )
        elif (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "robo"
        ):
            attrs.append((node.attr, node.lineno))
    return calls, attrs


def snippets(text):
    """`robo.name(...)` expressions in prose, cut at the matching parenthesis."""
    for m in re.finditer(r"robo\.(\w+)\s*\(", text):
        depth, i = 0, m.end() - 1
        for j in range(i, min(len(text), i + 400)):
            depth += {"(": 1, ")": -1}.get(text[j], 0)
            if depth == 0:
                yield m.group(1), text[m.start() : j + 1]
                break


def check(lib, prompt_text, label):
    probs, warns = [], []
    lessons = sorted((lib / "lessons").glob("*.md")) if (lib / "lessons").is_dir() else []
    skills = sorted((lib / "skills").glob("*.py")) if (lib / "skills").is_dir() else []
    lc = sum(len(p.read_text()) for p in lessons)
    sc = sum(len(p.read_text()) for p in skills)
    if lc > 12000:
        probs.append(f"lessons {lc} > 12000 chars")
    if len(skills) > 2 or sc > 15000:
        probs.append(f"skills {len(skills)} files / {sc} chars")
    if not lessons and not skills:
        probs.append("no lessons or skills")
    prompt_low = prompt_text.lower()

    def names(text, where, sources_too=True):
        # evidence: is meant to name the source tasks; do:/applies:/skills name no task of either side
        low = text.lower()
        for t in ROBODOJO_TASKS if sources_too else []:
            if re.search(rf"\b{re.escape(t.lower())}\b", low):
                probs.append(f"{where}: RoboDojo task name {t}")
        for t in LIBERO_HARD:
            if re.search(rf"\b{re.escape(t)}\b", low):
                (warns if t in prompt_low else probs).append(
                    f"{where}: {BENCH} object {t!r}"
                    + (" (the target prompt uses it too)" if t in prompt_low else "")
                )
        for t in LIBERO_SOFT if sources_too else []:
            if re.search(rf"\b{re.escape(t)}s?\b", low):
                warns.append(f"{where}: generic {BENCH} object word {t!r}")

    cards = cards_of(lessons)
    for cid, f, block in cards:
        miss = [k for k in FIELDS if k not in f]
        if miss:
            probs.append(f"{cid}: missing {miss}")
        names(f.get("do", "") + " " + f.get("applies", ""), f"{cid} do/applies")
        names(
            f.get("symptom", "") + " " + f.get("evidence", ""),
            f"{cid} symptom/evidence",
            sources_too=False,
        )
        for pat, what in SOURCE_ONLY:
            if re.search(pat, f.get("do", "")):
                probs.append(f"{cid} do: {what} (RoboDojo API, not {ROBOT})")
        for cam in set(re.findall(r"""["'`](\w+)["'`]""", block)) & OTHER_CAMERAS:
            probs.append(f"{cid}: camera {cam!r} is not a {ROBOT} camera {sorted(CAMERAS)}")
        for name, snip in snippets(block):
            expr = snip.replace("...", "None").replace("…", "None")
            try:
                tree = ast.parse(expr, mode="eval")
            except SyntaxError:
                if not hasattr(Real, name) or name in ABSENT:
                    probs.append(f"{cid}: robo.{name} does not exist on {ROBOT}")
                else:
                    warns.append(f"{cid}: could not parse {snip[:80]!r} to check its arguments")
                continue
            for cname, npos, kws, starred, _ in calls_in(tree)[0]:
                why = None if starred else bind(cname, [None] * npos, dict.fromkeys(kws))
                if why:
                    probs.append(f"{cid}: {why}  [{snip[:90]}]")
        for name in set(re.findall(r"robo\.(\w+)", block)):
            if not hasattr(Real, name) or name in ABSENT:
                probs.append(f"{cid}: robo.{name} does not exist on {ROBOT}")
        for field in set(re.findall(r"state\([^)]*\)\.(\w+)", block)):
            if field not in STATE_FIELDS and not (a.target == "robocasa" and field == "arm"):
                probs.append(f"{cid}: state().{field} is not a field of {ROBOT}.State")

    modules = {}
    for p in skills:
        src = p.read_text()
        try:
            tree = ast.parse(src)
        except SyntaxError as e:
            probs.append(f"{p.name}: {e}")
            continue
        names(src, p.name)
        calls, attrs = calls_in(tree)
        for cname, npos, kws, starred, line in calls:
            why = None if starred else bind(cname, [None] * npos, dict.fromkeys(kws))
            if why:
                probs.append(f"{p.name}:{line}: {why}")
        for aname, line in attrs:
            if not hasattr(Real, aname) or aname in ABSENT:
                probs.append(f"{p.name}:{line}: robo.{aname} does not exist on {ROBOT}")
        for field, line in {
            (n.attr, n.lineno)
            for n in ast.walk(tree)
            if isinstance(n, ast.Attribute)
            and isinstance(n.value, ast.Call)
            and isinstance(n.value.func, ast.Attribute)
            and n.value.func.attr == "state"
        }:
            if field not in STATE_FIELDS and not (a.target == "robocasa" and field == "arm"):
                probs.append(f"{p.name}:{line}: state().{field} is not a field of {ROBOT}.State")
        for lit in {
            n.value
            for n in ast.walk(tree)
            if isinstance(n, ast.Constant) and isinstance(n.value, str)
        }:
            if lit.startswith("cam_") or lit in OTHER_CAMERAS:
                probs.append(f"{p.name}: camera {lit!r} is not a {ROBOT} camera")
        if a.target == "robocasa":
            # the one arm: a literal arm in a motion call must be "right"
            for n in ast.walk(tree):
                if (
                    isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Attribute)
                    and isinstance(n.func.value, ast.Name)
                    and n.func.value.id == "robo"
                    and n.func.attr in ("move_to", "set_gripper", "home")
                ):
                    pos = {"move_to": 1, "set_gripper": 0, "home": 0}[n.func.attr]
                    arg = next(
                        (k.value for k in n.keywords if k.arg == "arm"),
                        n.args[pos] if len(n.args) > pos else None,
                    )
                    if isinstance(arg, ast.Constant) and arg.value != "right":
                        probs.append(
                            f"{p.name}:{n.lineno}: robo.{n.func.attr} with arm {arg.value!r} (the robot has one arm, 'right')"
                        )
                if isinstance(n, ast.Constant) and n.value in ("left", "both"):
                    warns.append(
                        f"{p.name}:{getattr(n, 'lineno', '?')}: string {n.value!r} (an arm name on RoboDojo?)"
                    )
        try:
            spec = importlib.util.spec_from_file_location(f"library.{p.stem}", p)
            m = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = m
            spec.loader.exec_module(m)
            fns = [
                n
                for n, v in vars(m).items()
                if callable(v)
                and not n.startswith("_")
                and getattr(v, "__module__", "") == spec.name
            ]
            nodoc = [n for n in fns if not (getattr(m, n).__doc__ or "").strip()]
            if nodoc:
                probs.append(f"{p.name}: public functions without a docstring {nodoc}")
            modules[p.stem] = m
        except Exception as e:
            probs.append(f"{p.name}: import failed {e!r}")
            fns = []
        print(f"   {p.name}: {len(src)} chars, functions {fns}")
    for cid, _fields, block in cards:
        for mod, attr in set(re.findall(r"\blibrary\.(\w+)(?:\.(\w+))?", block)):
            if mod not in modules:
                probs.append(
                    f"{cid}: names library.{mod}, which is not a skill module that imports"
                )
            elif attr and not hasattr(modules[mod], attr):
                probs.append(f"{cid}: names library.{mod}.{attr}, which {mod}.py does not define")
    print(
        f"{label}: {len(cards)} cards, lessons {lc} chars, skills {len(skills)} files {sc} chars;",
        "OK" if not probs else f"{len(probs)} PROBLEMS:",
    )
    for x in probs:
        print("   - PROBLEM", x)
    for x in sorted(set(warns)):
        print("   - warning", x)
    return not probs


ok = True
for d in a.dirs:
    ok &= check(d / "out", (d / "inputs/target/system_prompt.md").read_text(), d.name)
if a.lib:
    ok &= check(a.lib, a.prompt.read_text() if a.prompt else "", str(a.lib))
sys.exit(0 if ok else 1)
