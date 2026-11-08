"""Conservative checks for short cells and reusable skills, not an OS sandbox."""

from __future__ import annotations

import ast

PURE_IMPORTS = {"math", "numpy", "statistics", "collections", "typing", "__future__"}
FORBIDDEN = {
    "open",
    "exec",
    "eval",
    "compile",
    "getattr",
    "setattr",
    "delattr",
    "globals",
    "locals",
    "vars",
    "input",
    "breakpoint",
    "load",
    "save",
    "loadtxt",
    "savetxt",
    "fromfile",
    "tofile",
    "memmap",
    "ctypeslib",
    "f2py",
    "read",
    "write",
    "dump",
    "dumps",
    "load_library",
    "eval_seed",
    "layout_id",
    "seed",
    "__import__",
}


def check_source(source: str, *, skill: bool = False) -> list[str]:
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return [str(exc)]
    errors = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and (
            node.attr.startswith("_") or node.attr in FORBIDDEN
        ):
            errors.append(f"forbidden attribute: {node.attr}")
        if isinstance(node, ast.Name) and node.id in FORBIDDEN:
            errors.append(f"forbidden name: {node.id}")
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = (
                [alias.name for alias in node.names]
                if isinstance(node, ast.Import)
                else [node.module or ""]
            )
            if any(name.split(".")[0] not in PURE_IMPORTS | {"library"} for name in names):
                errors.append("imports must be pure computation or library modules")
            if isinstance(node, ast.ImportFrom) and node.level:
                errors.append("use explicit library imports")
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.Delete)):
            targets = getattr(node, "targets", [getattr(node, "target", None)])
            if any(
                isinstance(item, ast.Attribute)
                for target in targets
                if target
                for item in ast.walk(target)
            ):
                errors.append("attribute assignment is not allowed")
            if any(
                isinstance(item, ast.Name) and item.id in {"robo", "harness"}
                for target in targets
                if target
                for item in ast.walk(target)
            ):
                errors.append("robo and harness bindings are read-only")
        if skill and isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.decorator_list or any(
                isinstance(item, ast.Call)
                for value in [*node.args.defaults, *node.args.kw_defaults]
                if value
                for item in ast.walk(value)
            ):
                errors.append("decorators and calls in defaults run at import")
            if any(
                isinstance(item, ast.Call)
                and isinstance(item.func, ast.Name)
                and item.func.id == node.name
                for item in ast.walk(node)
            ):
                errors.append("recursive skills are not allowed")
            annotations = [node.returns, *[argument.annotation for argument in node.args.args]]
            if any(
                isinstance(item, ast.Call)
                for value in annotations
                if value
                for item in ast.walk(value)
            ):
                errors.append("calls in annotations run at import")
        if skill and isinstance(node, ast.AsyncFunctionDef):
            errors.append("skills must be synchronous functions")
        if skill and isinstance(node, ast.ClassDef):
            errors.append("skills must be functions, not classes")
    return sorted(set(errors))
