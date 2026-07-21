from __future__ import annotations

import ast
import os
import py_compile
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SECURITY_WRAPPERS = (
    "cf-attention-probe",
    "cf-cuda-smoke",
    "cf-runtime-capture",
    "cf-runtime-evidence-capture",
    "cf-runtime-preflight",
    "cf-streaming-acceptance",
    "cf-streaming-browser",
)


class SecurityWrapperTests(unittest.TestCase):
    def test_bytecode_lookup_is_redirected_before_every_project_import(self) -> None:
        for name in SECURITY_WRAPPERS:
            with self.subTest(name=name):
                path = PROJECT_ROOT / "scripts" / name
                tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
                pycache_lines = [
                    node.lineno
                    for node in ast.walk(tree)
                    if isinstance(node, ast.Assign)
                    and any(
                        isinstance(target, ast.Attribute)
                        and isinstance(target.value, ast.Name)
                        and target.value.id == "sys"
                        and target.attr == "pycache_prefix"
                        for target in node.targets
                    )
                ]
                dont_write_lines = [
                    node.lineno
                    for node in ast.walk(tree)
                    if isinstance(node, ast.Assign)
                    and any(
                        isinstance(target, ast.Attribute)
                        and isinstance(target.value, ast.Name)
                        and target.value.id == "sys"
                        and target.attr == "dont_write_bytecode"
                        for target in node.targets
                    )
                ]
                project_import_lines = [
                    node.lineno
                    for node in ast.walk(tree)
                    if isinstance(node, ast.ImportFrom)
                    and isinstance(node.module, str)
                    and node.module.startswith("bench.")
                ]
                inherited_bytecode_guards = {
                    node.targets[0].slice.value: node.lineno
                    for node in ast.walk(tree)
                    if isinstance(node, ast.Assign)
                    and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Subscript)
                    and isinstance(node.targets[0].value, ast.Attribute)
                    and isinstance(node.targets[0].value.value, ast.Name)
                    and node.targets[0].value.value.id == "os"
                    and node.targets[0].value.attr == "environ"
                    and isinstance(node.targets[0].slice, ast.Constant)
                    and isinstance(node.targets[0].slice.value, str)
                    and node.targets[0].slice.value
                    in {"PYTHONDONTWRITEBYTECODE", "PYTHONPYCACHEPREFIX"}
                }
                self.assertTrue(pycache_lines)
                self.assertTrue(dont_write_lines)
                self.assertTrue(project_import_lines)
                self.assertEqual(
                    set(inherited_bytecode_guards),
                    {"PYTHONDONTWRITEBYTECODE", "PYTHONPYCACHEPREFIX"},
                )
                self.assertLess(max(pycache_lines), min(project_import_lines))
                self.assertLess(max(dont_write_lines), min(project_import_lines))
                self.assertLess(
                    max(inherited_bytecode_guards.values()),
                    min(project_import_lines),
                )

    def test_wrapper_prevents_spawned_python_from_mutating_the_environment(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory) / "project"
            scripts = project / "scripts"
            bench = project / "bench"
            scripts.mkdir(parents=True)
            bench.mkdir()
            wrapper = scripts / "cf-attention-probe"
            shutil.copy2(PROJECT_ROOT / "scripts" / "cf-attention-probe", wrapper)
            (bench / "__init__.py").write_text("", encoding="utf-8")
            (project / "child_module.py").write_text("VALUE = 1\n", encoding="utf-8")
            child_program = (
                "import os, sys\n"
                "if 'PYTHONPYCACHEPREFIX' not in os.environ:\n"
                "    sys.pycache_prefix = None\n"
                f"sys.path.insert(0, {str(project)!r})\n"
                "import child_module\n"
            )
            (bench / "cf_attention_probe.py").write_text(
                "import subprocess, sys\n"
                "def main():\n"
                f"    subprocess.run([sys.executable, '-S', '-c', {child_program!r}], check=True)\n"
                "    return 0\n",
                encoding="utf-8",
            )

            completed = subprocess.run(
                [sys.executable, "-I", "-S", str(wrapper)],
                check=False,
                capture_output=True,
                text=True,
            )
            generated = list(project.rglob("*.pyc"))

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(generated, [])

    def test_attention_wrapper_ignores_a_standard_unchecked_hash_pycache(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory) / "project"
            scripts = project / "scripts"
            bench = project / "bench"
            scripts.mkdir(parents=True)
            bench.mkdir()
            wrapper = scripts / "cf-attention-probe"
            shutil.copy2(PROJECT_ROOT / "scripts" / "cf-attention-probe", wrapper)
            (bench / "__init__.py").write_text("", encoding="utf-8")
            module = bench / "cf_attention_probe.py"
            module.write_text(
                "def main():\n    print('STALE-BYTECODE')\n    return 0\n",
                encoding="utf-8",
            )
            source_stat = module.stat()
            standard_pycache = bench / "__pycache__"
            standard_pycache.mkdir()
            compiled = standard_pycache / (
                f"cf_attention_probe.{sys.implementation.cache_tag}.pyc"
            )
            py_compile.compile(
                str(module),
                cfile=str(compiled),
                doraise=True,
                invalidation_mode=py_compile.PycInvalidationMode.TIMESTAMP,
            )
            module.write_text(
                "def main():\n    print('FRESH-SOURCE!!')\n    return 0\n",
                encoding="utf-8",
            )
            self.assertEqual(module.stat().st_size, source_stat.st_size)
            os.utime(
                module,
                ns=(source_stat.st_atime_ns, source_stat.st_mtime_ns),
            )

            control_program = (
                "import sys\n"
                f"sys.path.insert(0, {str(project)!r})\n"
                "from bench.cf_attention_probe import main\n"
                "raise SystemExit(main())\n"
            )
            control = subprocess.run(
                [sys.executable, "-I", "-S", "-c", control_program],
                check=False,
                capture_output=True,
                text=True,
            )

            completed = subprocess.run(
                [sys.executable, "-I", "-S", str(wrapper)],
                check=False,
                capture_output=True,
                text=True,
            )

        self.assertEqual(control.returncode, 0, control.stderr)
        self.assertEqual(control.stdout.strip(), "STALE-BYTECODE")
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout.strip(), "FRESH-SOURCE!!")


if __name__ == "__main__":
    unittest.main()
