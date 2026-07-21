from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

from bench.displacement_batch import BatchComparisonError, render_markdown, run_batch
from bench.coherence_metrics import CoherenceMetricError
from bench.displacement_metrics import DisplacementMetricError


def _fake_score_clip(
    source: str | Path,
    *,
    config: Any = None,
    track_backend: Any = None,
    appearance_backend: Any = None,
) -> dict[str, Any]:
    del config, track_backend, appearance_backend
    path = Path(source)
    scores = {
        ("alpha", "00-ball.mp4"): 2.0,
        ("alpha", "02-rolling-object.mp4"): 6.0,
        ("beta", "00-ball.mp4"): 3.0,
        ("beta", "01-walker.mp4"): 5.0,
        ("beta", "02-rolling-object.mp4"): 4.0,
    }
    score = scores[(path.parent.name, path.name)]
    return {
        "schema_version": 1,
        "kind": "coherent-displacement-metrics",
        "source": {"path": str(path.resolve())},
        "foreground": {
            "trajectory_span_fraction_width": score / 10.0,
            "track_survival": 0.875,
        },
        "opaque_nested_sentinel": {
            "preserve": ["exactly", {"score": score, "valid": True}],
        },
        "displacement_score": score,
        "displaced_vs_collapsed": score >= 5.0,
    }


def _tree_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _filename_score_clip(
    source: str | Path,
    *,
    config: Any = None,
    track_backend: Any = None,
    appearance_backend: Any = None,
) -> dict[str, Any]:
    del config, track_backend, appearance_backend
    path = Path(source)
    score = float(len(path.stem))
    return {
        "foreground": {
            "trajectory_span_fraction_width": score / 10.0,
            "track_survival": 0.75,
        },
        "displacement_score": score,
        "displaced_vs_collapsed": score >= 5.0,
    }


def _fake_coherence_score_clip(
    source: str | Path,
    *,
    config: Any = None,
    appearance_backend: Any = None,
    flow_backend: Any = None,
) -> dict[str, Any]:
    del config, appearance_backend, flow_backend
    path = Path(source)
    score = float(len(path.stem)) / 2.0
    return {
        "schema_version": 1,
        "kind": "video-coherence-metrics",
        "source": {"path": str(path.resolve())},
        "coherence_score": score,
        "temporal_coherence_score": score + 0.5,
        "spatial_integrity_score": score - 0.5,
        "degrades_over_time": path.parent.name == "alpha",
        "segments": [
            {
                "name": "early",
                "coherence_score": score + 1.0,
                "opaque_nested_sentinel": {"preserve": [True, score]},
            }
        ],
    }


class DisplacementBatchTests(unittest.TestCase):
    def _layout(self, root: Path) -> tuple[Path, Path]:
        source = root / "input"
        output = root / "output"
        for condition in ("alpha", "beta"):
            (source / condition).mkdir(parents=True)
        for relative in (
            "alpha/00-ball.mp4",
            "alpha/02-rolling-object.mp4",
            "beta/00-ball.mp4",
            "beta/01-walker.mp4",
            "beta/02-rolling-object.mp4",
        ):
            (source / relative).write_bytes(b"fixture")
        prompts = root / "prompts.txt"
        prompts.write_text("ball\nwalker\nrolling-object\n", encoding="utf-8")
        return source, output

    def test_missing_middle_clip_is_an_explicit_hole(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, output = self._layout(root)

            report = run_batch(
                source,
                prompts_path=root / "prompts.txt",
                output_dir=output,
                compare=("alpha", "beta"),
                scorer=_fake_score_clip,
            )

            matrix = {row["prompt"]: row for row in report["matrix"]}
            self.assertEqual(matrix["walker"]["cells"]["alpha"]["status"], "missing")
            self.assertEqual(
                matrix["rolling-object"]["cells"]["alpha"]["displacement_score"],
                6.0,
            )
            self.assertEqual(
                report["holes"],
                [
                    {
                        "condition": "alpha",
                        "filename": "01-walker.mp4",
                        "prompt": "walker",
                    }
                ],
            )
            comparison = {row["prompt"]: row for row in report["comparison"]["per_prompt"]}
            self.assertEqual(comparison["walker"]["status"], "missing")
            self.assertIn("missing", comparison["walker"]["verdict"])
            self.assertEqual(report["comparison"]["mean_delta_b_minus_a"], -0.5)
            self.assertIn("MISSING", (output / "matrix.md").read_text(encoding="utf-8"))

    def test_repeated_run_is_byte_identical_and_deltas_are_stable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, output = self._layout(root)
            arguments = {
                "prompts_path": root / "prompts.txt",
                "output_dir": output,
                "compare": ("alpha", "beta"),
                "scorer": _fake_score_clip,
            }

            first = run_batch(source, **arguments)
            first_bytes = _tree_bytes(output)
            second = run_batch(source, **arguments)
            second_bytes = _tree_bytes(output)

            self.assertEqual(first, second)
            self.assertEqual(first_bytes, second_bytes)
            per_prompt = first["comparison"]["per_prompt"]
            self.assertEqual(per_prompt[0]["verdict"], "beta higher by 1.000000")
            self.assertEqual(per_prompt[2]["verdict"], "alpha higher by 2.000000")
            self.assertEqual(len(first["clips"]), 5)
            self.assertEqual(
                len(list((output / "clips").glob("*/*.json"))),
                5,
            )
            first_clip = first["clips"][0]
            source_path = source / first_clip["condition"] / first_clip["filename"]
            self.assertEqual(
                json.loads((output / first_clip["result_json"]).read_text(encoding="utf-8")),
                _fake_score_clip(source_path),
            )
            self.assertEqual(
                json.loads((output / first_clip["result_json"]).read_text(encoding="utf-8"))[
                    "opaque_nested_sentinel"
                ],
                {"preserve": ["exactly", {"score": 2.0, "valid": True}]},
            )
            encoded_report = (output / "report.json").read_text(encoding="utf-8")
            self.assertIn('"displacement_score": 2.000000', encoded_report)
            self.assertIn('"mean_delta_b_minus_a": -0.500000', encoded_report)

    def test_no_flag_ignores_coherence_and_preserves_legacy_tree_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, first_output = self._layout(root)
            second_output = root / "second-output"
            baseline = run_batch(
                source,
                prompts_path=root / "prompts.txt",
                output_dir=first_output,
                compare=("alpha", "beta"),
                scorer=_fake_score_clip,
            )
            calls: list[Path] = []

            def forbidden_coherence(path: str | Path, **kwargs: Any) -> dict[str, Any]:
                del kwargs
                calls.append(Path(path))
                raise AssertionError("coherence must remain opt-in")

            candidate = run_batch(
                source,
                prompts_path=root / "prompts.txt",
                output_dir=second_output,
                compare=("alpha", "beta"),
                scorer=_fake_score_clip,
                coherence_scorer=forbidden_coherence,
            )

            self.assertEqual(calls, [])
            self.assertEqual(candidate, baseline)
            self.assertEqual(_tree_bytes(second_output), _tree_bytes(first_output))
            self.assertEqual(candidate["schema_version"], 3)
            self.assertNotIn("with_coherence", candidate)

    def test_with_coherence_preserves_full_report_and_renders_two_axes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, output = self._layout(root)

            report = run_batch(
                source,
                prompts_path=root / "prompts.txt",
                output_dir=output,
                scorer=_fake_score_clip,
                with_coherence=True,
                coherence_scorer=_fake_coherence_score_clip,
            )

            cell = report["matrix"][0]["cells"]["alpha"]
            self.assertEqual(cell["coherence_status"], "scored")
            self.assertEqual(cell["coherence_score"], 3.5)
            self.assertEqual(cell["temporal_coherence_score"], 4.0)
            self.assertEqual(cell["spatial_integrity_score"], 3.0)
            self.assertTrue(cell["degrades_over_time"])
            self.assertEqual(
                cell["coherence_result_json"],
                "coherence/alpha/00-ball.mp4.json",
            )
            raw = json.loads(
                (output / cell["coherence_result_json"]).read_text(encoding="utf-8")
            )
            self.assertEqual(raw, _fake_coherence_score_clip(source / "alpha/00-ball.mp4"))
            self.assertEqual(
                raw["segments"][0]["opaque_nested_sentinel"],
                {"preserve": [True, 3.5]},
            )
            self.assertEqual(report["schema_version"], 4)
            self.assertTrue(report["with_coherence"])
            markdown = (output / "matrix.md").read_text(encoding="utf-8")
            self.assertIn("alpha motion", markdown)
            self.assertIn("alpha coherence", markdown)
            first_bytes = _tree_bytes(output)
            repeated = run_batch(
                source,
                prompts_path=root / "prompts.txt",
                output_dir=output,
                scorer=_fake_score_clip,
                with_coherence=True,
                coherence_scorer=_fake_coherence_score_clip,
            )
            self.assertEqual(repeated, report)
            self.assertEqual(_tree_bytes(output), first_bytes)

    def test_coherence_error_prunes_only_stale_axis_json(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "input"
            output = root / "output"
            for condition in ("alpha", "beta"):
                (source / condition).mkdir(parents=True)
                (source / condition / "a.mp4").write_bytes(b"fixture")
            prompts = root / "prompts.txt"
            prompts.write_text("first\n", encoding="utf-8")
            run_batch(
                source,
                prompts_path=prompts,
                output_dir=output,
                compare=("alpha", "beta"),
                scorer=_filename_score_clip,
                with_coherence=True,
                coherence_scorer=_fake_coherence_score_clip,
            )
            stale = output / "coherence/alpha/a.mp4.json"
            self.assertTrue(stale.is_file())

            def failing_alpha(path: str | Path, **kwargs: Any) -> dict[str, Any]:
                if Path(path).parent.name == "alpha":
                    raise CoherenceMetricError("coherence backend refused clip")
                return _fake_coherence_score_clip(path, **kwargs)

            report = run_batch(
                source,
                prompts_path=prompts,
                output_dir=output,
                compare=("alpha", "beta"),
                scorer=_filename_score_clip,
                with_coherence=True,
                coherence_scorer=failing_alpha,
            )

            alpha = report["matrix"][0]["cells"]["alpha"]
            self.assertEqual(alpha["status"], "scored")
            self.assertEqual(alpha["coherence_status"], "error")
            self.assertIsNone(alpha["coherence_score"])
            self.assertFalse(stale.exists())
            self.assertEqual(report["comparison"]["comparable_prompt_count"], 1)
            self.assertEqual(report["coherence_errors"][0]["error_message"], "coherence backend refused clip")
            markdown = (output / "matrix.md").read_text(encoding="utf-8")
            self.assertIn("ERROR", markdown)

    def test_unexpected_coherence_error_aborts_without_replacing_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "input"
            output = root / "output"
            (source / "alpha").mkdir(parents=True)
            (source / "alpha/a.mp4").write_bytes(b"fixture")
            prompts = root / "prompts.txt"
            prompts.write_text("first\n", encoding="utf-8")
            run_batch(
                source,
                prompts_path=prompts,
                output_dir=output,
                scorer=_filename_score_clip,
                with_coherence=True,
                coherence_scorer=_fake_coherence_score_clip,
            )
            before = _tree_bytes(output)

            for error_type in (OSError, RuntimeError):
                with self.subTest(error_type=error_type.__name__):
                    def failing(path: str | Path, **kwargs: Any) -> dict[str, Any]:
                        del path, kwargs
                        raise error_type("unexpected coherence failure")

                    with self.assertRaisesRegex(error_type, "unexpected coherence failure"):
                        run_batch(
                            source,
                            prompts_path=prompts,
                            output_dir=output,
                            scorer=_filename_score_clip,
                            with_coherence=True,
                            coherence_scorer=failing,
                        )
                    self.assertEqual(_tree_bytes(output), before)

    def test_changed_rerun_prunes_stale_owned_clip_json(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, output = self._layout(root)
            arguments = {
                "prompts_path": root / "prompts.txt",
                "output_dir": output,
                "scorer": _fake_score_clip,
            }
            run_batch(source, **arguments)
            stale_path = output / "clips" / "beta" / "02-rolling-object.mp4.json"
            self.assertTrue(stale_path.is_file())
            (source / "beta" / "02-rolling-object.mp4").unlink()

            report = run_batch(source, **arguments)

            reported = {record["result_json"] for record in report["clips"]}
            on_disk = {
                path.relative_to(output).as_posix()
                for path in (output / "clips").rglob("*.json")
            }
            self.assertEqual(on_disk, reported)
            self.assertFalse(stale_path.exists())

    def test_output_directory_cannot_be_the_input_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "input"
            (source / "alpha").mkdir(parents=True)
            (source / "alpha" / "a.mp4").write_bytes(b"fixture")
            prompts = root / "prompts.txt"
            prompts.write_text("first\n", encoding="utf-8")

            for _attempt in range(2):
                with self.assertRaisesRegex(BatchComparisonError, "output directory"):
                    run_batch(
                        source,
                        prompts_path=prompts,
                        output_dir=source,
                        scorer=_filename_score_clip,
                    )
            self.assertFalse((source / "clips").exists())
            self.assertFalse((source / "report.json").exists())
            self.assertFalse((source / "matrix.md").exists())

    def test_default_learned_backends_are_constructed_once_and_shared(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "input"
            for relative in ("alpha/a.mp4", "beta/a.mp4"):
                path = source / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"fixture")
            prompts = root / "prompts.txt"
            prompts.write_text("first\n", encoding="utf-8")
            track_backend = object()
            appearance_backend = object()
            received: list[tuple[Any, Any]] = []

            def mocked_score(source_path: str | Path, **kwargs: Any) -> dict[str, Any]:
                del source_path
                received.append(
                    (kwargs["track_backend"], kwargs["appearance_backend"])
                )
                return {
                    "foreground": {
                        "trajectory_span_fraction_width": 0.1,
                        "track_survival": 0.9,
                    },
                    "displacement_score": 1.0,
                    "displaced_vs_collapsed": False,
                }

            with (
                patch(
                    "bench.displacement_batch.CoTracker3Backend",
                    return_value=track_backend,
                ) as tracker_constructor,
                patch(
                    "bench.displacement_batch.DINOv2AppearanceBackend",
                    return_value=appearance_backend,
                ) as appearance_constructor,
                patch("bench.displacement_batch.score_clip", side_effect=mocked_score),
            ):
                run_batch(
                    source,
                    prompts_path=prompts,
                    output_dir=root / "output",
                )

            tracker_constructor.assert_called_once_with("auto")
            appearance_constructor.assert_called_once_with("auto")
            self.assertEqual(received, [(track_backend, appearance_backend)] * 2)

    def test_coherence_reuses_one_dino_and_one_raft_backend(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "input"
            for relative in ("alpha/a.mp4", "beta/a.mp4"):
                path = source / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"fixture")
            prompts = root / "prompts.txt"
            prompts.write_text("first\n", encoding="utf-8")
            track_backend = object()
            appearance_backend = object()
            flow_backend = object()
            displacement_received: list[tuple[Any, Any]] = []
            coherence_received: list[tuple[Any, Any]] = []

            def mocked_displacement(
                source_path: str | Path,
                **kwargs: Any,
            ) -> dict[str, Any]:
                del source_path
                displacement_received.append(
                    (kwargs["track_backend"], kwargs["appearance_backend"])
                )
                return {
                    "foreground": {
                        "trajectory_span_fraction_width": 0.1,
                        "track_survival": 0.9,
                    },
                    "displacement_score": 1.0,
                    "displaced_vs_collapsed": False,
                }

            def mocked_coherence(
                source_path: str | Path,
                **kwargs: Any,
            ) -> dict[str, Any]:
                del source_path
                coherence_received.append(
                    (kwargs["appearance_backend"], kwargs["flow_backend"])
                )
                return {
                    "coherence_score": 8.0,
                    "degrades_over_time": False,
                    "spatial_integrity_score": 7.5,
                    "temporal_coherence_score": 8.5,
                }

            with (
                patch(
                    "bench.displacement_batch.CoTracker3Backend",
                    return_value=track_backend,
                ) as tracker_constructor,
                patch(
                    "bench.displacement_batch.DINOv2AppearanceBackend",
                    return_value=appearance_backend,
                ) as appearance_constructor,
                patch(
                    "bench.displacement_batch.RaftSmallFlowBackend",
                    return_value=flow_backend,
                ) as flow_constructor,
                patch(
                    "bench.displacement_batch.score_clip",
                    side_effect=mocked_displacement,
                ),
                patch(
                    "bench.displacement_batch.score_coherence_clip",
                    side_effect=mocked_coherence,
                ),
            ):
                run_batch(
                    source,
                    prompts_path=prompts,
                    output_dir=root / "output",
                    with_coherence=True,
                )

            tracker_constructor.assert_called_once_with("auto")
            appearance_constructor.assert_called_once_with("auto")
            flow_constructor.assert_called_once_with("auto", batch_size=4)
            self.assertEqual(
                displacement_received,
                [(track_backend, appearance_backend)] * 2,
            )
            self.assertEqual(
                coherence_received,
                [(appearance_backend, flow_backend)] * 2,
            )

    def test_prompt_substrings_override_filename_sort(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "input"
            filenames = (
                "zz bright red rubber ball action.mp4",
                "aa adult in a blue jacket action.mp4",
            )
            for condition in ("alpha", "beta"):
                (source / condition).mkdir(parents=True)
                for filename in filenames:
                    (source / condition / filename).write_bytes(b"fixture")
            prompts = root / "prompts.txt"
            prompts.write_text("ball\nwalker\n", encoding="utf-8")

            report = run_batch(
                source,
                prompts_path=prompts,
                output_dir=root / "output",
                prompt_substrings={
                    "ball": "bright red rubber ball",
                    "walker": "blue jacket",
                },
                scorer=_filename_score_clip,
            )

            self.assertEqual(report["slot_mode"], "substring")
            self.assertEqual(
                report["matrix"][0]["cells"]["alpha"]["filename"],
                "zz bright red rubber ball action.mp4",
            )
            self.assertEqual(
                report["matrix"][1]["cells"]["alpha"]["filename"],
                "aa adult in a blue jacket action.mp4",
            )

    def test_invalid_condition_is_reported_without_scoring(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "input"
            for condition in ("valid", "oneforcing"):
                (source / condition).mkdir(parents=True)
                (source / condition / "a.mp4").write_bytes(b"fixture")
            prompts = root / "prompts.txt"
            prompts.write_text("first\n", encoding="utf-8")
            scored_conditions: list[str] = []

            def recording_scorer(path: str | Path, **kwargs: Any) -> dict[str, Any]:
                del kwargs
                scored_conditions.append(Path(path).parent.name)
                return _filename_score_clip(path)

            invalid_spec = {
                "oneforcing": {
                    "reason_code": "checkpoint_inference_path_incompatible",
                    "reason": "Checkpoint requires its native inference path.",
                    "remediation": "Rerun in the native repository before scoring.",
                }
            }
            report = run_batch(
                source,
                prompts_path=prompts,
                output_dir=root / "output",
                invalid_conditions=invalid_spec,
                scorer=recording_scorer,
            )

            self.assertEqual(scored_conditions, ["valid"])
            invalid_cell = report["matrix"][0]["cells"]["oneforcing"]
            self.assertEqual(invalid_cell["status"], "invalid")
            self.assertEqual(
                invalid_cell["reason_code"],
                "checkpoint_inference_path_incompatible",
            )
            self.assertIsNone(invalid_cell["displacement_score"])
            self.assertIsNone(invalid_cell["result_json"])
            self.assertEqual(len(report["invalid"]), 1)
            self.assertEqual(report["schema_version"], 3)
            self.assertIn("INVALID", (root / "output" / "matrix.md").read_text())
            self.assertFalse((root / "output" / "clips" / "oneforcing").exists())

    def test_metric_error_is_reported_and_stale_clip_json_is_pruned(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "input"
            output = root / "output"
            for condition in ("alpha", "beta"):
                (source / condition).mkdir(parents=True)
                (source / condition / "a.mp4").write_bytes(b"fixture")
            prompts = root / "prompts.txt"
            prompts.write_text("first\n", encoding="utf-8")

            run_batch(
                source,
                prompts_path=prompts,
                output_dir=output,
                compare=("alpha", "beta"),
                scorer=_filename_score_clip,
            )
            stale = output / "clips" / "alpha" / "a.mp4.json"
            self.assertTrue(stale.is_file())
            calls: list[str] = []

            def failing_alpha(path: str | Path, **kwargs: Any) -> dict[str, Any]:
                del kwargs
                condition = Path(path).parent.name
                calls.append(condition)
                if condition == "alpha":
                    raise DisplacementMetricError("too few persistent tracks")
                return _filename_score_clip(path)

            report = run_batch(
                source,
                prompts_path=prompts,
                output_dir=output,
                compare=("alpha", "beta"),
                scorer=failing_alpha,
            )

            self.assertEqual(calls, ["alpha", "beta"])
            error_cell = report["matrix"][0]["cells"]["alpha"]
            self.assertEqual(
                error_cell,
                {
                    "condition": "alpha",
                    "displaced": None,
                    "displacement_score": None,
                    "error_message": "too few persistent tracks",
                    "error_type": "DisplacementMetricError",
                    "filename": "a.mp4",
                    "prompt": "first",
                    "result_json": None,
                    "span_fraction": None,
                    "status": "error",
                    "survival": None,
                },
            )
            self.assertEqual(len(report["errors"]), 1)
            self.assertEqual(report["comparison"]["error_prompt_count"], 1)
            self.assertEqual(report["comparison"]["comparable_prompt_count"], 0)
            self.assertEqual(report["comparison"]["per_prompt"][0]["status"], "error")
            self.assertIsNone(
                report["comparison"]["per_prompt"][0]["delta_b_minus_a"]
            )
            self.assertFalse(stale.exists())
            markdown = (output / "matrix.md").read_text(encoding="utf-8")
            self.assertIn("ERROR", markdown)
            self.assertIn("UNAVAILABLE", markdown)

    def test_unexpected_non_metric_error_propagates_without_replacing_outputs(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "input"
            output = root / "output"
            (source / "alpha").mkdir(parents=True)
            (source / "alpha" / "a.mp4").write_bytes(b"fixture")
            prompts = root / "prompts.txt"
            prompts.write_text("first\n", encoding="utf-8")
            run_batch(
                source,
                prompts_path=prompts,
                output_dir=output,
                scorer=_filename_score_clip,
            )
            before = _tree_bytes(output)

            for error_type in (OSError, RuntimeError):
                with self.subTest(error_type=error_type.__name__):
                    def failing_scorer(
                        path: str | Path,
                        **kwargs: Any,
                    ) -> dict[str, Any]:
                        del path, kwargs
                        raise error_type("unexpected scorer failure")

                    with self.assertRaisesRegex(
                        error_type,
                        "unexpected scorer failure",
                    ):
                        run_batch(
                            source,
                            prompts_path=prompts,
                            output_dir=output,
                            scorer=failing_scorer,
                        )

                    self.assertEqual(_tree_bytes(output), before)

    def test_error_row_is_excluded_from_nonempty_comparison_mean(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "input"
            for condition in ("alpha", "beta"):
                (source / condition).mkdir(parents=True)
                for filename in ("a.mp4", "b.mp4"):
                    (source / condition / filename).write_bytes(b"fixture")
            prompts = root / "prompts.txt"
            prompts.write_text("first\nsecond\n", encoding="utf-8")

            def scorer(path: str | Path, **kwargs: Any) -> dict[str, Any]:
                del kwargs
                resolved = Path(path)
                if resolved.parent.name == "alpha" and resolved.name == "a.mp4":
                    raise DisplacementMetricError("unscorable first clip")
                scores = {
                    ("alpha", "b.mp4"): 2.0,
                    ("beta", "a.mp4"): 10.0,
                    ("beta", "b.mp4"): 5.0,
                }
                score = scores[(resolved.parent.name, resolved.name)]
                return {
                    "foreground": {
                        "trajectory_span_fraction_width": score / 10.0,
                        "track_survival": 0.9,
                    },
                    "displacement_score": score,
                    "displaced_vs_collapsed": False,
                }

            report = run_batch(
                source,
                prompts_path=prompts,
                output_dir=root / "output",
                compare=("alpha", "beta"),
                scorer=scorer,
            )

            comparison = report["comparison"]
            self.assertEqual(comparison["error_prompt_count"], 1)
            self.assertEqual(comparison["comparable_prompt_count"], 1)
            self.assertEqual(comparison["mean_delta_b_minus_a"], 3.0)
            self.assertEqual(
                [row["status"] for row in comparison["per_prompt"]],
                ["error", "compared"],
            )

    def test_multiple_comparisons_share_one_scoring_pass(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "input"
            scores = {"alpha": 1.0, "beta": 2.0, "gamma": 4.0}
            calls: list[str] = []
            for condition in scores:
                (source / condition).mkdir(parents=True)
                (source / condition / "a.mp4").write_bytes(b"fixture")
            prompts = root / "prompts.txt"
            prompts.write_text("first\n", encoding="utf-8")

            def scorer(path: str | Path, **kwargs: Any) -> dict[str, Any]:
                del kwargs
                condition = Path(path).parent.name
                calls.append(condition)
                score = scores[condition]
                return {
                    "foreground": {
                        "trajectory_span_fraction_width": score / 10.0,
                        "track_survival": 0.9,
                    },
                    "displacement_score": score,
                    "displaced_vs_collapsed": False,
                }

            report = run_batch(
                source,
                prompts_path=prompts,
                output_dir=root / "output",
                comparisons=(("alpha", "beta"), ("alpha", "gamma")),
                scorer=scorer,
            )

            self.assertEqual(calls, ["alpha", "beta", "gamma"])
            self.assertEqual(len(report["comparisons"]), 2)
            self.assertEqual(
                [item["mean_delta_b_minus_a"] for item in report["comparisons"]],
                [1.0, 3.0],
            )
            markdown = (root / "output" / "matrix.md").read_text(encoding="utf-8")
            self.assertEqual(markdown.count("## Comparison:"), 2)

    def test_degraded_score_is_marked_and_excluded_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "input"
            for condition in ("alpha", "beta"):
                (source / condition).mkdir(parents=True)
                (source / condition / "a.mp4").write_bytes(b"fixture")
            prompts = root / "prompts.txt"
            prompts.write_text("first\n", encoding="utf-8")

            def scorer(path: str | Path, **kwargs: Any) -> dict[str, Any]:
                del kwargs
                condition = Path(path).parent.name
                score = 2.0 if condition == "alpha" else 5.0
                report = {
                    "foreground": {
                        "trajectory_span_fraction_width": score / 10.0,
                        "track_survival": 0.9,
                    },
                    "displacement_score": score,
                    "displaced_vs_collapsed": False,
                }
                if condition == "alpha":
                    report["coherence_degraded"] = True
                    report["camera_compensated"] = False
                return report

            report = run_batch(
                source,
                prompts_path=prompts,
                output_dir=root / "output",
                compare=("alpha", "beta"),
                scorer=scorer,
            )

            alpha = report["matrix"][0]["cells"]["alpha"]
            self.assertIs(alpha["coherence_degraded"], True)
            self.assertIs(alpha["camera_compensated"], False)
            comparison = report["comparison"]
            self.assertIs(comparison["include_degraded"], False)
            self.assertEqual(comparison["degraded_prompt_count"], 1)
            self.assertEqual(comparison["comparable_prompt_count"], 0)
            self.assertIsNone(comparison["mean_delta_b_minus_a"])
            row = comparison["per_prompt"][0]
            self.assertEqual(row["status"], "degraded")
            self.assertEqual(row["degraded_conditions"], ["alpha"])
            self.assertIsNone(row["delta_b_minus_a"])
            markdown = (root / "output" / "matrix.md").read_text(encoding="utf-8")
            self.assertIn("2.000000*", markdown)
            self.assertIn("Screen-space-only fallback", markdown)
            self.assertIn("DEGRADED", markdown)
            self.assertIn("UNAVAILABLE", markdown)

    def test_include_degraded_opt_in_compares_and_keeps_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "input"
            for condition in ("alpha", "beta"):
                (source / condition).mkdir(parents=True)
                (source / condition / "a.mp4").write_bytes(b"fixture")
            prompts = root / "prompts.txt"
            prompts.write_text("first\n", encoding="utf-8")

            def scorer(path: str | Path, **kwargs: Any) -> dict[str, Any]:
                del kwargs
                condition = Path(path).parent.name
                score = 2.0 if condition == "alpha" else 5.0
                return {
                    "camera_compensated": condition != "alpha",
                    "coherence_degraded": condition == "alpha",
                    "foreground": {
                        "trajectory_span_fraction_width": score / 10.0,
                        "track_survival": 0.9,
                    },
                    "displacement_score": score,
                    "displaced_vs_collapsed": False,
                }

            report = run_batch(
                source,
                prompts_path=prompts,
                output_dir=root / "output",
                compare=("alpha", "beta"),
                include_degraded=True,
                scorer=scorer,
            )

            comparison = report["comparison"]
            self.assertIs(report["include_degraded_comparisons"], True)
            self.assertIs(comparison["include_degraded"], True)
            self.assertEqual(comparison["degraded_prompt_count"], 1)
            self.assertEqual(comparison["comparable_prompt_count"], 1)
            self.assertEqual(comparison["mean_delta_b_minus_a"], 3.0)
            row = comparison["per_prompt"][0]
            self.assertEqual(row["status"], "compared")
            self.assertEqual(row["degraded_conditions"], ["alpha"])
            self.assertEqual(row["delta_b_minus_a"], 3.0)
            markdown = (root / "output" / "matrix.md").read_text(encoding="utf-8")
            self.assertIn("| first | 2.000000* | 5.000000 | 3.000000 |", markdown)

    def test_partial_degraded_markers_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "input"
            (source / "alpha").mkdir(parents=True)
            (source / "alpha" / "a.mp4").write_bytes(b"fixture")
            prompts = root / "prompts.txt"
            prompts.write_text("first\n", encoding="utf-8")

            def scorer(path: str | Path, **kwargs: Any) -> dict[str, Any]:
                report = _filename_score_clip(path, **kwargs)
                report["coherence_degraded"] = True
                return report

            with self.assertRaisesRegex(BatchComparisonError, "degraded markers"):
                run_batch(
                    source,
                    prompts_path=prompts,
                    output_dir=root / "output",
                    scorer=scorer,
                )

    def test_p0_fixture_pins_all_matrix_fields_and_default_models(self) -> None:
        fixture_path = (
            Path(__file__).resolve().parents[1]
            / "fixtures"
            / "p0_displacement_calibration.json"
        )
        fixture = json.loads(fixture_path.read_text(encoding="utf-8"))

        self.assertEqual(
            fixture["tracker"],
            "facebookresearch/co-tracker:cotracker3_offline",
        )
        self.assertEqual(
            fixture["appearance"],
            "facebookresearch/dinov2:dinov2_vits14",
        )
        for row in fixture["rows"]:
            with self.subTest(blind_id=row["blind_id"]):
                self.assertIn("survival", row)
                self.assertIsInstance(row["survival"], float)

    def test_prompt_count_must_match_filename_slots(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, output = self._layout(root)
            (root / "prompts.txt").write_text("ball\nwalker\n", encoding="utf-8")

            with self.assertRaisesRegex(BatchComparisonError, "prompt count"):
                run_batch(
                    source,
                    prompts_path=root / "prompts.txt",
                    output_dir=output,
                    scorer=_fake_score_clip,
                )

    def test_nested_uppercase_mp4_is_rejected_instead_of_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "input"
            (source / "alpha").mkdir(parents=True)
            (source / "alpha" / "00-ball.mp4").write_bytes(b"fixture")
            nested = source / "alpha" / "nested"
            nested.mkdir()
            (nested / "hidden.MP4").write_bytes(b"fixture")
            prompts = root / "prompts.txt"
            prompts.write_text("ball\n", encoding="utf-8")

            with self.assertRaisesRegex(BatchComparisonError, "direct"):
                run_batch(
                    source,
                    prompts_path=prompts,
                    output_dir=root / "output",
                    scorer=_filename_score_clip,
                )

    def test_arbitrary_filenames_use_positional_order_and_report_trailing_hole(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "input"
            for relative in (
                "alpha/a.mp4",
                "alpha/z-long.mp4",
                "beta/different.mp4",
            ):
                path = source / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"fixture")
            prompts = root / "prompts.txt"
            prompts.write_text("first\nsecond\n", encoding="utf-8")

            report = run_batch(
                source,
                prompts_path=prompts,
                output_dir=root / "output",
                compare=("alpha", "beta"),
                scorer=_filename_score_clip,
            )

            self.assertEqual(report["slot_mode"], "positional")
            self.assertEqual(
                report["matrix"][0]["cells"]["beta"]["filename"],
                "different.mp4",
            )
            self.assertEqual(report["matrix"][1]["cells"]["beta"]["status"], "missing")
            self.assertEqual(
                report["holes"],
                [{"condition": "beta", "filename": None, "prompt": "second"}],
            )

    def test_distinct_singleton_filenames_are_both_the_first_position(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "input"
            for relative in ("alpha/a.mp4", "beta/z.mp4"):
                path = source / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"fixture")
            prompts = root / "prompts.txt"
            prompts.write_text("first\nsecond\n", encoding="utf-8")

            report = run_batch(
                source,
                prompts_path=prompts,
                output_dir=root / "output",
                scorer=_filename_score_clip,
            )

            self.assertEqual(report["slot_mode"], "positional")
            self.assertEqual(report["matrix"][0]["cells"]["alpha"]["filename"], "a.mp4")
            self.assertEqual(report["matrix"][0]["cells"]["beta"]["filename"], "z.mp4")
            self.assertEqual(report["matrix"][1]["cells"]["alpha"]["status"], "missing")
            self.assertEqual(report["matrix"][1]["cells"]["beta"]["status"], "missing")

    def test_markdown_escapes_emphasis_in_arbitrary_names(self) -> None:
        report = {
            "conditions": ["model_A"],
            "matrix": [
                {
                    "prompt": "*ball*",
                    "cells": {
                        "model_A": {
                            "displaced": False,
                            "displacement_score": 1.0,
                            "span_fraction": 0.1,
                            "status": "scored",
                            "survival": 0.9,
                        }
                    },
                }
            ],
        }

        markdown = render_markdown(report)

        self.assertIn("model\\_A score", markdown)
        self.assertIn("\\*ball\\*", markdown)


@unittest.skipUnless(
    os.environ.get("P0_DISPLACEMENT_ROOT"),
    "set P0_DISPLACEMENT_ROOT for the learned-backend P0 regression",
)
class P0DisplacementBatchRegressionTests(unittest.TestCase):
    def test_all_eleven_clips_reproduce_session_21_fixture_exactly(self) -> None:
        p0_root = Path(os.environ["P0_DISPLACEMENT_ROOT"]).resolve()
        fixture_path = (
            Path(__file__).resolve().parents[1]
            / "fixtures"
            / "p0_displacement_calibration.json"
        )
        fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
        expected = {row["blind_id"]: row for row in fixture["rows"]}
        sources = {
            ("ring-off", "00-ball.mp4", "m07"): "cell-a-off/ball/clip.mp4",
            ("ring-off", "01-walker.mp4", "v18"): "cell-a-off/walker/clip.mp4",
            ("ring-off", "02-rolling-object.mp4", "p62"): "cell-a-off/rolling-object/clip.mp4",
            ("ring-off", "03-vehicle.mp4", "x27"): "cell-a-off/vehicle/clip.mp4",
            ("ring-on", "00-ball.mp4", "q31"): "cell-a-on/ball/clip.mp4",
            ("ring-on", "01-walker.mp4", "c44"): "cell-a-on/walker/clip.mp4",
            ("ring-on", "02-rolling-object.mp4", "h09"): "cell-a-on/rolling-object/clip.mp4",
            ("ring-on", "03-vehicle.mp4", "b53"): "cell-a-on/vehicle/clip.mp4",
            ("d1-rolling-taehv", "00-ball.mp4", "k14"): "cell-b/d1-rolling-taehv/clip.mp4",
            ("d2-reset-every-block", "00-ball.mp4", "s80"): "cell-b/d2-reset-every-block/clip.mp4",
            ("d0-full-wan-vae", "00-ball.mp4", "r36"): "cell-c-wan/clip.mp4",
        }

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            batch_root = root / "input"
            output = root / "output"
            for condition, filename, _blind_id in sources:
                (batch_root / condition).mkdir(parents=True, exist_ok=True)
                (batch_root / condition / filename).symlink_to(
                    p0_root / sources[(condition, filename, _blind_id)]
                )
            prompts = root / "prompts.txt"
            prompts.write_text(
                "ball\nwalker\nrolling-object\nvehicle\n",
                encoding="utf-8",
            )

            report = run_batch(
                batch_root,
                prompts_path=prompts,
                output_dir=output,
                compare=("ring-off", "ring-on"),
            )

            by_key = {
                (record["condition"], record["filename"]): record
                for record in report["clips"]
            }
            for condition, filename, blind_id in sources:
                with self.subTest(blind_id=blind_id):
                    record = by_key[(condition, filename)]
                    full_report = json.loads(
                        (output / record["result_json"]).read_text(encoding="utf-8")
                    )
                    self.assertEqual(
                        full_report["displacement_score"],
                        expected[blind_id]["metric_score"],
                    )
                    self.assertEqual(
                        full_report["foreground"]["trajectory_span_fraction_width"],
                        expected[blind_id]["span_fraction"],
                    )
                    self.assertEqual(
                        full_report["displaced_vs_collapsed"],
                        expected[blind_id]["displaced"],
                    )
                    self.assertEqual(
                        full_report["foreground"]["track_survival"],
                        expected[blind_id]["survival"],
                    )
                    self.assertEqual(full_report["models"]["tracker"], fixture["tracker"])
                    self.assertEqual(
                        full_report["models"]["appearance"],
                        fixture["appearance"],
                    )
                    self.assertEqual(
                        {
                            "frame_count": full_report["source"]["frame_count"],
                            "height": full_report["source"]["height"],
                            "width": full_report["source"]["width"],
                        },
                        fixture["clip_contract"],
                    )
                    self.assertEqual(
                        full_report["decision_threshold"],
                        fixture["calibration"]["decision_threshold"],
                    )
                    self.assertNotIn("coherence_degraded", full_report)
                    self.assertNotIn("camera_compensated", full_report)
                    self.assertIs(record["coherence_degraded"], False)
                    self.assertIs(record["camera_compensated"], True)
                    self.assertEqual(
                        {
                            "displacement_score": record["displacement_score"],
                            "span_fraction": record["span_fraction"],
                            "survival": record["survival"],
                            "displaced": record["displaced"],
                        },
                        {
                            "displacement_score": expected[blind_id]["metric_score"],
                            "span_fraction": round(expected[blind_id]["span_fraction"], 6),
                            "survival": round(expected[blind_id]["survival"], 6),
                            "displaced": expected[blind_id]["displaced"],
                        },
                    )

            self.assertEqual(len(report["clips"]), 11)
            self.assertEqual(len(report["holes"]), 9)


if __name__ == "__main__":
    unittest.main()
