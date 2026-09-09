#!/usr/bin/env python3
"""Experiment plumbing: input validation, checkpoints, resume and provenance."""
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

try:
    import numpy as np
except ImportError:
    print("NumPy unavailable: install requirements-dev.txt to run experiment regressions")
    sys.exit(77)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
import insert_strategies as exp
from compare_to_paper import f1
from python_support import default_lges_repo, resolve_lges_repo


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="xges-experiment-tests-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.args = exp.argument_parser().parse_args([
            "--num-vars", "3", "--num-samples", "12", "--num-graphs", "1",
            "--out-dir", str(self.root), "--binary", sys.executable])

    def test_reference_layouts(self):
        project = self.root / "project with spaces"
        reference = (project / "lges").resolve()
        utility = reference / "ges/ges/utils.py"
        utility.parent.mkdir(parents=True)
        utility.write_text("")
        (reference / "src/experiments").mkdir(parents=True)
        (project / "ges/ges").mkdir(parents=True)
        (project / "ges/ges/utils.py").write_text("")
        self.assertEqual(default_lges_repo(reference / "cpp"), reference)
        self.assertEqual(default_lges_repo(project / "standalone"), reference)
        self.assertEqual(default_lges_repo(project / "code/standalone"), reference)
        self.assertEqual(resolve_lges_repo(reference), reference)
        with self.assertRaisesRegex(FileNotFoundError, "--lges-repo"):
            resolve_lges_repo(project / "missing")

    def test_helpers_from_another_working_directory(self):
        root = Path(__file__).resolve().parents[1]
        for script in ("experiments/paper_main.py", "experiments/insert_strategies.py",
                       "experiments/compare_to_paper.py", "tests/gen_fixtures.py"):
            proc = subprocess.run([sys.executable, str(root / script), "--help"],
                                  cwd=self.root, capture_output=True, text=True, timeout=10)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertIn("--lges-repo", proc.stdout)

    def manifest(self, args, name):
        return {"settings": {"num_vars": args.num_vars, "num_graphs": args.num_graphs,
                              "num_samples": args.num_samples, "er": args.er,
                              "alpha": 1, "normalization": "always_row_l1"}, "report": name}

    def invoke(self, run_one):
        with patch.object(exp, "run_manifest", side_effect=self.manifest), \
                patch.object(exp, "run_one", side_effect=run_one), contextlib.redirect_stdout(io.StringIO()):
            return exp.run_experiment(self.args, lambda dag: dag + dag.T)

    def test_failed_run_resume_and_torn_append(self):
        temp_paths = []
        calls = 0
        def interrupted(data_path, tmp, flags, strategy, p, **kwargs):
            nonlocal calls
            calls += 1
            temp_paths.append(Path(tmp))
            if calls == 3:
                raise RuntimeError("simulated failed search")
            return np.zeros((p, p), dtype=int), {"time": 0.01, "score": -10, "score check": -10}
        with self.assertRaisesRegex(RuntimeError, "simulated"):
            self.invoke(interrupted)
        self.assertTrue(all(not path.exists() for path in temp_paths))
        partial = self.root / "insert_strategies.partial.csv"
        self.assertEqual(len(exp.read_csv(partial)), 2)
        metadata = json.loads((self.root / "insert_strategies.partial.json").read_text())
        self.assertEqual(metadata["status"], "failed")
        with partial.open("a") as f:
            f.write("3,0,torn row")  # interrupted between append and checkpoint commit
        self.args.resume = True
        calls = 0
        def remaining(data_path, tmp, flags, strategy, p, **kwargs):
            nonlocal calls
            calls += 1
            return np.zeros((p, p), dtype=int), {"time": 0.01, "score": -10, "score check": -10}
        rows, metadata = self.invoke(remaining)
        self.assertEqual(calls, 7)
        self.assertEqual(len(rows), 9)
        self.assertEqual(metadata["status"], "complete")
        self.assertFalse(partial.exists())
        saved, saved_meta = exp.load_results(self.root)
        self.assertEqual(saved, rows)
        self.assertEqual(saved_meta, metadata)
        self.assertIn("n=12", exp.result_caption(saved_meta))
        # Plot-only must use saved metadata, never current defaults.
        self.args.num_samples = 999
        self.assertNotIn("999", exp.result_caption(saved_meta))
        final = self.root / "insert_strategies.csv"
        final.write_text(final.read_text().replace("0.01", "0.02", 1))
        with self.assertRaisesRegex(ValueError, "do not match"):
            exp.load_results(self.root)

    def test_interrupt_before_checkpoint_commit(self):
        write_json = exp.write_json
        def interrupt_manifest(value, path):
            if value.get("status") == "running" and value.get("completed_rows") == 1:
                raise KeyboardInterrupt()
            write_json(value, path)
        def result(data_path, tmp, flags, strategy, p, **kwargs):
            return np.zeros((p, p), dtype=int), {"time": 0.01, "score": -10, "score check": -10}
        with patch.object(exp, "write_json", side_effect=interrupt_manifest):
            with self.assertRaises(KeyboardInterrupt):
                self.invoke(result)
        meta = json.loads((self.root / "insert_strategies.partial.json").read_text())
        self.assertEqual(meta["status"], "interrupted")
        self.assertEqual(meta["completed_rows"], 0)
        self.assertEqual(meta["checkpoint_bytes"], 0)
        self.args.resume = True
        rows, _ = self.invoke(result)
        self.assertEqual(len(rows), 9)

    def test_resume_rejects_changed_settings(self):
        def fail(*args, **kwargs):
            raise RuntimeError("failure before first row")
        with self.assertRaises(RuntimeError):
            self.invoke(fail)
        self.args.resume = True
        self.args.num_samples += 1
        with self.assertRaisesRegex(ValueError, "Cannot resume"):
            self.invoke(fail)

    def test_invalid_arguments(self):
        for name, value in [("num_vars", [1]), ("num_vars", [3, 3]), ("num_samples", 1),
                            ("num_graphs", 0), ("seed", -1), ("er", float("nan")),
                            ("timeout", 0), ("binary", str(self.root / "missing"))]:
            original = getattr(self.args, name)
            setattr(self.args, name, value)
            with self.subTest(name=name), self.assertRaises(ValueError):
                exp.validate_args(self.args)
            setattr(self.args, name, original)

    def test_graph_reader(self):
        graph = self.root / "graph.csv"
        for text in ["0,1\n0,1\n", "0,0\n0,1\n1,0\n", "0,1\n0,2\n1,0\n", "0,1\n1,0\n0,0\n"]:
            graph.write_text(text)
            with self.assertRaises(ValueError):
                exp.read_cpdag(graph, 2)
        graph.write_text("1,0\n0,1\n0,0\n")
        np.testing.assert_array_equal(exp.read_cpdag(graph, 2), [[0, 0], [1, 0]])

    def test_subprocess_failures_and_invalid_statistics(self):
        with patch.object(exp.subprocess, "run", side_effect=subprocess.TimeoutExpired("xges", 1)):
            with self.assertRaisesRegex(RuntimeError, "exceeded"):
                exp.run_one("data.npy", self.root, [], "standard", 2, timeout=1)
        with patch.object(exp.subprocess, "run", return_value=subprocess.CompletedProcess([], 4, "stdout", "stderr")):
            with self.assertRaisesRegex(RuntimeError, "exited 4"):
                exp.run_one("data.npy", self.root, [], "standard", 2)
        def output(stats_text):
            def run(cmd, **kwargs):
                Path(cmd[cmd.index("--output") + 1]).write_text("0,1\n0,0\n0,0\n")
                Path(cmd[cmd.index("--stats") + 1]).write_text(stats_text)
                return subprocess.CompletedProcess(cmd, 0, "", "")
            return run
        for stats in ["time,nan\nscore,1\nscore check,1\n", "time,1\n", "time,-1\nscore,1\nscore check,1\n",
                      "time,1\nscore,1\nscore check,2\n", "time,1\nscore,1\nscore check,1\nscore,1\n"]:
            with patch.object(exp.subprocess, "run", side_effect=output(stats)), self.assertRaises(ValueError):
                exp.run_one("data.npy", self.root, [], "standard", 2)

    def test_f1_empty_cases_and_csv_validation(self):
        empty = np.zeros((2, 2), dtype=int)
        edge = np.array([[0, 1], [0, 0]])
        self.assertEqual(f1(empty, empty), (1, 1, 1))
        self.assertEqual(f1(edge, empty), (0, 0, 0))
        self.assertEqual(f1(empty, edge), (0, 0, 0))
        self.assertEqual(f1(edge, edge), (1, 1, 1))
        malformed = self.root / "bad.csv"
        malformed.write_text("p,graph\n3,0\n")
        with self.assertRaises(ValueError):
            exp.read_csv(malformed)
        self.assertIn("unrecorded", exp.result_caption(None))


if __name__ == "__main__":
    unittest.main()
