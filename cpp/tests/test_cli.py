#!/usr/bin/env python3
"""CLI regressions use only Python's standard library (no NumPy installation)."""
import csv
import math
from pathlib import Path
import random
import struct
import subprocess
import sys
import tempfile
import unittest

BINARY = str(Path(sys.argv[1]).resolve())
del sys.argv[1]


def npy_bytes(shape, values, dtype="<f8", fortran=False, version=1, header=None):
    if header is None:
        header = repr({"descr": dtype, "fortran_order": fortran, "shape": shape})
    prefix_size = 10 if version == 1 else 12
    header = header.encode("utf-8")
    header += b" " * ((-(prefix_size + len(header) + 1)) % 64) + b"\n"
    prefix = b"\x93NUMPY" + bytes([version, 0])
    length = struct.pack("<H" if version == 1 else "<I", len(header))
    codes = {"f8": "d", "f4": "f", "i8": "q", "u8": "Q"}
    payload = struct.pack(dtype[0] + codes[dtype[1:]] * len(values), *values)
    return prefix + length + header + payload


class CLITests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="xges-cli-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.input = self.root / "input.npy"
        rng = random.Random(834)
        self.data = [[rng.gauss(0, 1) for _ in range(3)] for _ in range(80)]
        for row in self.data:
            row[1] += 0.9 * row[0]
            row[2] += 0.8 * row[1]
        self.input.write_bytes(npy_bytes((80, 3), [x for row in self.data for x in row]))
        self.output = self.root / "graph.csv"
        self.stats = self.root / "stats.csv"

    def run_cli(self, *flags, ok=True, input_path=None):
        proc = subprocess.run([BINARY, "--input", str(input_path or self.input),
                               "--output", str(self.output), "--stats", str(self.stats),
                               "-v0", "-a1", *flags], capture_output=True, text=True, timeout=15)
        self.assertNotIn("AddressSanitizer", proc.stderr)
        self.assertNotIn("runtime error:", proc.stderr)
        if ok:
            self.assertEqual(proc.returncode, 0, proc.stderr)
            with self.stats.open() as f:
                stats = {row[0].strip(): float(row[1]) for row in csv.reader(f)}
            self.assertTrue(all(math.isfinite(value) for value in stats.values()))
            self.assertAlmostEqual(stats["score"], stats["score check"], places=7)
            self.assertGreaterEqual(stats["time"], 0)
            # Search score counters exclude initial and final graph scoring.
            families = stats.get("score_insert-#calls", 0) + stats.get("score_delete-#calls", 0)
            self.assertEqual(stats.get("local_score-#calls-total", 0),
                             2 * (families + stats.get("lges-gate-#calls", 0)))
            return self.output.read_text(), stats
        self.assertEqual(proc.returncode, 2, proc.stderr)
        self.assertIn("error:", proc.stderr)
        return proc

    def test_all_modes(self):
        for strategy in ("standard", "safe", "conservative"):
            for flags in ([], ["-0"], ["-b", "ops"], ["-b", "ops-r"], ["-b", "ges"], ["-b", "ges-r"], ["-b", "ges-t"], ["-b", "lges0"], ["-b", "lges"], ["-b", "lges+"]):
                with self.subTest(strategy=strategy, flags=flags):
                    self.run_cli("--insert", strategy, *flags)

    def test_strict_lges_dispatch(self):
        for mode in ("lges0", "lges", "lges+"):
            for strategy in ("safe", "conservative"):
                _, stats = self.run_cli("-b", mode, "--insert", strategy, "--strict-lges")
                self.assertEqual(stats["config-strict_lges"], 1)
        self.run_cli("--strict-lges", ok=False)
        self.run_cli("--strict-lges", "-b", "ges-t", ok=False)

    def test_layout_endianness_versions_and_cache(self):
        expected_graph, expected_stats = self.run_cli("-0")
        for version in (1, 2, 3):
            for fortran in (False, True):
                for endian in ("<", ">", "="):
                    with self.subTest(version=version, fortran=fortran, endian=endian):
                        values = ([row[j] for j in range(3) for row in self.data] if fortran
                                  else [x for row in self.data for x in row])
                        self.input.write_bytes(npy_bytes((80, 3), values, endian + "f8", fortran, version))
                        graph, stats = self.run_cli("-0", "--cache-size", "2")
                        self.assertEqual(graph, expected_graph)
                        self.assertAlmostEqual(stats["score"], expected_stats["score"], places=8)
                        self.assertLessEqual(stats.get("local_score-cache-peak-entries", 0), 2)
        graph, _ = self.run_cli("-0", "--cache-size", "0")
        self.assertEqual(graph, expected_graph)

    def test_single_column_debug(self):
        self.input.write_bytes(npy_bytes((3, 1), [1, 2, 4]))
        graph, _ = self.run_cli("-v2")
        self.assertEqual(graph.strip(), "0\n0")

    def test_bad_files(self):
        cases = [b"", b"\x93NUMPY\x01\x00", b"invalid input",
                 npy_bytes((3,), [1, 2, 3]), npy_bytes((2, 2, 1), [1, 2, 3, 4]),
                 npy_bytes((0, 3), []), npy_bytes((3, 0), []), npy_bytes((1, 3), [1, 2, 3]),
                 npy_bytes((2, 2), [1, 2, 3, 4], "<f4"), npy_bytes((2, 2), [1, 2, 3, 4], "<i8"),
                 npy_bytes((2, 2), [1, 2, 3, 4])[:-1], npy_bytes((2, 2), [1, 2, 3, 4]) + b"x",
                 npy_bytes((2, 2), [], header="{'descr':'<f8','fortran_order':False,'shape':(-2,2)}"),
                 npy_bytes((2, 2), [], header="{'descr':'<f8','fortran_order':False,'shape':(9999999999999999999999999,2)}"),
                 npy_bytes((2, 2), [], header="{'descr':'<f8','descr':'<f8','fortran_order':False,'shape':(2,2)}"),
                 npy_bytes((2, 2), [], header="{'descr':'<f8','shape':(2,2)}"),
                 npy_bytes((2, 2), [], header="{'descr':'<f8','fortran_order':False,'shape':(2,2)} junk")]
        for i, contents in enumerate(cases):
            with self.subTest(case=i):
                self.input.write_bytes(contents)
                self.run_cli(ok=False)
        self.run_cli(ok=False, input_path=self.root / "missing.npy")

    def test_bad_numerics_and_options(self):
        original = self.input.read_bytes()
        for special in (math.nan, math.inf, -math.inf):
            values = [x for row in self.data for x in row]
            values[0] = special
            self.input.write_bytes(npy_bytes((80, 3), values))
            for mode in ([], ["-b", "ops"]):
                self.run_cli("--insert", "conservative", *mode, ok=False)
        self.input.write_bytes(npy_bytes((3, 2), [1, 1, 1, 2, 1, 3]))
        self.run_cli(ok=False)  # constant column
        self.input.write_bytes(npy_bytes((3, 2), [1, 1, 2, 2, 3, 3]))
        self.run_cli(ok=False)  # deterministic dependence
        self.input.write_bytes(original)
        for flags in (["--alpha", "-1"], ["--alpha", "0"], ["--alpha", "nan"], ["--alpha", "inf"],
                      ["--insert", "bad"], ["--baseline", "bad"], ["-v4"], ["--cache-size", "-1"],
                      ["-0", "--baseline", "ges"], ["-0", "--baseline", "lges"],
                      ["-0", "--baseline", "lges+"], ["--unknown"]):
            with self.subTest(flags=flags):
                self.run_cli(*flags, ok=False)

    def test_output_failures_and_path_collisions(self):
        self.run_cli("--output", str(self.root / "missing" / "graph.csv"), ok=False)
        self.run_cli("--stats", str(self.root / "missing" / "stats.csv"), ok=False)
        self.run_cli("--output", str(self.root), ok=False)
        original = self.input.read_bytes()
        self.run_cli("--output", str(self.input), ok=False)
        self.assertEqual(self.input.read_bytes(), original)
        self.run_cli("--stats", str(self.output), ok=False)

    def test_ground_truth(self):
        graph, _ = self.run_cli("-0")
        truth = self.root / "truth.csv"
        truth.write_text(graph)
        _, stats = self.run_cli("-0", "--graph_truth", str(truth))
        self.assertEqual(stats["shd"], 0)
        # A chain DAG completes to the same undirected chain as this easy data.
        truth.write_text("0,1,2\n0,1,0\n0,0,1\n0,0,0\n")
        self.run_cli("--graph_truth", str(truth))
        truth.write_text("0,1\n0,1\n0,0\n")
        self.run_cli("--graph_truth", str(truth), ok=False)
        truth.write_text("0,1,2\n0,1,0\n0,0,1\n1,0,0\n")
        self.run_cli("--graph_truth", str(truth), ok=False)

    def test_help_and_missing_input(self):
        help_result = subprocess.run([BINARY, "--help"], capture_output=True, text=True, timeout=10)
        self.assertEqual(help_result.returncode, 0)
        missing = subprocess.run([BINARY], capture_output=True, text=True, timeout=10)
        self.assertEqual(missing.returncode, 2)
        self.assertIn("--input is required", missing.stderr)


if __name__ == "__main__":
    unittest.main()
