#!/usr/bin/env python3
"""Regression tests for retrieve.py, focused on the fallback paths.

Run with:
    python3 -m unittest discover -s tests -v

Uses only the standard library so it runs on any interpreter that can run
the tool itself. Tests that need the real corpus or a live Ollama skip
cleanly when those are unavailable, so the suite is safe to run anywhere.
"""

import inspect
import os
import subprocess
import sys
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import embed  # noqa: E402
import retrieve  # noqa: E402

DB = os.path.join(REPO, "memory.db")
HAS_DB = os.path.exists(DB)
QUERY = "python environment cleanup"


def ollama_up():
    """True if the embedding service answers, so semantic tests can run."""
    try:
        embed.get_embedding("ping")
        return True
    except Exception:
        return False


HAS_OLLAMA = ollama_up()

needs_db = unittest.skipUnless(HAS_DB, f"corpus not present at {DB}")
needs_ollama = unittest.skipUnless(HAS_OLLAMA, "ollama not reachable")


class TestPublicApi(unittest.TestCase):
    """browse.py imports these; their call signatures must stay compatible."""

    def test_retrieve_accepts_legacy_kwargs(self):
        params = inspect.signature(retrieve.retrieve).parameters
        for name in ("query", "db_path", "top_k", "window", "depth", "budget",
                     "token_budget", "session_id", "roles", "no_tools",
                     "no_vectors", "project"):
            self.assertIn(name, params)

    def test_new_diag_param_is_optional(self):
        self.assertIs(inspect.signature(retrieve.retrieve)
                      .parameters["diag"].default, None)
        self.assertIs(inspect.signature(retrieve.open_db)
                      .parameters["diag"].default, None)


class TestDiagnostics(unittest.TestCase):
    def setUp(self):
        self.diag = retrieve.RetrievalDiagnostics()

    def test_warnings_are_deduplicated(self):
        self.diag.warn("same")
        self.diag.warn("same")
        self.diag.warn("different")
        self.assertEqual(self.diag.warnings, ["same", "different"])

    def test_brute_force_is_suboptimal_not_degraded(self):
        """Slow path still returns complete results; that is not degradation."""
        self.diag.vector_mode = "brute-force"
        self.assertFalse(self.diag.degraded)
        self.assertTrue(self.diag.suboptimal)

    def test_unavailable_is_degraded(self):
        self.diag.vector_mode = "unavailable"
        self.assertTrue(self.diag.degraded)
        self.assertFalse(self.diag.suboptimal)

    def test_disabled_on_purpose_is_not_degraded(self):
        self.diag.vector_mode = "disabled"
        self.assertFalse(self.diag.degraded)

    def test_any_warning_marks_degraded(self):
        self.diag.vector_mode = "knn"
        self.diag.warn("something odd")
        self.assertTrue(self.diag.degraded)

    def test_summary_includes_mode_and_seed_counts(self):
        self.diag.backend = "sqlite3"
        self.diag.vector_mode = "brute-force"
        self.diag.fts_seeds, self.diag.vector_seeds = 5, 3
        joined = " ".join(self.diag.summary_lines())
        self.assertIn("vector_mode=brute-force", joined)
        self.assertIn("fts=5", joined)
        self.assertIn("vector=3", joined)


class TestEmbedFailureClassification(unittest.TestCase):
    """Each failure mode must produce a message that says what to do."""

    def test_import_error_names_the_interpreter(self):
        msg = retrieve._describe_embed_failure(
            ImportError("No module named 'requests'"), embed)
        self.assertIn("dependency missing", msg)
        self.assertIn(sys.executable, msg)

    def test_connection_error_names_the_service(self):
        class ConnectionError(Exception):  # noqa: A001 - mimics requests
            pass

        msg = retrieve._describe_embed_failure(ConnectionError("refused"), embed)
        self.assertIn("unreachable", msg)
        self.assertIn("ollama serve", msg)

    def test_unknown_error_is_passed_through_with_type(self):
        msg = retrieve._describe_embed_failure(ValueError("bad payload"), embed)
        self.assertIn("ValueError", msg)
        self.assertIn("bad payload", msg)


@needs_db
class TestRetrievalFallback(unittest.TestCase):
    """Degradation must never lose results, and must never be silent."""

    def tearDown(self):
        # Undo any monkeypatching so tests stay independent.
        import importlib
        importlib.reload(embed)

    def _run(self, **kwargs):
        diag = retrieve.RetrievalDiagnostics()
        chunks = retrieve.retrieve(QUERY, db_path=DB, diag=diag, **kwargs)
        return chunks, diag

    @needs_ollama
    def test_healthy_path_produces_vector_seeds(self):
        chunks, diag = self._run()
        self.assertTrue(chunks)
        self.assertGreater(diag.vector_seeds, 0)
        self.assertFalse(diag.degraded)

    def test_unreachable_service_still_returns_results(self):
        embed.OLLAMA_URL = "http://localhost:9/api/embeddings"  # discard port
        chunks, diag = self._run()
        self.assertTrue(chunks, "keyword results must survive an embedding outage")
        self.assertEqual(diag.vector_mode, "unavailable")
        self.assertTrue(diag.warnings, "an outage must not be silent")

    def test_missing_dependency_still_returns_results(self):
        def raise_import(_text):
            raise ImportError("No module named 'requests'")

        embed.get_embedding = raise_import
        chunks, diag = self._run()
        self.assertTrue(chunks)
        self.assertIn("dependency missing", diag.vector_reason)

    def test_unexpected_error_is_reported_not_swallowed(self):
        def raise_weird(_text):
            raise ValueError("malformed embedding response")

        embed.get_embedding = raise_weird
        chunks, diag = self._run()
        self.assertTrue(chunks)
        self.assertIn("ValueError", diag.vector_reason)
        self.assertTrue(diag.warnings)

    def test_no_vectors_is_quiet(self):
        chunks, diag = self._run(no_vectors=True)
        self.assertTrue(chunks)
        self.assertEqual(diag.vector_mode, "disabled")
        self.assertFalse(diag.warnings, "opting out is not a fault")

    @needs_ollama
    def test_seed_rows_carry_no_embedding_blob(self):
        """Phase-2 hydration should return chunk rows, not scoring rows."""
        conn = retrieve.open_db(DB)
        try:
            seeds = retrieve.vector_search(conn, QUERY, top_k=3)
        finally:
            conn.close()
        self.assertTrue(seeds)
        for _, row in seeds:
            self.assertNotIn("embedding", row)
            self.assertIn("text", row)
            self.assertIn("seq_index", row)


@needs_db
class TestCli(unittest.TestCase):
    def _cli(self, *args):
        return subprocess.run(
            [sys.executable, os.path.join(REPO, "retrieve.py"),
             QUERY, "--db", DB, *args],
            capture_output=True, text=True, cwd=REPO, timeout=180,
        )

    def test_default_run_succeeds(self):
        proc = self._cli()
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("<session_memory", proc.stdout)

    def test_explain_writes_diagnostics_to_stderr_only(self):
        proc = self._cli("--explain")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("diag:", proc.stderr)
        self.assertNotIn("diag:", proc.stdout,
                         "diagnostics must not pollute the context block")

    def test_json_output_is_valid(self):
        import json
        proc = self._cli("--format", "json")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIsInstance(json.loads(proc.stdout), list)


@needs_db
class TestOutputParity(unittest.TestCase):
    """Refactors of the search path must not change what users receive."""

    @classmethod
    def setUpClass(cls):
        cls.baseline = os.path.join(REPO, "_parity_baseline.py")
        ref = os.environ.get("PARITY_REF", "")
        if not ref:
            raise unittest.SkipTest(
                "set PARITY_REF=<git-ref> to diff output against an older revision")
        result = subprocess.run(["git", "show", f"{ref}:retrieve.py"],
                                capture_output=True, text=True, cwd=REPO)
        if result.returncode != 0:
            raise unittest.SkipTest(f"cannot read retrieve.py at {ref}")
        with open(cls.baseline, "w") as fh:
            fh.write(result.stdout)

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.baseline):
            os.remove(cls.baseline)

    def _output(self, script, *args):
        return subprocess.run(
            [sys.executable, script, QUERY, "--db", DB, *args],
            capture_output=True, text=True, cwd=REPO, timeout=180,
        ).stdout

    def test_context_output_matches_baseline(self):
        self.assertEqual(self._output(self.baseline),
                         self._output(os.path.join(REPO, "retrieve.py")))

    def test_no_vectors_output_matches_baseline(self):
        self.assertEqual(self._output(self.baseline, "--no-vectors"),
                         self._output(os.path.join(REPO, "retrieve.py"),
                                      "--no-vectors"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
