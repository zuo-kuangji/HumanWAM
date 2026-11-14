"""Exercise the real orchestration CLI/config/IO with no model dependencies."""

import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
import warnings
import zipfile
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

import humanwam
from humanwam import cli, registry
from humanwam.config import load_config
from humanwam.data import audit_metadata
from humanwam.observations import observation_from_npz


class CoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        (self.root / "configs").mkdir()
        self.environment = mock.patch.dict(os.environ, {"HUMANWAM_ROOT": str(self.root)})
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def write_config(self, value, name="configs/test.json"):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        return path

    def write_dataset(self, info=None, episodes=None):
        directory = self.root / "dataset"
        (directory / "meta").mkdir(parents=True, exist_ok=True)
        if info is None:
            info = {"fps": 30, "total_frames": 90, "total_episodes": 2}
        if episodes is None:
            episodes = [{"episode_index": 0, "length": 30}, {"episode_index": 1, "length": 60}]
        (directory / "meta/info.json").write_text(json.dumps(info))
        (directory / "meta/episodes.jsonl").write_text(
            "\n".join(json.dumps(row) for row in episodes) + "\n"
        )
        return directory

    def test_actual_cli_list_never_imports_model_runtime(self):
        package_src = str(Path(humanwam.__file__).resolve().parent.parent)
        code = """
import importlib.abc, json, sys
class RejectHeavy(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch','jax','flax','transformers','gr00t','openpi','imagewam','DiT4DiT'}:
            raise AssertionError('list imported a model runtime: ' + fullname)
        return None
sys.meta_path.insert(0, RejectHeavy())
from humanwam.cli import main
raise SystemExit(main(['list']))
"""
        env = dict(os.environ, PYTHONPATH=package_src)
        result = subprocess.run(
            [sys.executable, "-c", code],
            check=False,
            text=True,
            capture_output=True,
            env=env,
            cwd=self.root,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            set(json.loads(result.stdout)),
            {"groot_n17", "pi05", "imagewam_delta", "imagewam_joint_reference", "dit4dit", "psi0", "fastwam", "fastwam_sonic"},
        )

    def test_config_rejects_unresolved_environment_at_any_depth(self):
        name = "HUMANWAM_TEST_MISSING_VARIABLE_9A73"
        with mock.patch.dict(os.environ):
            os.environ.pop(name, None)
            path = self.write_config(
                {
                    "version": 1,
                    "backend": "pi05",
                    "options": {"checkpoint_dir": "${" + name + "}/model"},
                }
            )
            with self.assertRaisesRegex(ValueError, "Unset environment"):
                load_config(path)

    def test_source_dir_is_relative_to_repo_not_caller_or_config_folder(self):
        path = self.write_config(
            {
                "version": 1,
                "backend": "groot_n17",
                "source_dir": "third_party/Isaac-GR00T",
                "options": {"model_path": "nvidia/GR00T-N1.7-3B"},
            },
            "configs/deep/portable.json",
        )
        result = load_config(path)
        self.assertEqual(Path(result["source_dir"]), self.root / "third_party/Isaac-GR00T")
        self.assertEqual(result["options"]["model_path"], "nvidia/GR00T-N1.7-3B")
        self.assertEqual(result["_config_path"], str(path.resolve()))

    def test_cli_native_dry_run_routes_without_spawning(self):
        path = self.write_config(
            {
                "version": 1,
                "backend": "pi05",
                "python": "/pi05 env/python",
                "source_dir": "third_party/openpi",
                "options": {"config_name": "pi05_libero", "exp_name": "contains spaces"},
            }
        )
        plan = {
            "argv": ["/pi05 env/python", "scripts/train.py", "contains spaces"],
            "cwd": "/native project",
            "env": {"SAFE": "yes"},
        }
        builder = mock.Mock(return_value=plan)
        module = SimpleNamespace(build_train_command=builder)
        buffer = io.StringIO()
        with (
            mock.patch.object(cli.subprocess, "call") as run,
            mock.patch.object(registry.importlib, "import_module", return_value=module) as imported,
            redirect_stdout(buffer),
        ):
            code = cli.main(["train", "--config", str(path), "--dry-run"])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(buffer.getvalue()), plan)
        run.assert_not_called()
        imported.assert_called_once_with("humanwam.backends.pi05_commands")
        options, source = builder.call_args.args
        self.assertEqual(options["python"], "/pi05 env/python")
        self.assertEqual(options["exp_name"], "contains spaces")
        self.assertEqual(Path(source), self.root / "third_party/openpi")

    def test_npz_native_groot_groups_and_json_language(self):
        file = self.root / "native.npz"
        video = np.zeros((1, 1, 4, 5, 3), dtype=np.uint8)
        state = np.zeros((1, 1, 6), dtype=np.float32)
        np.savez(file, **{"video.ego_view": video, "state.left_leg": state})
        language = {"annotation.human.task_description": [["pick up the object"]]}
        result = observation_from_npz(file, {"language": language})
        self.assertEqual(set(result), {"video", "state", "language"})
        np.testing.assert_array_equal(result["video"]["ego_view"], video)
        np.testing.assert_array_equal(result["state"]["left_leg"], state)
        self.assertEqual(result["language"], language)

    def test_npz_duplicate_metadata_and_group_collisions_rejected(self):
        file = self.root / "conflict.npz"
        np.savez(file, prompt=np.asarray("first"))
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            observation_from_npz(file, {"prompt": "second"})
        for keys in [
            [("state", np.zeros(2)), ("state.left_leg", np.zeros(6))],
            [("state.left_leg", np.zeros(6)), ("state", np.zeros(2))],
        ]:
            np.savez(file, **dict(keys))
            with self.assertRaises(ValueError):
                observation_from_npz(file)

    def test_npz_duplicate_zip_entry_and_object_pickle_rejected(self):
        file = self.root / "duplicate.npz"
        buffer = io.BytesIO()
        np.save(buffer, np.zeros(6))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            with zipfile.ZipFile(file, "w") as archive:
                archive.writestr("state.left_leg.npy", buffer.getvalue())
                archive.writestr("state.left_leg.npy", buffer.getvalue())
        with self.assertRaises(ValueError):
            observation_from_npz(file)
        np.savez(file, dangerous=np.array([{"not": "an array contract"}], dtype=object))
        with self.assertRaises(ValueError):
            observation_from_npz(file)

    def test_metadata_duration_audit_is_read_only_and_explicit(self):
        directory = self.write_dataset()
        files = sorted((directory / "meta").iterdir())
        before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
        result = audit_metadata(directory)
        self.assertEqual((result["episodes"], result["frames"]), (2, 90))
        self.assertEqual(result["min_seconds"], 1.0)
        self.assertEqual(result["max_seconds"], 2.0)
        self.assertEqual(result["mean_seconds"], 1.5)
        self.assertFalse(result["source_modified"])
        self.assertFalse(result["raw_numeric_video_integrity_checked"])
        self.assertEqual(
            before, {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
        )

    def test_metadata_nonfinite_or_nonpositive_fps_rejected(self):
        for fps in [float("nan"), float("inf"), 0, -30, True]:
            with self.subTest(fps=fps):
                directory = self.write_dataset(
                    info={"fps": fps, "total_frames": 90, "total_episodes": 2}
                )
                with self.assertRaises(ValueError):
                    audit_metadata(directory)

    def test_metadata_counts_and_episode_indices_require_exact_integers(self):
        for field, value in [
            ("total_frames", 90.0),
            ("total_episodes", 2.0),
            ("total_frames", True),
        ]:
            info = {"fps": 30, "total_frames": 90, "total_episodes": 2, field: value}
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                audit_metadata(self.write_dataset(info=info))
        for field, value in [
            ("episode_index", 0.0),
            ("episode_index", False),
            ("episode_index", -1),
            ("length", 30.0),
            ("length", True),
        ]:
            episodes = [{"episode_index": 0, "length": 30}, {"episode_index": 1, "length": 60}]
            episodes[0][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                audit_metadata(self.write_dataset(episodes=episodes))


if __name__ == "__main__":
    unittest.main()
