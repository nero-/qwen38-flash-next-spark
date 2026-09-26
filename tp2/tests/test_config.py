import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import active_requests, find_ipv4_gid, load_config, peer_command, selected_rails


ROOT = Path(__file__).resolve().parents[1]


class DeploymentConfigTests(unittest.TestCase):
    def test_default_and_both_cable_maps(self):
        config = load_config(ROOT)
        self.assertEqual(config["default_profile"], "hc-adaptive+cg4+m5500h")
        self.assertEqual(config["kv_cache_gib"], 24)
        self.assertEqual(config["default_cables"], 2)
        self.assertEqual(
            [rail["hca"] for rail in selected_rails(config, 0, 2)],
            ["rocep1s0f1", "roceP2p1s0f0"],
        )
        self.assertEqual(
            [rail["interface"] for rail in selected_rails(config, 0, 1)],
            ["enp1s0f1np1", "enP2p1s0f1np1"],
        )

    def test_peer_command_keeps_shell_argument_boundaries(self):
        config = load_config(ROOT)
        command = peer_command(config, ["python3", "/tmp/path with space/launch.py", "check", "--rank", "1"])
        self.assertEqual(command[:4], ["ssh", "-o", "BatchMode=yes", "-o"])
        self.assertIn("'/tmp/path with space/launch.py'", command[-1])
        self.assertEqual(command[-2], "10.100.168.1")

    def test_ipv4_mapped_gid_selects_roce_v2_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            port = Path(tmp) / "ports/1"
            ndevs = port / "gid_attrs/ndevs"
            types = port / "gid_attrs/types"
            ndevs.mkdir(parents=True)
            types.mkdir(parents=True)
            for index, kind in (("0", "RoCE v1"), ("1", "RoCE v2")):
                (ndevs / index).write_text("dac0\n")
                (types / index).write_text(kind + "\n")
                (port / "gids").mkdir(exist_ok=True)
                (port / "gids" / index).write_text("::ffff:192.0.2.4\n")
            self.assertEqual(find_ipv4_gid(Path(tmp), "dac0"), 1)

    def test_unreadable_gid_slots_are_skipped(self):
        # Unpopulated sysfs GID slots raise EINVAL on read; discovery must skip them.
        with tempfile.TemporaryDirectory() as tmp:
            port = Path(tmp) / "ports/1"
            for sub in ("gid_attrs/ndevs", "gid_attrs/types", "gids"):
                (port / sub).mkdir(parents=True)
            (port / "gid_attrs/ndevs/2").mkdir()  # reading a directory raises OSError
            (port / "gid_attrs/ndevs/3").write_text("dac0\n")
            (port / "gid_attrs/types/3").write_text("RoCE v2\n")
            (port / "gids/3").write_text("::ffff:192.0.2.4\n")
            self.assertEqual(find_ipv4_gid(Path(tmp), "dac0"), 3)

    def test_request_metrics_gate_switching(self):
        self.assertFalse(active_requests('vllm:num_requests_running{engine="0"} 0\n'))
        self.assertTrue(active_requests('vllm:num_requests_waiting{engine="0"} 1\n'))
        with self.assertRaises(ValueError):
            active_requests('vllm:num_requests_running{engine="0"} busy\n')

    def test_mounted_source_files_match_release_manifest(self):
        manifest = json.loads((ROOT / "optimization/candidate-sha256.json").read_text())
        for name, expected in manifest.items():
            path = ROOT / "optimization" / name
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), expected, name)


if __name__ == "__main__":
    unittest.main()
