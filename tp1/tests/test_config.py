import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config import active_requests, load_config, parse_profile


def command(profile):
    out = subprocess.check_output([sys.executable, str(ROOT / "launch.py"), "check", "--profile", profile])
    return json.loads(out)


def arg(cmd, flag):
    return cmd[cmd.index(flag) + 1]


class ConfigTests(unittest.TestCase):
    def test_default_profile_parses(self):
        config = load_config(ROOT)
        parse_profile(config["default_profile"])

    def test_profile_parsing(self):
        parsed = parse_profile("tp1+cg4+kv24+mtp4")
        self.assertEqual(parsed["flags"], {"cg4"})
        self.assertEqual((parsed["kv"], parsed["mtp"], parsed["seqs"]), (24, 4, None))
        for bad in ("tp2", "tp1+cg4+cg4", "tp1+kv0", "tp1+kv99", "tp1+nope", "tp1+kv20+kv22", "tp1;rm"):
            with self.assertRaises(ValueError):
                parse_profile(bad)

    def test_active_requests(self):
        self.assertFalse(active_requests('vllm:num_requests_running{m="x"} 0.0\n'))
        self.assertTrue(active_requests('vllm:num_requests_waiting{m="x"} 2.0\n'))


class LaunchCommandTests(unittest.TestCase):
    def test_base_is_tp1_disk_ple_bf16(self):
        cmd = command("tp1")
        self.assertEqual(arg(cmd, "--tensor-parallel-size"), "1")
        self.assertIn("VLLM_PLE_TABLE_MEMORY=disk", cmd)
        self.assertIn("VLLM_MXFP8_LM_HEAD=0", cmd)
        self.assertEqual(arg(cmd, "--mamba-ssm-cache-dtype"), "bfloat16")
        self.assertEqual(arg(cmd, "--max-cudagraph-capture-size"), "64")
        spec = json.loads(arg(cmd, "--speculative-config"))
        self.assertEqual(spec["num_speculative_tokens"], 3)
        self.assertEqual(spec["rejection_sample_method"], "block")
        self.assertNotIn("cudagraph_capture_sizes", json.loads(arg(cmd, "--compilation-config")))

    def test_modifiers_change_one_axis(self):
        base = command("tp1")
        for profile, expect in {
            "tp1+mxhead": {"VLLM_MXFP8_LM_HEAD=1"},
            "tp1+fp32ssm": {"float32"},
            "tp1+kv24": {str(24 * 2**30)},
        }.items():
            cmd = command(profile)
            # Cache mount differs per profile; ignore it.
            diff = {x for x in cmd if x not in base and not x.startswith("type=bind,src=")}
            self.assertEqual(diff, expect, profile)

    def test_cg4_sizes_follow_depth_and_seqs(self):
        sizes = json.loads(arg(command("tp1+cg4+mtp4+seqs8"), "--compilation-config"))["cudagraph_capture_sizes"]
        self.assertEqual(sizes, [1, 2, *range(5, 41, 5)])

    def test_resident_requires_explicit_kv(self):
        result = subprocess.run([sys.executable, str(ROOT / "launch.py"), "check", "--profile", "tp1+resident"],
                                capture_output=True)
        self.assertNotEqual(result.returncode, 0)
        cmd = command("tp1+resident+kv4")
        self.assertIn("VLLM_PLE_CPU_OFFLOAD=0", cmd)
        self.assertFalse(any(x.startswith("VLLM_PLE_TABLE_MEMORY=") for x in cmd))


if __name__ == "__main__":
    unittest.main()
