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
        parse_profile(config["default_profile"], config.get("experimental_images", {}))

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


class CsfAndImageTests(unittest.TestCase):
    IMG = "eugr/spark-vllm-b12x@sha256:" + "a" * 64

    def test_image_modifier_parses_only_when_configured(self):
        self.assertEqual(parse_profile("tp1+cg4+n1007", {"n1007": self.IMG})["image"], "n1007")
        with self.assertRaises(ValueError):
            parse_profile("tp1+n1007")
        with self.assertRaises(ValueError):
            parse_profile("tp1+n1007+n1008", {"n1007": self.IMG, "n1008": self.IMG})

    def test_csf_requires_an_image_modifier(self):
        result = subprocess.run([sys.executable, str(ROOT / "launch.py"), "check", "--profile", "tp1+csf"],
                                capture_output=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(b"NVFP4-CSF reader", result.stderr)

    def test_csf_command_uses_reader_and_mounts_checkpoint(self):
        import config as config_module
        data = load_config(ROOT)
        data["experimental_images"] = {**data["experimental_images"], "n1007": self.IMG}
        config_module.validate_config(data)
        # launch.py reads node-config.json; exercise it through a temporary copy.
        import shutil, tempfile
        with tempfile.TemporaryDirectory() as tmp:
            for name in ("launch.py", "config.py"):
                shutil.copy(ROOT / name, Path(tmp) / name)
            (Path(tmp) / "node-config.json").write_text(json.dumps(data))
            out = subprocess.check_output([sys.executable, str(Path(tmp) / "launch.py"), "check",
                                           "--profile", "tp1+cg4+mxhead+csf+n1007+kv23"])
        cmd = json.loads(out)
        self.assertEqual(arg(cmd, "--quantization"), "nvfp4_csf")
        self.assertEqual(arg(cmd, "--load-format"), "nvfp4_csf")
        self.assertTrue(cmd[cmd.index("serve") + 1].endswith("/serve-csf"))
        self.assertTrue(any(x.endswith("/model-csf,readonly") for x in cmd))
        self.assertIn(self.IMG, cmd)
        self.assertEqual(arg(cmd, "--kv-cache-memory-bytes"), str(23 * 2**30))

    def test_bad_experimental_image_rejected(self):
        import config as config_module
        data = load_config(ROOT)
        configured = data["experimental_images"]
        data["experimental_images"] = {**configured, "kk": "sha256:" + "b" * 64}
        config_module.validate_config(data)  # local image ID is accepted
        for images in ({"cg4": self.IMG}, {"kv24": self.IMG}, {"n1007": "eugr/spark-vllm-b12x:latest"},
                       {"kk": "qwen-tp1-kk:csf"}):
            data["experimental_images"] = {**configured, **images}
            with self.assertRaises(ValueError):
                config_module.validate_config(data)


if __name__ == "__main__":
    unittest.main()
