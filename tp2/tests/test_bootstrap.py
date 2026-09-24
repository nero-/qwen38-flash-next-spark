import os
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


class BootstrapDryRunTests(unittest.TestCase):
    def test_bootstrap_config_and_preflight_order_with_mocked_ssh(self):
        with tempfile.TemporaryDirectory() as temporary:
            temp = Path(temporary)
            log = temp / "calls.txt"
            ssh = temp / "ssh"
            ssh.write_text(
                "#!/usr/bin/env python3\n"
                "import os,sys\n"
                "args=sys.argv[1:]\n"
                "with open(os.environ['BOOTSTRAP_MOCK_LOG'],'a') as f: f.write('ssh ' + ' '.join(args) + '\\n')\n"
                "cmd=' '.join(args)\n"
                "if 'docker inspect -f' in cmd: print('false')\n"
                "elif 'model.sha256.json' in cmd and 'cat >' not in cmd: print('{}')\n"
                "elif 'cat >' in cmd: sys.stdin.read()\n"
            )
            rsync = temp / "rsync"
            rsync.write_text(
                "#!/usr/bin/env python3\n"
                "import os,sys\n"
                "with open(os.environ['BOOTSTRAP_MOCK_LOG'],'a') as f: f.write('rsync ' + ' '.join(sys.argv[1:]) + '\\n')\n"
            )
            ssh.chmod(0o755)
            rsync.chmod(0o755)
            env = dict(os.environ, PATH=f"{temp}:{os.environ['PATH']}", BOOTSTRAP_MOCK_LOG=str(log))
            result = subprocess.run(["bash", str(ROOT / "tp2/bootstrap.sh")], cwd=ROOT, env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            calls = log.read_text().splitlines()
            preflight = [i for i, call in enumerate(calls) if "docker info" in call]
            self.assertEqual(len(preflight), 2)
            first_copy = next(i for i, call in enumerate(calls) if call.startswith("rsync "))
            self.assertTrue(all(i < first_copy for i in preflight))
            self.assertTrue(any("spark-r0" in call and "docker pull" in call for call in calls))
            self.assertTrue(any("spark-r1" in call and "docker pull" in call for call in calls))
            self.assertTrue(any("peer_ssh" not in call and "model_manifest.py check" in call for call in calls))
            self.assertFalse(any("--delete" in call for call in calls))


if __name__ == "__main__":
    unittest.main()
