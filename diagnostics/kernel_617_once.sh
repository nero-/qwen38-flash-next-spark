#!/usr/bin/env bash
# Install the signed 6.17 kernel alongside 7.0, then select ONE boot into it.
# Run only after the current benchmark finishes. Requires interactive sudo.
set -euo pipefail
export LC_ALL=C
[[ $EUID == 0 ]] || { echo 'Run with sudo.' >&2; exit 1; }
target=6.17.0-1032-nvidia
expected_current=7.0.0-1019-nvidia
[[ $(uname -r) == "$expected_current" ]] || { echo 'Unexpected running kernel; stop and review.' >&2; exit 1; }
[[ $(cat /sys/module/nvidia/version) == 580.178.04 ]] || { echo 'Unexpected NVIDIA driver; stop and review.' >&2; exit 1; }
[[ -e /boot/vmlinuz-$expected_current ]] || exit 1
backup="/var/lib/qwen38-kernel-ab/$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$backup"
cp -a /etc/default/grub /etc/default/grub.d /boot/grub/grub.cfg /boot/grub/grubenv "$backup/"
uname -a > "$backup/uname-before.txt"
cat /proc/cmdline > "$backup/cmdline-before.txt"
dpkg-query -W > "$backup/packages-before.txt"
grub-editenv /boot/grub/grubenv list > "$backup/grubenv-before.txt"
if grep -q '^next_entry=.' "$backup/grubenv-before.txt"; then
  echo 'A one-time boot is already scheduled; stop and review.' >&2
  exit 1
fi
packages=(
  linux-image-6.17.0-1032-nvidia=6.17.0-1032.32
  linux-headers-6.17.0-1032-nvidia=6.17.0-1032.32
  linux-modules-6.17.0-1032-nvidia=6.17.0-1032.32
  linux-nvidia-6.17-headers-6.17.0-1032=6.17.0-1032.32
  linux-modules-nvidia-580-open-6.17.0-1032-nvidia=6.17.0-1032.32+1
)
apt-get --simulate --no-remove install "${packages[@]}" > "$backup/apt-plan.txt"
cat "$backup/apt-plan.txt"
/home/nero/venvs/qwen38/bin/python - "$backup/apt-plan.txt" <<'PY'
import pathlib, sys
allowed = {
    'linux-image-6.17.0-1032-nvidia', 'linux-headers-6.17.0-1032-nvidia',
    'linux-modules-6.17.0-1032-nvidia', 'linux-nvidia-6.17-headers-6.17.0-1032',
    'linux-modules-nvidia-580-open-6.17.0-1032-nvidia'}
for line in pathlib.Path(sys.argv[1]).read_text().splitlines():
    if line.startswith('Remv ') or (line.startswith('Inst ') and line.split()[1] not in allowed):
        raise SystemExit('Unexpected package change: ' + line)
PY
apt-get -y --no-remove install "${packages[@]}"
[[ $(modinfo -k "$target" -F version nvidia) == 580.178.04 ]] || { echo 'Driver mismatch; not rebooting.' >&2; exit 1; }
[[ -n $(modinfo -k "$target" -F signer nvidia) ]] || { echo 'Unsigned driver; not rebooting.' >&2; exit 1; }
[[ -s /boot/initrd.img-$target && -s /boot/vmlinuz-$target ]] || exit 1
update-grub
/home/nero/venvs/qwen38/bin/python - "$target" "$backup/boot-entry.txt" <<'PY'
import pathlib, re, sys
text = pathlib.Path('/boot/grub/grub.cfg').read_text()
if 'set default="${next_entry}"' not in text:
    raise SystemExit('GRUB does not expose the expected one-time boot support')
submenus = re.findall(r"^submenu 'Advanced options[^']*'.*?\$menuentry_id_option '([^']+)'", text, re.M)
entries = re.findall(r"^\s*menuentry '[^']*, with Linux " + re.escape(sys.argv[1]) + r"'.*?\$menuentry_id_option '([^']+)'", text, re.M)
if len(submenus) != 1 or len(entries) != 1:
    raise SystemExit(f'Ambiguous boot entries: {submenus!r}, {entries!r}')
pathlib.Path(sys.argv[2]).write_text(submenus[0] + '>' + entries[0] + '\n')
PY
# Refuse to reboot over in-flight model requests; the helper stops only this model.
if curl --fail --silent --max-time 5 http://127.0.0.1:8000/health >/dev/null; then
  runuser -u nero -- /home/nero/venvs/qwen38/bin/python /home/nero/projects/qwen38-flash-next-spark/diagnostics/tp1_session.py stop
else
  if ss -H -ltn 'sport = :8000' | grep -q .; then
    echo 'API port is occupied but unhealthy; not rebooting.' >&2
    exit 1
  fi
fi
entry=$(cat "$backup/boot-entry.txt")
grub-reboot "$entry"
grub-editenv /boot/grub/grubenv list | grep -Fx "next_entry=$entry"
printf 'Backup: %s\nOne-time kernel: %s\nPermanent default and kernel 7.0 remain available. Rebooting now.\n' "$backup" "$target"
sync
systemctl reboot
