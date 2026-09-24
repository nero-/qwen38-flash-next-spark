#!/usr/bin/env bash
# Run with sudo on each Spark, after stopping the TP2 service.
set -euo pipefail
[[ $EUID == 0 ]] || { echo 'Run this script with sudo.' >&2; exit 1; }
case "$(hostname -s)" in
  spark-r0) ifaces=(enp1s0f1np1 enP2p1s0f1np1) ;;
  spark-r1) ifaces=(enp1s0f0np0 enP2p1s0f0np0) ;;
  *) echo 'Unexpected hostname; refusing to change network settings.' >&2; exit 1 ;;
esac
for iface in "${ifaces[@]}"; do
  ip link show dev "$iface" >/dev/null
  config=$(netplan get "network.ethernets.$iface")
  [[ -n "$config" && "$config" != null ]] || { echo "Missing netplan ID: $iface" >&2; exit 1; }
done
file=/etc/netplan/99-z-qwen-dac-mtu.yaml
backup=/var/lib/qwen-tp2/mtu-$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$backup"
ip -j address show > "$backup/addresses-before.json"
netplan get > "$backup/netplan-before.yaml"
chmod 600 "$backup"/*
[[ ! -e "$file" ]] || cp -a "$file" "$backup/previous-override.yaml"
install -m 600 /dev/null "$file"
{
  echo 'network:'
  echo '  version: 2'
  echo '  ethernets:'
  for iface in "${ifaces[@]}"; do
    printf '    %s:\n      mtu: 9000\n' "$iface"
  done
} > "$file"
if ! netplan generate; then
  if [[ -e "$backup/previous-override.yaml" ]]; then cp -a "$backup/previous-override.yaml" "$file"; else rm "$file"; fi
  netplan generate
  echo 'Generation failed; restored prior configuration.' >&2; exit 1
fi
# Reload generated profiles without bouncing management or DAC connections.
nmcli connection reload
for iface in "${ifaces[@]}"; do
  ip link set dev "$iface" mtu 9000
  [[ $(cat "/sys/class/net/$iface/mtu") == 9000 ]]
  ip -br address show dev "$iface"
done
echo "MTU 9000 applied to DAC interfaces only. Backup: $backup"
echo 'Once both Sparks are updated, verify each link with ping -M do -s 8972.'
