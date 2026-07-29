#!/usr/bin/env bash

set -euo pipefail

channel_id="${1:-0}"
target_user="${SUDO_USER:-$(id -un)}"
target_group="$(id -gn "${target_user}")"
device_dir="/dev/nvidia-caps-imex-channels"
device_path="${device_dir}/channel${channel_id}"

if [[ ! "${channel_id}" =~ ^[0-9]+$ ]]; then
  echo "Channel id must be a non-negative integer: ${channel_id}" >&2
  exit 2
fi

major="$(awk '$2 == "nvidia-caps-imex-channels" { print $1 }' /proc/devices)"
if [[ -z "${major}" ]]; then
  echo "The NVIDIA driver did not register nvidia-caps-imex-channels." >&2
  exit 1
fi

sudo install -d -m 0755 "${device_dir}"
if [[ -e "${device_path}" && ! -c "${device_path}" ]]; then
  echo "Refusing to replace non-device path: ${device_path}" >&2
  exit 1
fi
if [[ ! -c "${device_path}" ]]; then
  sudo mknod "${device_path}" c "${major}" "${channel_id}"
fi

sudo chown "${target_user}:${target_group}" "${device_path}"
sudo chmod 0600 "${device_path}"

actual="$(stat -c '%t:%T' "${device_path}")"
expected="$(printf '%x:%x' "${major}" "${channel_id}")"
if [[ "${actual}" != "${expected}" ]]; then
  echo "Unexpected device number for ${device_path}: ${actual}, expected ${expected}" >&2
  exit 1
fi

echo "Configured ${device_path} for ${target_user}:${target_group} (mode 0600)."
