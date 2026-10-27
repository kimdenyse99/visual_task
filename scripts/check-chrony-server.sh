#!/usr/bin/env bash
# Verify this machine (the visual_task Pi) is actually serving NTP time.
# Run scripts/install-chrony-server.sh first.
set -euo pipefail

printf 'chrony service status:\n'
systemctl is-active chrony || {
    printf 'error: chrony is not running - run install-chrony-server.sh first\n' >&2
    exit 1
}

printf '\ntracking (this machine'\''s own clock state):\n'
chronyc -n tracking

printf '\nclients served so far (look for the mx-eye Pi'\''s IP once it has polled):\n'
chronyc -n clients

printf '\nIf the mx-eye Pi'\''s IP never appears above after running its own\n'
printf 'check-chrony-client.sh, double check: the IP/subnet passed to\n'
printf '"allow" during install-chrony-server.sh actually covers the mx-eye\n'
printf "Pi's address, and no firewall on this machine blocks inbound UDP port 123.\n"
