#!/usr/bin/env bash
# Configure this machine (the visual_task Pi) as the NTP server for the
# direct-link setup described in mx_eye's AGENTS.md, "Clock synchronization
# (direct link)": the consumer device (this machine) is the NTP server, the
# mx_eye host is the chrony client (see mx_eye's
# scripts/install-chrony-client.sh, which this is the server-side
# counterpart to - that script configures the mx-eye Pi to sync FROM this
# machine; this script configures this machine to actually serve time).
set -euo pipefail

CONF=/etc/chrony/chrony.conf
USAGE='usage: sudo install-chrony-server.sh [--dry-run] <ALLOWED-CLIENT-OR-SUBNET>'
MARKER='# Managed by visual_task (mirrors mx_eye install-chrony-client.sh)'

DRY_RUN=0
if [ "${1:-}" = --dry-run ]; then
    DRY_RUN=1
    shift
fi

if [ $# -ne 1 ]; then
    printf '%s\n' "$USAGE" >&2
    exit 2
fi
ALLOWED="$1"

# The argument becomes a line of chrony configuration: accept an IPv4/IPv6
# address, a CIDR subnet (e.g. 169.254.0.0/16), or a hostname - nothing else.
case "$ALLOWED" in
-*)
    printf 'error: unknown option %s\n%s\n' "$ALLOWED" "$USAGE" >&2
    exit 2
    ;;
*[!A-Za-z0-9.:/_-]*)
    printf 'error: %s is not an IPv4/IPv6 address, CIDR subnet, or hostname\n' "$ALLOWED" >&2
    exit 2
    ;;
esac

config() {
    cat <<EOF
$MARKER: NTP server for the direct-link mx-eye client.
# Re-run install-chrony-server.sh after editing; local edits are lost.
# This machine has no battery-backed RTC, so its own clock is wrong on
# every boot (often reset to 1970) until something corrects it. If this
# machine ALSO has a separate internet-connected interface (e.g. wlan0,
# alongside the isolated direct-link eth0 to the mx-eye Pi), these pool
# servers let chrony correct this machine's own clock from the real
# internet - then everything downstream (this machine's own clock, and
# whatever it serves to the mx-eye client over the direct link) is
# actually accurate, not just internally self-consistent.
pool 2.debian.pool.ntp.org iburst
pool time.cloudflare.com iburst
# Fallback only: used when none of the pool servers above are reachable
# (e.g. no internet, or the direct link is the only interface up right
# now) so the mx-eye client still has *something* to sync to rather than
# nothing at all. chrony automatically prefers a real, reachable pool
# server over this the moment one is available again - no extra
# configuration needed for that preference.
local stratum 10
allow $ALLOWED
# "allow" governs NTP time-serving only. chronyc's own monitoring
# commands (e.g. "chronyc clients", used by check-chrony-server.sh) are a
# separate subsystem: cmdallow controls which NETWORK SOURCES may reach
# the command socket at all (127.0.0.1 here = this machine only), but
# "chronyc clients" additionally requires the calling process itself to
# be root - run it with sudo even when cmdallow is already correct, or it
# fails with "501 Not authorised" despite chrony otherwise working fine
# (NTP time-serving and "chronyc tracking" are unaffected either way).
cmdallow 127.0.0.1
driftfile /var/lib/chrony/chrony.drift
makestep 1.0 3
rtcsync
EOF
}

if [ "$DRY_RUN" -eq 1 ]; then
    printf 'would write %s:\n\n' "$CONF"
    config
    exit 0
fi

if [ "$(id -u)" -ne 0 ]; then
    printf 'error: run this script as root, for example with sudo\n' >&2
    exit 1
fi

if ! command -v chronyd >/dev/null 2>&1; then
    if ! command -v apt-get >/dev/null 2>&1; then
        printf 'error: install chrony first; this script uses apt-get\n' >&2
        exit 1
    fi
    printf 'installing chrony...\n'
    apt-get install -y --no-install-recommends chrony
fi

# chrony must be the only time daemon on this host.
systemctl disable --now systemd-timesyncd 2>/dev/null || true

if [ -f "$CONF" ] && ! grep -qF "$MARKER" "$CONF"; then
    cp -a "$CONF" "$CONF.bak"
    printf 'kept the previous configuration at %s.bak\n' "$CONF"
fi

config >"$CONF"
printf 'wrote %s:\n\n' "$CONF"
config

systemctl enable chrony
systemctl restart chrony
printf '\nchrony restarted; this machine now serves time to %s.\n' "$ALLOWED"
printf 'now run scripts/install-chrony-client.sh on the mx-eye Pi, pointed at this machine'\''s IP.\n'
