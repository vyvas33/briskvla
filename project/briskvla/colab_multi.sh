#!/usr/bin/env bash
# Manage multiple Google Colab accounts in `colab` CLI and mount a shared 5 TB Google Drive
# via rclone onto any account's Colab VM.
#
# Each account gets isolated session state (~/.config/colab-cli/accounts/<acct>/sessions.json)
# and isolated ADC credentials (~/.config/colab-cli/accounts/<acct>/gcloud/), so two accounts
# never clobber each other or your default ~/.config/gcloud setup.
#
# One-time setup:
#   # 1. Authenticate both Colab accounts (each opens a browser login)
#   project/briskvla/colab_multi.sh auth acct5tb
#   project/briskvla/colab_multi.sh auth acct25gb
#
#   # 2. Configure rclone once locally with the 5 TB Drive account (remote name: gdrive5tb)
#   rclone config   # n -> gdrive5tb -> drive -> sign in with the 5 TB account
#
# Typical workflow (running on the 25 GB account's credits, writing to the 5 TB Drive):
#   project/briskvla/colab_multi.sh acct25gb new -s briskvla --gpu A100
#   project/briskvla/colab_multi.sh mount acct25gb briskvla gdrive5tb
#   project/briskvla/colab_multi.sh acct25gb exec -s briskvla --timeout 0 -f project/briskvla/colab_distill.py
#   project/briskvla/colab_multi.sh flush acct25gb briskvla
#   project/briskvla/colab_multi.sh acct25gb stop -s briskvla
set -euo pipefail

ACCOUNTS_ROOT=${COLAB_ACCOUNTS_ROOT:-"$HOME/.config/colab-cli/accounts"}
RCLONE_CONF=${RCLONE_CONF:-"$HOME/.config/rclone/rclone.conf"}
COLAB_SCOPES="openid,https://www.googleapis.com/auth/cloud-platform,https://www.googleapis.com/auth/userinfo.email,https://www.googleapis.com/auth/colaboratory"

usage() {
  cat <<'EOF'
Usage:
  colab_multi.sh auth  <acct>                       Authenticate <acct> via isolated gcloud ADC
  colab_multi.sh mount <acct> <session> [remote]    Upload rclone.conf and mount <remote>: at /content/drive/MyDrive
  colab_multi.sh flush <acct> <session>             Flush & unmount /content/drive/MyDrive before stopping VM
  colab_multi.sh list                               List configured accounts
  colab_multi.sh <acct> <colab args...>             Run any `colab` command as <acct> (e.g. new, sessions, exec, stop)

Environment overrides:
  COLAB_ACCOUNTS_ROOT  Base dir for per-account state (default: ~/.config/colab-cli/accounts)
  RCLONE_CONF          Local rclone config to upload (default: ~/.config/rclone/rclone.conf)
  RCLONE_REMOTE        Default remote name for `mount` (default: gdrive5tb)
EOF
}

require_account_name() {
  local acct=${1:-}
  if [[ -z "$acct" || ! "$acct" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "Invalid account name: '$acct' (use letters, digits, ., _, -)" >&2
    exit 1
  fi
}

run_colab_as() {
  local acct=$1
  shift
  require_account_name "$acct"
  local acct_dir="$ACCOUNTS_ROOT/$acct"
  local gcloud_dir="$acct_dir/gcloud"
  local adc_file="$gcloud_dir/application_default_credentials.json"
  local sessions_file="$acct_dir/sessions.json"

  if [[ ! -f "$adc_file" ]]; then
    echo "Account '$acct' is not authenticated yet. Run: $0 auth $acct" >&2
    exit 1
  fi

  CLOUDSDK_CONFIG="$gcloud_dir" \
    GOOGLE_APPLICATION_CREDENTIALS="$adc_file" \
    colab --auth=adc --config "$sessions_file" "$@"
}

cmd_auth() {
  local acct=${1:?usage: colab_multi.sh auth <acct>}
  require_account_name "$acct"
  local acct_dir="$ACCOUNTS_ROOT/$acct"
  local gcloud_dir="$acct_dir/gcloud"
  local adc_file="$gcloud_dir/application_default_credentials.json"
  local sessions_file="$acct_dir/sessions.json"

  mkdir -p "$gcloud_dir"
  chmod 700 "$acct_dir" "$gcloud_dir"

  echo "Authenticating Colab account '$acct' (isolated ADC in $gcloud_dir)..."
  CLOUDSDK_CONFIG="$gcloud_dir" \
    gcloud auth application-default login --scopes="$COLAB_SCOPES"

  echo "Verifying credentials for '$acct'..."
  CLOUDSDK_CONFIG="$gcloud_dir" \
    GOOGLE_APPLICATION_CREDENTIALS="$adc_file" \
    colab --auth=adc --config "$sessions_file" whoami || true
  echo "Done. Use: $0 $acct sessions"
}

cmd_mount() {
  local acct=${1:?usage: colab_multi.sh mount <acct> <session> [remote]}
  local session=${2:?usage: colab_multi.sh mount <acct> <session> [remote]}
  local remote=${3:-${RCLONE_REMOTE:-gdrive5tb}}

  if [[ ! -f "$RCLONE_CONF" ]]; then
    echo "Missing local rclone config at $RCLONE_CONF." >&2
    echo "Run 'rclone config' locally to add remote '$remote' for your 5 TB Drive account first." >&2
    exit 1
  fi

  echo "[$acct/$session] Uploading $RCLONE_CONF..."
  run_colab_as "$acct" ssh -s "$session" -- "mkdir -p /root/.config/rclone && chmod 700 /root/.config/rclone"
  run_colab_as "$acct" upload -s "$session" "$RCLONE_CONF" /root/.config/rclone/rclone.conf

  echo "[$acct/$session] Mounting ${remote}: at /content/drive/MyDrive..."
  run_colab_as "$acct" ssh -s "$session" -- bash -s -- "$remote" <<'REMOTE_EOF'
set -euo pipefail
REMOTE="$1"
MOUNT_DIR="/content/drive/MyDrive"
chmod 600 /root/.config/rclone/rclone.conf

if mountpoint -q "$MOUNT_DIR" 2>/dev/null; then
  echo "Already mounted at $MOUNT_DIR"
  exit 0
fi

if ! command -v rclone >/dev/null 2>&1 || ! command -v fusermount3 >/dev/null 2>&1; then
  DEBIAN_FRONTEND=noninteractive apt-get -qq update
  DEBIAN_FRONTEND=noninteractive apt-get -qq install -y fuse3 rclone > /dev/null
fi

mkdir -p "$MOUNT_DIR"
rclone mount "${REMOTE}:" "$MOUNT_DIR" \
  --vfs-cache-mode writes \
  --vfs-cache-max-age 10m \
  --vfs-write-back 5s \
  --daemon </dev/null >/tmp/rclone.log 2>&1

for _ in $(seq 1 30); do
  if mountpoint -q "$MOUNT_DIR" 2>/dev/null; then
    echo "Mounted ${REMOTE}: at $MOUNT_DIR"
    ls -la "$MOUNT_DIR" | head -n 15
    exit 0
  fi
  sleep 0.5
done

echo "Timed out waiting for rclone mount at $MOUNT_DIR" >&2
exit 1
REMOTE_EOF
}

cmd_flush() {
  local acct=${1:?usage: colab_multi.sh flush <acct> <session>}
  local session=${2:?usage: colab_multi.sh flush <acct> <session>}

  echo "[$acct/$session] Flushing and unmounting /content/drive/MyDrive..."
  run_colab_as "$acct" ssh -s "$session" -- bash -s <<'REMOTE_EOF'
set -euo pipefail
MOUNT_DIR="/content/drive/MyDrive"
sync
if mountpoint -q "$MOUNT_DIR" 2>/dev/null; then
  fusermount3 -u "$MOUNT_DIR" || fusermount -u "$MOUNT_DIR"
  for _ in $(seq 1 60); do
    pgrep -x rclone >/dev/null 2>&1 || break
    sleep 1
  done
  echo "Flushed and unmounted $MOUNT_DIR cleanly."
else
  echo "$MOUNT_DIR is not mounted."
fi
REMOTE_EOF
}

cmd_list() {
  if [[ ! -d "$ACCOUNTS_ROOT" ]]; then
    echo "No accounts configured under $ACCOUNTS_ROOT"
    return
  fi
  for dir in "$ACCOUNTS_ROOT"/*; do
    [[ -d "$dir" ]] || continue
    local acct
    acct=$(basename "$dir")
    local status="unauthenticated"
    [[ -f "$dir/gcloud/application_default_credentials.json" ]] && status="ready"
    echo "$acct ($status) -> $dir"
  done
}

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

case "$1" in
  -h|--help|help)
    usage
    ;;
  auth)
    shift
    cmd_auth "$@"
    ;;
  mount)
    shift
    cmd_mount "$@"
    ;;
  flush)
    shift
    cmd_flush "$@"
    ;;
  list)
    cmd_list
    ;;
  *)
    acct=$1
    shift
    if [[ $# -lt 1 ]]; then
      usage
      exit 1
    fi
    run_colab_as "$acct" "$@"
    ;;
esac
