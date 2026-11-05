#!/bin/bash
# The upstream README's rendering path on one card, without a monitor:
#   python scripts/launch_nodes.py --robot sim_ur_pybullet_apple_interactive --robot_name robot_iphone
#   python scripts/run_env_sim.py --agent replay_trajectory_and_save
# Xvfb takes the PyBullet GUI and cv2.imshow windows. run_env_sim.py keeps looping after the last
# trajectory, so this script stops both processes once the agent reports there is nothing left.
#   bash radeon/replay.sh [out_dir]
set -uo pipefail
cd "$(dirname "$0")/.."
OUT=$(realpath -m "${1:-results/replay}")
mkdir -p "$OUT"
export DISPLAY=${DISPLAY:-:99}

Xvfb "$DISPLAY" -screen 0 1920x1080x24 -nolisten tcp >"$OUT/xvfb.log" 2>&1 &
XVFB=$!
sleep 2

python -u scripts/launch_nodes.py --robot sim_ur_pybullet_apple_interactive --robot_name robot_iphone \
  >"$OUT/server.log" 2>&1 &
SERVER=$!
until grep -q "Ready to serve." "$OUT/server.log"; do
  if ! kill -0 "$SERVER" 2>/dev/null; then
    echo "server exited before serving"; tail -n 40 "$OUT/server.log"; kill "$XVFB"; exit 1
  fi
  sleep 2
done

date -Is > "$OUT/client_start.txt"
python -u scripts/run_env_sim.py --agent replay_trajectory_and_save >"$OUT/client.log" 2>&1 &
CLIENT=$!
( sleep 120; ffmpeg -loglevel error -f x11grab -video_size 1920x1080 -i "$DISPLAY" -frames:v 1 -y "$OUT/display.png" ) &

status=0
until grep -q "No more recorded trajectories." "$OUT/client.log"; do
  if ! kill -0 "$CLIENT" 2>/dev/null; then
    echo "client exited before the replay finished"; tail -c 4000 "$OUT/client.log"; status=1; break
  fi
  if ! kill -0 "$SERVER" 2>/dev/null; then
    echo "server exited during the replay"; tail -n 40 "$OUT/server.log"; status=1; break
  fi
  sleep 5
done
date -Is > "$OUT/client_end.txt"

kill "$CLIENT" "$SERVER" 2>/dev/null
wait "$CLIENT" "$SERVER" 2>/dev/null
kill "$XVFB" 2>/dev/null
exit $status
