#!/bin/bash
# Idempotent launcher for the pinned HF trainer. ssh cmdline never contains the
# pattern below, so pkill cannot kill the calling session.
pkill -9 -f '^/root/venv444/bin/python train_hf.py' 2>/dev/null
sleep 1
cd /root || exit 1
rm -f hf.log
setsid nohup /root/venv444/bin/python train_hf.py > /root/hf.log 2>&1 < /dev/null &
sleep 10
if pgrep -f '^/root/venv444/bin/python train_hf.py' > /dev/null; then
  echo "LAUNCHED pid=$(pgrep -f '^/root/venv444/bin/python train_hf.py' | head -1)"
else
  echo "FAILED TO START"
fi
tail -3 /root/hf.log 2>/dev/null
