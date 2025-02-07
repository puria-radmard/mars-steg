#!/bin/bash
apt-get update && apt-get install -y tmux
apt-get install -y tmux
tmux new -s mysession
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo "Setup completed successfully."
chmod +x ./scripts/run_pricing_game_first_runpod_test.sh
./scripts/run_pricing_game_first_runpod_test.sh
