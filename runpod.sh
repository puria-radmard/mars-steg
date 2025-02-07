#!/bin/bash
python3.11 -m pip install --upgrade pip
python3.11 -m pip install -r mars-steg/requirements.txt


HF_TOKEN=
python3.11 -c "from huggingface_hub import HfFolder; HfFolder.save_token('$HF_TOKEN')"

echo "Setup completed successfully."

