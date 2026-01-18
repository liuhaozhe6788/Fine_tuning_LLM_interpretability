#!/usr/bin/env bash
set -e

# Install requirements (allow system override if needed)
pip install -r requirements.txt --break-system-packages

# Uninstall bitsandbytes (known conflicts)
pip uninstall bitsandbytes -y || true