"""Operator-only secret provisioning; input never goes through LLM/API/logs."""
import argparse
from getpass import getpass
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from axen.evolution.connectors import WindowsCredentialVault

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("service", choices=["github", "youtube"])
    args = parser.parse_args()
    secret = getpass("Enter narrowly scoped read-only token/API key (hidden): ")
    if not secret: raise SystemExit("Empty credential refused")
    WindowsCredentialVault().put(args.service, secret)
    del secret
    print("OS credential stored. Separate explicit connector authorization is still required.")
