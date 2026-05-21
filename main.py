"""
main.py — Entry point for Smart RSU: AI Traffic Intelligence Engine.

Usage:
    python main.py
    python main.py --config config/settings.yaml
    python main.py --config config/settings_edge.yaml   # Raspberry Pi / Jetson config
"""

import argparse
from ai.pipeline import TrafficIntelligencePipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description="Smart RSU — AI Traffic Intelligence Engine | IIT Madras Hackathon 2026"
    )
    parser.add_argument(
        "--config", type=str,
        default="config/settings.yaml",
        help="Path to settings YAML (default: config/settings.yaml)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    pipeline = TrafficIntelligencePipeline(config_path=args.config)
    pipeline.run()