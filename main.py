"""
main.py — Entry point for AI Traffic Intelligence Engine.
Usage:
    python main.py
    python main.py --config config/settings.yaml
"""

import argparse
from ai.pipeline import TrafficIntelligencePipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description="AI Powered Smart Road Infrastructure — Traffic Intelligence Engine"
    )
    parser.add_argument(
        "--config", type=str,
        default="config/settings.yaml",
        help="Path to settings YAML"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    pipeline = TrafficIntelligencePipeline(config_path=args.config)
    pipeline.run()