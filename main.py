"""Main entry point for the forex trading bot."""

import os
import yaml
from dotenv import load_dotenv


def main():
    """Initialize and start the forex trading bot."""
    load_dotenv()

    config_path = os.path.join(os.path.dirname(__file__), "config", "config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    print("Forex Bot — Phase 1 foundation loaded successfully.")
    print(f"Instruments: {config['broker']['instruments']}")
    print(f"Primary timeframe: {config['broker']['primary_timeframe']}")


if __name__ == "__main__":
    main()
