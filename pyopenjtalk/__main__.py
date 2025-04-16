import argparse
import json
import sys

from . import run_frontend


def main() -> None:
    """Command line interface for pyopenjtalk.run_frontend()"""
    parser = argparse.ArgumentParser(description='Run OpenJTalk"s text processing frontend')
    parser.add_argument("text", type=str, help="Input text")
    parser.add_argument("--run-marine", action="store_true", help="Estimate accent using marine")
    parser.add_argument("--use-vanilla", action="store_true", help="Return vanilla NJDFeature list")
    args = parser.parse_args()

    try:
        features = run_frontend(args.text, run_marine=args.run_marine, use_vanilla=args.use_vanilla)
        for feature in features:
            print(json.dumps(feature, ensure_ascii=False))
    except Exception as e:
        print(f"Error: {e!s}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
