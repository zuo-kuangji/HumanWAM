#!/usr/bin/env python3
"""Create a fresh metadata-only GR00T view of existing local model assets."""

import argparse
import json

from humanwam.assets import prepare_gr00t_offline_view


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--backbone", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    receipt = prepare_gr00t_offline_view(args.checkpoint, args.backbone, args.output)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
