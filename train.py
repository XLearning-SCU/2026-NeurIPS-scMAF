#!/usr/bin/env python3
"""Backward-compatible wrapper for the packaged scMAF CLI."""

from scmaf.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
