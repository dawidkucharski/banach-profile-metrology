"""Allow running the BTRI pipeline as ``python -m btri``."""

from .cli import main

raise SystemExit(main())
