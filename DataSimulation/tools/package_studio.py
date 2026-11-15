"""Compatibility entry point for the current portable release builder.

The former September builder is preserved in docs/archive. Current releases
include the explicitly selectable historical models.
"""
from package_current_circular import main

if __name__ == "__main__":
    main()
