"""
run_tests.py — Test runner for the split test suite.
Discovers all test_*.py files in the tests/ folder and runs them
with the same ✓/✗ output format as the original tests.py.

Run: python run_tests.py
"""

import sys
import unittest

from tests.mock_data import VerboseRunner, _LOOP


def main():
    loader = unittest.TestLoader()

    # Load each module explicitly to preserve class grouping
    import importlib
    import pkgutil
    import tests

    modules = []
    for importer, modname, ispkg in pkgutil.iter_modules(tests.__path__):
        if modname.startswith("test_"):
            mod = importlib.import_module(f"tests.{modname}")
            modules.append(mod)

    suite = unittest.TestSuite()
    for mod in modules:
        suite.addTests(loader.loadTestsFromModule(mod))

    runner = VerboseRunner(verbosity=0, stream=sys.stdout)
    result = runner.run(suite)
    _LOOP.close()
    sys.exit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
