"""Benchmark local embedding models on a client's documents."""

import sys

# fine_print is an editable dependency, so importing it would write .pyc files into
# ../fine-print, which this project must only read. This has to run before that import.
sys.dont_write_bytecode = True
