#!/usr/bin/env bash
#
# Delete all generated files in:
#   • Excel-Files/
#   • Pipeline/PDFs/
#   • JSON-Metadata/
#
# Directories themselves are kept, only their contents are removed.

set -euo pipefail

# Project root = script’s directory
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

echo "Clearing Excel-Files …"
rm -rf Excel-Files/*

echo "Clearing PDFs …"
rm -rf PDFs/*

echo "Clearing JSON-Metadata …"
rm -rf JSON-Metadata/*

echo "Done."