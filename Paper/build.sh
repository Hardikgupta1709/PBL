#!/bin/bash
# Build the research paper
# Requires: pdflatex + bibtex (from texlive-full or mactex)
# Install: brew install --cask mactex-no-gui  (or: brew install basictex)

set -e
cd "$(dirname "$0")"

echo "=== Building Paper ==="

# First pass
pdflatex -interaction=nonstopmode main.tex

# Bibliography
bibtex main

# Second + third pass (resolve references)
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex

echo ""
echo "=== Build complete: main.pdf ==="
ls -la main.pdf
