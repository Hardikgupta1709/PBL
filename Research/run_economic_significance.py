"""Day-5 economic significance refresh for updated success pairs."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Research.run_reviewer_fixes import compute_economic_significance


def main():
    compute_economic_significance(quick=False)


if __name__ == '__main__':
    main()
