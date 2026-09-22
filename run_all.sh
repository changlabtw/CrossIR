#!/usr/bin/env bash
#
# Run the whole analysis, in the order the notebooks depend on each other.
#
# Two orderings are load-bearing and are the reason this is a script rather than
# "run the notebooks": 02 must precede 07, which appends a fourth sheet to the
# workbooks 02 writes, and 07 must precede 08, which splits participants by the
# label 07 predicts. Notebooks 02-05 depend only on 01.
#
# Raw cohort files must already be in place; see the Data availability section of
# README.md. Expect roughly 25 minutes, most of it notebook 08.
set -euo pipefail

cd "$(dirname "$0")"

for notebook in notebooks/0[1-8]_*.ipynb; do
    echo "==> ${notebook}"
    jupyter nbconvert --to notebook --execute --inplace "${notebook}"
done

echo "==> done; every table and figure in output/ has been regenerated"
