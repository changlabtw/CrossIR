#!/usr/bin/env bash
#
# Run the whole analysis, in the order the notebooks depend on each other.
#
# The orderings that matter are data dependencies: 01 writes the tables everything
# else reads, then 06 -> 07 -> 08, because 07 scores Taiwan Biobank with the model
# 06 saves and 08 splits participants by the label 07 predicts. Notebooks 02-05
# need only 01 and can run in any order.
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
