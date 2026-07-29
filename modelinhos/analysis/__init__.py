"""Dataset/recipe analysis toolkit. Facts are computed from samples,
verdicts read fact DataFrames; "split" is not a concept here -- callers
add it as a column. Kept import-thin on purpose (the Sample contract
and the matcher): planned for extraction into a separate library."""
