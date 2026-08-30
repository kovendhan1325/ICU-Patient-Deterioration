import pandas as pd
import glob
import os

files = glob.glob('required files/*.csv.gz')
for f in files:
    df = pd.read_csv(f, nrows=0) # Just to get columns
    cols = len(df.columns)
    # To get rows without loading entire file into memory, we can count lines, or if it's small, read_csv
    # Alternatively, just load the whole thing if it's not too huge.
    # vitalPeriodic is 19MB zipped, so loading might take a few seconds
    # using a simple line count for gzip
    import gzip
    with gzip.open(f, 'rt', encoding='utf-8') as gz:
        rows = sum(1 for _ in gz) - 1 # subtract header
    print(f"{os.path.basename(f)}: Rows={rows}, Columns={cols}")
