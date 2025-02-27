import pandas as pd
import pathlib as pl

csv_path = pl.Path('/home/lorenz/data/mrct1000_nobatch/dataset_insight.csv')

df = pd.read_csv(csv_path)

df["RTComplete"] = (df["RTFiles"] != 0) & (df["RTFiles"] == df["RT_with_matching_CT"])

print(sum(df["RTComplete"]))

df.to_csv(csv_path.parent/('extended_'+str(csv_path.name)), index=False)