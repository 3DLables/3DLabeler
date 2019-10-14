import pandas as pd
from io import StringIO

def tag_parser(string):
    with open(string) as f:
        t = f.read()
        t = t.split("Points =\n")[1]
        t = t.replace(" 0.1 1 1 \"Marker\"", "")
        t = t.replace(";", "")
        t = t.replace(" \n", "\n")
        t = StringIO(t)
        df = pd.read_csv(t, sep = " ", header=None).loc[:, 1:]
        df.columns = ["x", "y", "z"]
        
    return df
