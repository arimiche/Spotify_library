import pandas as pd

# Function to load the data
def loaddata(x):
    dfraw = pd.read_csv(x)
    df = pd.DataFrame(dfraw)
    return df
