import pandas as pd

# Load the data from CSV file
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

df = load_data('archive/allocation.csv')

# Drop specified columns
df = df.drop(columns=["ContainerID", "ItemDescription", "Qty_Used", "FabricID"], axis=1)

# Drop rows with NaN values
df = df.dropna()

print(df.info())
