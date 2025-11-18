import numpy as np
import polars as pl
from sklearn.preprocessing import StandardScaler, OneHotEncoder

data = np.load('./DGraphFin/dgraphfin.npz')

print(f"Available keys: {[data.files]}")

x = data['x']
y = data['y']
edge_type = data['edge_type']
edge_index = data['edge_index']
edge_timestamp = data['edge_timestamp']
train_mask = data['train_mask']
valid_mask = data['valid_mask']
test_mask = data['test_mask']

edges_df = pl.Dataframe({
    "src_node": edge_index[:, 0],
    "destination_node": edge_index[:, 1],
    "timestamp": edge_timestamp,
    "edge_type": edge_type
})

#remove self-loop nodes
edges_df = edges_df.filter(pl.col("src_node") != pl.col("destination_node"))

# remove out of range edges
numNodes = len(x)
edges_df = edges_df.filter((pl.col("src_node") >= 0) & 
                           (pl.col("src_node") < numNodes) &
                           (pl.col("destination_node") >= 0) &
                           (pl.col("destination_node") < numNodes))

#sort the edges by its timestamp
edges_df = edges_df.sort("timestamp")
