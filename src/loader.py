import torch
import numpy as np
import polars as pl
from sklearn.preprocessing import StandardScaler, OneHotEncoder

data = np.load('./data/DGraphFin/dgraphfin.npz')

print(f"Available keys: {[data.files]}")

x = data['x']
y = data['y']
edge_type = data['edge_type']
edge_index = data['edge_index']
edge_timestamp = data['edge_timestamp']
train_mask = data['train_mask']
valid_mask = data['valid_mask']
test_mask = data['test_mask']

edges_df = pl.DataFrame({
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

# convert edges to numpy for easier handling missing values
edge_index = edges_df.select(['src_node', 'destination_node']).to_numpy()
edge_type = edges_df['edge_type'].to_numpy()
edge_timestamp = edges_df['timestamp'].to_numpy()

# using Trick B for handling missing values according to https://doi.org/10.48550/arXiv.2207.03579
# step1: convert True and False flags to binary flags for missing values
isMissing = ( x==-1 )
missing_flags = isMissing.astype(np.float32)

# step2: replace -1 to 0 by masking it
x_binary = x.copy().astype(np.float32)
x_binary[isMissing] = 0

# step3: concatenate the array
x_flags = np.concatenate((x_binary, missing_flags), axis=1) # x array but with added 17 new columns of flags

# normalise training_mask
numFeatures = 17
training_feature = x_flags[train_mask, :numFeatures]

scaler = StandardScaler()
scaler.fit(training_feature)

# normalise all features
x_normalised = x_flags.copy()
all_features = x_normalised[:, :numFeatures]
normalised_features = scaler.transform(all_features)
x_normalised[:, :numFeatures] = normalised_features

# convert the data type to PyG format
#featrures
x_tensor = torch.FloatTensor(x_normalised)

# labels
# change background nodes of 2 and 3 to -1 as they are not needed for loss function or predictions
y_labels = y.copy()
y_labels[(y_labels == 2) | (y_labels == 3)] = -1
y_tensor = torch.FloatTensor(y_labels)

#edges
# since edge_index in format of (999, 2), we transpose it to follow PyG format
edge_index_tensor = torch.FloatTensor(edge_index.T)
edge_type_tensor = torch.FloatTensor(edge_type)
edge_timestamp_tensor = torch.FloatTensor(edge_timestamp)

#masks
train_mask_tensor = torch.FloatTensor(train_mask)
valid_mask_tensor = torch.FloatTensor(valid_mask)
test_mask_tensor = torch.FloatTensor(test_mask)

edge_type = data['edge_type']
edge_index = data['edge_index']
edge_timestamp = data['edge_timestamp']
train_mask = data['train_mask']
valid_mask = data['valid_mask']
test_mask = data['test_mask']

# save into pYg format
cleaned_data = {
    'x': x_tensor,
    'y': y_tensor,
    'edge_index': edge_index_tensor,
    'edge_type': edge_type_tensor,
    'edge_timestamp': edge_timestamp_tensor,
    'train_mask': train_mask_tensor,
    'valid_mask': valid_mask_tensor,
    'test_mask': test_mask_tensor,
}

torch.save(cleaned_data, "./data/dgraphfin_cleaned.pt")