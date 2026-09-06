import os
import urllib.request
import pandas as pd
import torch

# Fix for OpenMP runtime crash on Windows (PyTorch + NumPy conflict)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from src.graph_utils import build_adjacency_matrix

def download_lookup_if_missing(raw_dir):
    lookup_path = os.path.join(raw_dir, "taxi_zone_lookup.csv")
    if not os.path.exists(lookup_path):
        print("Downloading official NYC taxi_zone_lookup.csv...")
        url = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv"
        urllib.request.urlretrieve(url, lookup_path)
        print("Saved lookup file to:", lookup_path)
    return lookup_path

def process_raw_csv_data():
    raw_dir = os.path.join("data", "raw")
    raw_csv_path = os.path.join(raw_dir, "2017_Yellow_Taxi_Trip_Data.csv")
    processed_dir = os.path.join("data", "processed")
    os.makedirs(processed_dir, exist_ok=True)
    
    # 1. Download Lookup File
    lookup_path = download_lookup_if_missing(raw_dir)
    zone_names_df = pd.read_csv(lookup_path)
    
    # 2. Load Trip Data
    print("Loading 2017 Yellow Taxi Trip CSV data...")
    df = pd.read_csv(raw_csv_path)
    
    # 3. Filter valid taxi zones
    df_clean = df[
        (df['PULocationID'] >= 1) & (df['PULocationID'] <= 263) &
        (df['DOLocationID'] >= 1) & (df['DOLocationID'] <= 263)
    ]
    print(f"Cleaned dataset records: {len(df_clean):,} rows")
    
    # 4. Build Zone Pair Counts & Adjacency Matrix
    print("Aggregating zone-to-zone trips for graph topology...")
    zone_pairs = (
        df_clean.groupby(['PULocationID', 'DOLocationID'])
        .size()
        .reset_index(name='trip_count')
    )
    
    adj_matrix = build_adjacency_matrix(zone_pairs, num_zones=263)
    adj_matrix_path = os.path.join(processed_dir, "adjacency_matrix.pt")
    torch.save(adj_matrix, adj_matrix_path)
    print("Saved ->", adj_matrix_path)
    
    # 5. Calculate Zone Features & Merge Zone Names
    print("Aggregating individual zone metrics...")
    zone_summary = (
        df_clean.groupby('PULocationID')
        .agg(
            avg_fare=('total_amount', 'mean'),
            avg_trip_distance=('trip_distance', 'mean'),
            total_trips=('passenger_count', 'count')
        )
        .reset_index()
        .rename(columns={'PULocationID': 'LocationID'})
    )
    
    # Merge with neighborhood names
    zone_summary = zone_names_df.merge(zone_summary, on='LocationID', how='left').fillna(0)
    
    zone_summary_path = os.path.join(processed_dir, "zone_features.csv")
    zone_summary.to_csv(zone_summary_path, index=False)
    print("Saved ->", zone_summary_path)

if __name__ == "__main__":
    process_raw_csv_data()