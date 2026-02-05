
import os
import pandas as pd
import numpy as np

def read_dataset(dataset_path, extracted_path, sampling_frequency):
    columns = ['X', 'Y', 'Z']
    df = pd.DataFrame()
    
    # Walk through the directory and read all .txt files
    for root, dirs, files in os.walk(extracted_path):
        for filename in files:
            if filename.startswith('Accelerometer'):
                file_path = os.path.join(root, filename)
                
                # Extract activity and volunteer info from filename
                # Filename format: Accelerometer-2011-04-11-13-28-18-brush_teeth-f1.txt
                file_parts = filename.split('-')
                
                # The activity is usually the second to last part (e.g., brush_teeth)
                # But sometimes it might be different, let's look at the folder name
                activity = os.path.basename(root)
                
                # The volunteer is the last part before .txt extension
                volunteer = file_parts[-1].split('.')[0]
                
                # Read the file
                try:
                    # Read the dataset from the file
                    # Files are space separated, no header
                    dataset = pd.read_csv(file_path, sep=' ', header=None, names=columns)
                    
                    # Add labels
                    dataset['Activity'] = activity
                    dataset['Volunteer'] = volunteer
                    
                    # Append the dataset to the DataFrame
                    df = pd.concat([df, dataset])
                except Exception as e:
                    print(f"Error reading {filename}: {e}")
                    continue
                    
    return df

# Path handling
base_dir = '/home/odin/achal/Pervasive-Computing-for-Health/tutorial_3'
data_path = os.path.join(base_dir, 'adl_dataset')
extract_path = os.path.join(base_dir, 'adl_dataset_extracted')

# Check raw extract
if not os.path.exists(extract_path):
    print("Dataset not extracted!")
    exit(1)

print("Reading dataset...")
df = read_dataset(data_path, extract_path, 32)
print(f"Total rows: {len(df)}")
print(f"Total volunteers: {df['Volunteer'].nunique()}")

print("\n--- Sample Counts per Volunteer ---")
counts = df.groupby('Volunteer').size().sort_values()
print(counts)

print("\n--- Check Constraints ---")
for w_size in [150, 175, 200]:
    valid_users = counts[counts >= w_size].count()
    print(f"Window Size {w_size}: {valid_users} / {len(counts)} volunteers have enough data.")
    dropped = counts[counts < w_size]
    if len(dropped) > 0:
        print(f"   Dropped: {dropped.index.tolist()}")
