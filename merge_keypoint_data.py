"""
Keypoint Data Merger Utility

This script merges multiple CSV files containing training data for the
keypoint classifier. Useful for combining datasets from different
collection sessions or merging backups with new data.
"""
import csv

# Paths to your CSV files
old_csv_path = 'model/keypoint_classifier/oldkeypoint.csv'  # Your backup file
new_csv_path = 'model/keypoint_classifier/keypoint.csv'      # Your current file with new 'A' samples
merged_csv_path = 'model/keypoint_classifier/keypoint_merged.csv'  # Output file

# Read all rows from both files
old_rows = []
new_rows = []

with open(old_csv_path, 'r', newline='') as f:
    reader = csv.reader(f)
    old_rows = list(reader)

with open(new_csv_path, 'r', newline='') as f:
    reader = csv.reader(f)
    new_rows = list(reader)

# Combine the rows
merged_rows = old_rows + new_rows

# Write the merged data to a new file
with open(merged_csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(merged_rows)

print(f"Merged data saved to {merged_csv_path}") 