import os
import pandas as pd
import re

# Function to sanitize filenames
def sanitize_filename(filename):
    # Replace invalid characters with underscore
    # Windows doesn't allow: < > : " / \ | ? *
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
    return sanitized

# Path to your Parquet file
parquet_file = 'meinwissen_output_layer.parquet'

# Directory to save the text files
output_dir = 'src/late_chunking/data/documents'

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Load the Parquet file using Pandas and PyArrow
df = pd.read_parquet(parquet_file, engine='pyarrow')

# Ensure required columns exist
if 'name' not in df.columns or 'page_content' not in df.columns:
    raise ValueError("The Parquet file must contain 'name' and 'page_content' columns.")

# Limit the DataFrame to the first 5 rows for testing
df_test = df

# Iterate through the first 5 rows and write each to a text file
for index, row in df_test.iterrows():
    # Extract file name and content
    file_name = sanitize_filename(f"{row['name']}.txt")
    content = row['page_content']

    # Full path for the text file
    file_path = os.path.join(output_dir, file_name)

    # Write content to the text file
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(content)

print(f"Test: First 5 text files successfully written to: {output_dir}")
