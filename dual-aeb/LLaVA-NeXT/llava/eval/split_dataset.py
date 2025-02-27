import json
import os
import argparse
import math

def split_dataset(input_file, output_dir, num_chunks):
    with open(input_file, "r") as f:
        data = json.load(f)

    chunk_size = math.ceil(len(data) / num_chunks)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i in range(num_chunks):
        chunk = data[i*chunk_size:(i+1)*chunk_size]
        chunk_file = os.path.join(output_dir, f"chunk_{i}.json")
        with open(chunk_file, "w") as outfile:
            json.dump(chunk, outfile, indent=4)
        print(f"Chunk {i} saved to {chunk_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split dataset into chunks.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input dataset JSON file.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the output chunks.")
    parser.add_argument("--num_chunks", type=int, required=True, help="Number of chunks to split the dataset into.")
    
    args = parser.parse_args()
    split_dataset(args.input_file, args.output_dir, args.num_chunks)
