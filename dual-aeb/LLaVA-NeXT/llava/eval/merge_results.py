import json
import os
import argparse

def merge_results(input_dir, output_file):
    merged_results = []
    
    for filename in os.listdir(input_dir):
        if filename.endswith("_conversations.json"):
            file_path = os.path.join(input_dir, filename)
            with open(file_path, "r") as f:
                data = json.load(f)
                merged_results.extend(data)
    
    with open(output_file, "w") as outfile:
        json.dump(merged_results, outfile, indent=4)
    print(f"Merged results saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge multiple result files into one.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing the result files.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output merged JSON file.")
    
    args = parser.parse_args()
    merge_results(args.input_dir, args.output_file)
