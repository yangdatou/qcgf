# Improved code

import sys
import os
from typing import List

def main():
    # Get the command line arguments
    args: List[str] = sys.argv[1:]
    
    # Check if there are enough arguments
    if len(args) < 2:
        print("Not enough arguments")
        return

    # Get the input and output file paths
    input_file: str = args[0]
    output_file: str = args[1]

    # Check if the input file exists
    if not os.path.exists(input_file):
        print("Input file does not exist")
        return

    # Read the content of the input file
    with open(input_file, 'r') as f:
        content: str = f.read()

    # Modify the content
    modified_content: str = content.upper()

    # Write the modified content to the output file
    with open(output_file, 'w') as f:
        f.write(modified_content)

if __name__ == '__main__':
    main