# Script to count unique values in each column of a text file
import sys

def count_unique_values(file_path):

    try:
        with open(file_path, 'r') as file:
            data = [line.split() for line in file if line.strip()]

        num_columns = len(data[0])
        unique_counts = [len(set(column)) for column in zip(*data)]

        for i, count in enumerate(unique_counts, start=1):
            print(f"Column {i}: {count} unique values")

    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Replace with your actual file path
PSR_name: str = sys.argv[1]
count_unique_values(PSR_name + '_overlap_frame_tie.txt')
