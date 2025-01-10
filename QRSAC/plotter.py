import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import pickle

def plot_numpy_array(arr, input_file, output_file=None):
    # Convert input to NumPy array if it's not already
    if not isinstance(arr, np.ndarray):
        arr = np.array(arr)
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(np.array(range(len(arr)))*10, arr, marker='o')
    
    # Customize the plot
    plt.title(input_file[14:-14])
    plt.xlabel('Episode')
    plt.ylabel('Mean Rewards')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Save the figure if output file is specified
    if output_file:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        plt.savefig(output_file)
        print(f"Figure saved to {output_file}")

def load_pickle_file(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Plot data from a pickle file')
    parser.add_argument('--input', '-i', help='Path to the input pickle file')
    parser.add_argument('--output', '-o', help='Output file path to save the figure (e.g., plots/my_plot.png)', default=None)
    
    # Parse arguments
    args = parser.parse_args()
    
    # Load data from pickle file
    try:
        data = load_pickle_file(args.input)
        
        # Plot the data
        plot_numpy_array(data, args.input, args.output)
    
    except FileNotFoundError:
        print(f"Error: File {args.input} not found.")
    except pickle.UnpicklingError:
        print(f"Error: Unable to unpickle the file {args.input}.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == '__main__':
    main()