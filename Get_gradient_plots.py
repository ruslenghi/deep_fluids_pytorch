import glob
import numpy as np
import matplotlib.pyplot as plt

def plot_gradients(input_folder='my_gradients', output_folder='my_gradients'):
    
    for np_name in glob.glob(f'{input_folder}/*'):
        # Load the gradient data from the file
        gradient_data = np.loadtxt(np_name)
        epochs = list(range(len(gradient_data)))

        # Plot the gradient data
        plt.plot(epochs, gradient_data, c='r')
        plt.xlabel('Epochs')
        plt.ylabel('Gradient')
        plt.title(np_name[len(input_folder)+1:])
        
        # Save the plot as a PNG image
        output_file = f'{output_folder}/{np_name[len(input_folder)+1:]}_gradients.png'
        plt.savefig(output_file, dpi=300)
        plt.clf()

        print(f"Saved plot: {output_file}")

if __name__ == "__main__":
    plot_gradients()
