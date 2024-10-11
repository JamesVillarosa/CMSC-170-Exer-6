# Villarosa, James Carl V.
# GH 4L

import numpy as np

def readfile():                                                                             # Function that read file and set value for variables
    with open("input.txt", "r") as f:
        content = f.readlines()                                                             # Ready every lines in input.txt
    
    learning_rate = float(content[0].strip())                                               # Get learning rate (r)
    threshold = float(content[1].strip())                                                   # Get threshold (t)
    bias_weight = float(content[2].strip())                                                 # Get bias weight (wb)
    
    feature_vectors = [list(map(float, line.strip().split())) for line in content[3:]]      # Read the lines starting for line 4 and contain it in list
    feature_vectors = np.array(feature_vectors)                                             # Convert created list into numpy array
    
    weights = np.zeros(feature_vectors.shape[1] - 1)                                        # Initialize weights to zeros
    
    return learning_rate, threshold, bias_weight, weights, feature_vectors                  # Return variables need in computing perceptron


def perceptron(learning_rate, threshold, bias_weight, weights, feature_vectors):            # Function that implement perception algorithm and write output in output.txt
    max_iterations = 1000                                                                   # Initialize max iteration
    iteration = 1                                                                           # Initialize iteration counter
    num_features = feature_vectors.shape[1] - 1                                             # Number of x's

    with open("output.txt", "w") as f:                                                      # Open the output file for writing

        f.write(f"Using {num_features} x's, r={learning_rate}, t={threshold}, wb={bias_weight}\n\n")    # Write the x's, r, t, and wb we use in perceptron
        
        while iteration <= max_iterations:                                                  
            f.write(f"Iteration {iteration}:\n")                                            # Write the iteration count

                                                                                            # Dynamically generate header based on the number of features
            header = "  ".join([f"x{i}" for i in range(num_features)]) + "  " + "  ".join([f"w{i}" for i in range(num_features)]) + "   wb     a     y     z\n"
            f.write(header)
            
            has_errors = False                                                              # Assume no errors to track if we will adjust weights

            for vector in feature_vectors:                                                  # Loop through each feature vector in the dataset
                x = vector[:-1]                                                             # Feature values
                z = vector[-1]                                                              # Target label
                
                a = np.dot(x, weights) + bias_weight                                        # Computer perceptron value
                
                if a >= threshold:                                                          # Determine classification
                    y = 1
                else:
                    y = 0
                
                feature_str = "   ".join([f"{xi:.1f}" for xi in x])                         # Initialize dynamically features (xn)
                weight_str = "   ".join([f"{wi:.1f}" for wi in weights])                    # Intialize dynamically weights (wn)
                f.write(f"{feature_str}   {weight_str}   {bias_weight:.1f}   {a:.1f}   {y}   {z}\n")    # Write the computed values
                
                error = z - y                                                               # Calculate error

                if error != 0:                                                              # If there's an error
                    weights = weights + (learning_rate * error * x)                         # Adjust weights based on the error and features
                    bias_weight = bias_weight + (learning_rate * error)                     # Update bias weight using the error
                    has_errors = True                                                       # Set has_errors to True since there's an error
            
            if not has_errors:                                                              # If no errors were found, stop (weights converged)
                break
            
            iteration += 1                                                                  # Increment iteration count

        if iteration > max_iterations:                                                      # If iteration count > 1000, displate INTERRUPTED
            f.write(f"INTERRUPTED: input.txt is not linearly separable. Stopped at {max_iterations}th iteration.\n")


def main():
    learning_rate, threshold, bias_weight, weights, feature_vectors = readfile()            # Call for read file to get the value of each variable
    perceptron(learning_rate, threshold, bias_weight, weights, feature_vectors)             # Call for function perceptron to compute and write output

main()                                                                                      # Call for main function