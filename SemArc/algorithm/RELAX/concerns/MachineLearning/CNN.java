package cnn;

import utilities.Dataset;
import utilities.ImageAugmentor;

import java.util.List;

/**
 * Represents a Convolutional Neural Network (CNN) with multiple layers.
 * Supports forward and backward propagation, training with mini-batches, and testing.
 *
 * <p>Features</p>
 * <ul>
 *     <li>Customizable layer structure.</li>
 *     <li>Supports different cost functions.</li>
 *     <li>Allows optional data augmentation.</li>
 *     <li>Mini-batch training with gradient application.</li>
 * </ul>
 *
 * <p>Usage:</p>
 * <pre>{@code
 *     CNN A = new CNN(layers, costFunction, allowAugmenting); // Create a network
 *     Matrix B = A.forward(input); // Forward propagation
 *     A.backward(predictedOutput, expectedOutput); // Backpropagation
 *     Matrix C = A.predict(input); // Returns output of forward propagation
 *     A.train(dataset, learningRate, maxEpochs, batchSize, targetCost, continuous); // Train network on data
 *     A.test(dataset); // Test network on dataset and print success rate
 * }</pre>
 *
 * @author Braeden West
 * @version 1.0 (2025-03-16)
 * @since 2025-03-16
 */

public class CNN {
    private List<Layer> layers; // Holds all layers (convolution, pooling, fully connected)
    private CostFunction costFunction; // Cost function for network
    private boolean allowAugmenting;

    public CNN(List<Layer> layers, CostFunction costFunction, boolean allowAugmenting) {
        this.layers = layers;
        this.costFunction = costFunction;
        this.allowAugmenting = allowAugmenting;
    }

    // Forward propagate through the entire network
    public Matrix forward(Matrix input) {
        Matrix output = input;
        output = output.flatten(); // TODO ADD FIX TO FLATTEN
        for (Layer layer : layers) {
            output = layer.forward(output); // Pass previous output as the input to the next layer
        }
        return output; // Final output
    }

    // Backward propagate through the entire network
    public void backward(Matrix predictedOutput, Matrix expectedOutput) {
        Matrix error = null;
        for (int i = layers.size() - 1; i >= 0; i--) {
            if (i != layers.size() - 1) {
                error = layers.get(i).backward(error);
            } else { // Output layer
                error = ((FullyConnectedLayer) layers.get(i)).backward(predictedOutput, expectedOutput);
            }
        }
    }

    // Predict correct output
    public Matrix predict(Matrix input) {
        return forward(input);
    }

    // Train the network on dataset with configurable settings
    public void train(Dataset dataset, double learningRate, int maxEpochs, int batchSize, Double targetCost, boolean continuous) {
        int epoch = 1;
        double currentCost = Double.MAX_VALUE;

        while (continuous || (epoch <= maxEpochs && (targetCost == null || currentCost > targetCost))) {
            dataset.shuffleDataset(); // Shuffle at the start of each epoch
            double totalCost = 0;
            int batchCount = 0;

            // Process dataset in mini batches
            while (dataset.hasNextBatch()) {
                Dataset.Batch batch = dataset.getNextBatch(batchSize);
                List<Matrix> inputs = batch.inputs();
                List<Matrix> expectedOutputs = batch.outputs();
                for (int i = 0; i < inputs.size(); i++) {
                    Matrix input = inputs.get(i);
                    Matrix expectedOutput = expectedOutputs.get(i);

                    if (allowAugmenting) {
                        input = ImageAugmentor.augment(input);
                    }

                    // Forward pass
                    Matrix predictedOutput = forward(input);

                    // Compute cost
                    Matrix outputError = costFunction.derivative(predictedOutput, expectedOutput);
                    totalCost += costFunction.calculate(predictedOutput, expectedOutput);

                    // Backward pass
                    backward(predictedOutput, expectedOutput);
                }

                // Apply accumulated gradients
                for (Layer layer : layers) {
                    if (layer instanceof FullyConnectedLayer) {
                        ((FullyConnectedLayer) layer).applyGradient(learningRate);
                        ((FullyConnectedLayer) layer).clearGradient();
                    }
                }
                batchCount++;
            }

            // Compute average cost for each epoch
            currentCost = totalCost / batchCount;
            System.out.println("Epoch " + epoch + ": Cost = " + currentCost);
            epoch++;
        }
    }

    // Test the network on a dataset
    public void test(Dataset dataset) {
        int correct = 0;
        for (int i = 0; i < dataset.getSize(); i++) {
            Matrix output = predict(dataset.getDataAtIndex(i));
            int[] sortedIndices = output.sortFlattenedByIndex();
            int[] expectedIndices = dataset.getOutputAtIndex(i).sortFlattenedByIndex();
            if (sortedIndices[0] == expectedIndices[0]) {
                correct++;
            }
        }

        // Calculate percentage and print results
        double percentage = ((double) correct / dataset.getSize()) * 100;
        String formattedPercentage = String.format("%.2f%%", percentage);
        System.out.println("Test Results: " + correct + "/" + dataset.getSize() + " = " + formattedPercentage);
    }

    public List<Layer> getLayers() {
        return layers;
    }

    public CostFunction getCostFunction() {
        return costFunction;
    }

    public boolean isAllowAugmenting() {
        return allowAugmenting;
    }
}