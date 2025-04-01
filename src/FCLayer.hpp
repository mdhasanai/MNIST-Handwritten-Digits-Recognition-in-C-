#include <iostream>
#include <fstream>
#include <random>
#include <cmath>
#include <vector>
#include "dataset_utils.hpp"
#include "Eigen/Dense"

class FCLayer {
private:
    // Vector-based implementation
    std::vector<std::vector<double>> weights;
    std::vector<double> biases;
    std::vector<double> input;
    std::vector<double> output;
    // Eigen-based implementation
    Eigen::MatrixXd eigen_weights;
    Eigen::VectorXd eigen_biases;
    Eigen::MatrixXd input_tensor;
    size_t input_size;
    size_t output_size;
    
public:
    FCLayer(size_t input_size, size_t output_size, unsigned seed = 42) 
        : input_size(input_size), output_size(output_size) {
        // RNG for vector implementation
        std::mt19937 rng(seed);
        // Xavier initialization
        std::uniform_real_distribution<double> dist(-1.0 / std::sqrt(input_size), 
                                                   1.0 / std::sqrt(input_size));
        // Initialize weights and biases for vector implementation
        weights.resize(output_size, std::vector<double>(input_size));
        biases.resize(output_size, 0.0);
        for(size_t i = 0; i < output_size; ++i) {
            for(size_t j = 0; j < input_size; ++j) {
                weights[i][j] = dist(rng);
            }
        }
        // Initialize Eigen weights and biases
        double range = 1.0 / std::sqrt(input_size);
        eigen_weights = Eigen::MatrixXd::Random(input_size, output_size);
        eigen_weights = eigen_weights * range;
        // Initialize biases to zeros
        eigen_biases = Eigen::VectorXd::Zero(output_size);
    }

    std::vector<double> forward(const std::vector<double>& input_data) {
        input = input_data;
        // output = W * input + b
        output = matvec(weights, input);
        for(size_t i = 0; i < output_size; ++i) {
            output[i] += biases[i];
        }
        return output;
    }
    
    Eigen::MatrixXd forwardND(const Eigen::MatrixXd& input_data) {
        // Save input tensor for backward-pass
        input_tensor = input_data;
        // Compute output: (input * weights) + biases
        Eigen::MatrixXd output = input_data * eigen_weights;
        // Add biases to each row
        for (int i = 0; i < output.rows(); i++) {
            output.row(i) += eigen_biases.transpose();
        }
        return output;
    }
    
    std::vector<double> backward(const std::vector<double>& grad_output, double learning_rate) {
        // Compute gradients for weights and biases
        std::vector<std::vector<double>> weights_grad(output_size, std::vector<double>(input_size, 0.0));
        std::vector<double> biases_grad = grad_output;
        for(size_t i = 0; i < output_size; ++i) {
            for(size_t j = 0; j < input_size; ++j) {
                weights_grad[i][j] = grad_output[i] * input[j];
            }
        }
        // Compute gradient for previous layer grad = W^T * grad_output
        std::vector<double> grad_input(input_size, 0.0);
        for(size_t i = 0; i < input_size; ++i) {
            double sum = 0.0;
            for(size_t j = 0; j < output_size; ++j) {
                sum += weights[j][i] * grad_output[j];
            }
            grad_input[i] = sum;
        }
        // Update weights and biases: W = W - lr * grad, b = b - lr * grad
        for(size_t i = 0; i < output_size; ++i) {
            for(size_t j = 0; j < input_size; ++j) {
                weights[i][j] -= learning_rate * weights_grad[i][j];
            }
        }
        for(size_t i = 0; i < output_size; ++i) {
            biases[i] -= learning_rate * biases_grad[i];
        }
        return grad_input;
    }
    
    // Eigen-based backward pass
    Eigen::MatrixXd backwardND(const Eigen::MatrixXd& error_tensor, double learning_rate) {
        // Compute weight gradients
        Eigen::MatrixXd gradient_weights = input_tensor.transpose() * error_tensor;
        // Compute bias gradients (sum across all samples)
        Eigen::VectorXd gradient_biases = error_tensor.colwise().sum();
        // Update weights: W = W - lr * grad
        eigen_weights = eigen_weights - learning_rate * gradient_weights;
        // Update biases: b = b - lr * grad
        eigen_biases = eigen_biases - learning_rate * gradient_biases;
        // Compute gradient for previous layer
        Eigen::MatrixXd grad_input = error_tensor * eigen_weights.transpose();
        return grad_input;
    }
    
    std::vector<double> getOutput() const {
        return output;
    }
};