#include <iostream>
#include <fstream>
#include <random>
#include <cmath>
#include <vector>
#include "Eigen/Dense"

class SoftmaxLayer {
private:
    std::vector<double> input;
    std::vector<double> output;
    Eigen::MatrixXd input_tensor_cache;
    Eigen::MatrixXd y_hat;
    
public:
    SoftmaxLayer() {}
    
    std::vector<double> forward(const std::vector<double>& input_data) {
        input = input_data;
        size_t size = input.size();
        output.resize(size);
        // Find max value for numerical stability
        double max_val = input[0];
        for(size_t i = 1; i < size; ++i) {
            max_val = std::max(max_val, input[i]);
        }
        // Compute exp(x - max) and sum
        double sum_exp = 0.0;
        std::vector<double> exp_values(size);
        for(size_t i = 0; i < size; ++i) {
            exp_values[i] = std::exp(input[i] - max_val);
            sum_exp += exp_values[i];
        }
        // Compute softmax: exp(x - max) / sum(exp(x - max))
        for(size_t i = 0; i < size; ++i) {
            output[i] = exp_values[i] / sum_exp;
        }
        return output;
    }
    
    Eigen::MatrixXd forwardND(const Eigen::MatrixXd& input_tensor) {
        // Cache input for backward pass
        input_tensor_cache = input_tensor;
        // Calculate the softmax of the input tensor
        // Shift by max value for numerical stability
        auto shifted = input_tensor.colwise() - input_tensor.rowwise().maxCoeff();
        Eigen::MatrixXd exp_tensor = shifted.array().exp();
        y_hat = exp_tensor.array().colwise() / exp_tensor.array().rowwise().sum();
        return y_hat;
    }
    
    std::vector<double> backward(const std::vector<double>& target) {
        size_t size = output.size();
        std::vector<double> grad_input(size);
        // gradient = output - target
        for(size_t i = 0; i < size; ++i) {
            grad_input[i] = output[i] - target[i];
        }
        return grad_input;
    }
    
    Eigen::MatrixXd backwardND(const Eigen::MatrixXd& error_tensor, double learning_rate = 0.01) {
        // Calculate the gradient of the loss with respect to the input
        // Calculate weighted sum of errors for each sample
        Eigen::MatrixXd weighted_error_sum = (error_tensor.array() * y_hat.array()).rowwise().sum();
        // Broadcast the sum back to the original shape to subtract it from each error element
        Eigen::MatrixXd adjusted_error = error_tensor.array() - 
                                        (weighted_error_sum.replicate(1, error_tensor.cols())).array();
        // Perform the final element-wise multiplication with y_hat
        Eigen::MatrixXd grad_input = y_hat.array() * adjusted_error.array();
        return grad_input;
    }
    std::vector<double> getOutput() const {
        return output;
    }
    Eigen::MatrixXd getOutputND() const {
        return y_hat;
    }
    
    double calculateLoss(const std::vector<double>& target) {
        double loss = 0.0;
        size_t size = output.size();
        for(size_t i = 0; i < size; ++i) {
            if(target[i] > 0) {
                loss -= target[i] * std::log(std::max(output[i], 1e-7));
            }
        }
        return loss;
    }
    
    double calculateLossND(const Eigen::MatrixXd& target) {
        // Cross-entropy loss
        Eigen::MatrixXd clipped_output = y_hat.array().max(1e-7);
        Eigen::MatrixXd log_probs = clipped_output.array().log();
        // Element-wise multiplication and sum
        double loss = -(target.array() * log_probs.array()).sum() / target.rows();
        return loss;
    }
};