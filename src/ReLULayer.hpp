#include <iostream>
#include <random>
#include <cmath>
#include <vector>
#include "dataset_utils.hpp"
#include "Eigen/Dense"

class ReLULayer {
private:
    std::vector<double> input;
    std::vector<double> output;
    Eigen::MatrixXd input_tensor_cache;
    
public:
    ReLULayer() {}
    
    std::vector<double> forward(const std::vector<double>& input_data) {
        input = input_data;
        size_t size = input.size();
        output.resize(size);
        // out = max(0, in)
        for(size_t i = 0; i < size; ++i) {
            output[i] = std::max(0.0, input[i]);
        }
        return output;
    }
    
    Eigen::MatrixXd forwardND(const Eigen::MatrixXd& input_tensor) {
        // Cache input for backward pass
        input_tensor_cache = input_tensor;
        // Element-wise max with 0
        Eigen::MatrixXd output = input_tensor.cwiseMax(0.0);
        return output;
    }
    
    std::vector<double> backward(const std::vector<double>& grad_output, double learning_rate) {
        size_t size = input.size();
        std::vector<double> grad_input(size);
        // grad_in = grad_out * (in > 0 ? 1 : 0)
        for(size_t i = 0; i < size; ++i) {
            grad_input[i] = grad_output[i] * (input[i] > 0 ? 1.0 : 0.0);
        }
        return grad_input;
    }
    
    Eigen::MatrixXd backwardND(const Eigen::MatrixXd& error_tensor, double learning_rate) {
        // Create mask where input > 0
        Eigen::MatrixXd mask = (input_tensor_cache.array() >= 0.0).cast<double>();
        // Element-wise multiplication of error tensor with mask
        Eigen::MatrixXd grad_input = error_tensor.array() * mask.array();
        return grad_input;
    }
    std::vector<double> getOutput() const {
        return output;
    }
};