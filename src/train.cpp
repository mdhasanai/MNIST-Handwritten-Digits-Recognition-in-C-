#include <iostream>
#include <fstream>
#include <random>
#include <cmath>
#include <string>
#include <vector>
#include <chrono>
#include "FCLayer.hpp"
#include "ReLULayer.hpp"
#include "SoftmaxLayer.hpp"
#include "dataset_utils.hpp"
#include "NetworkConfig.hpp"

class Timer {
private:
    std::chrono::high_resolution_clock::time_point start_time;
    std::string name;
public:
    Timer(const std::string& name) : name(name) {
        start_time = std::chrono::high_resolution_clock::now();
    }
    ~Timer() {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
        std::cout << name << " took " << duration << " ms" << std::endl;
    }
};

class NeuralNetwork {
    // Neural Network with vector-based implementation
private:
    FCLayer fc1;
    ReLULayer relu;
    FCLayer fc2;
    SoftmaxLayer softmax;
    double learning_rate;
public:
    NeuralNetwork(size_t input_size, size_t hidden_size, size_t output_size, 
                 double learning_rate, unsigned seed = 42)
        : fc1(input_size, hidden_size, seed),
          relu(),
          fc2(hidden_size, output_size, seed+1),
          softmax(),
          learning_rate(learning_rate) {}
    
    std::vector<double> forward(const std::vector<double>& input) {
        std::vector<double> fc1_output = fc1.forward(input);
        std::vector<double> relu_output = relu.forward(fc1_output);
        std::vector<double> fc2_output = fc2.forward(relu_output);
        std::vector<double> softmax_output = softmax.forward(fc2_output);
        return softmax_output;
    }
    
    void backward(const std::vector<double>& target) {
        std::vector<double> softmax_grad = softmax.backward(target);
        std::vector<double> fc2_grad = fc2.backward(softmax_grad, learning_rate);
        std::vector<double> relu_grad = relu.backward(fc2_grad, learning_rate);
        std::vector<double> fc1_grad = fc1.backward(relu_grad, learning_rate);
    }
    
    double calculateLoss(const std::vector<double>& target) {
        return softmax.calculateLoss(target);
    }
    
    size_t predict(const std::vector<double>& input) {
        std::vector<double> output = forward(input);
        size_t pred_label = 0;
        double max_prob = output[0];
        for(size_t j = 1; j < output.size(); ++j) {
            if(output[j] > max_prob) {
                max_prob = output[j];
                pred_label = j;
            }
        }
        return pred_label;
    }
};

class NeuralNetworkND {
    // Neural Network with Eigen-based implementation
private:
    FCLayer fc1;
    ReLULayer relu;
    FCLayer fc2;
    SoftmaxLayer softmax;
    double learning_rate;
    
public:
    NeuralNetworkND(size_t input_size, size_t hidden_size, size_t output_size, 
                  double learning_rate, unsigned seed = 42)
        : fc1(input_size, hidden_size, seed),
          relu(),
          fc2(hidden_size, output_size, seed+1),
          softmax(),
          learning_rate(learning_rate) {}
    
    Eigen::MatrixXd forwardND(const Eigen::MatrixXd& input) {
        Eigen::MatrixXd fc1_output = fc1.forwardND(input);
        Eigen::MatrixXd relu_output = relu.forwardND(fc1_output);
        Eigen::MatrixXd fc2_output = fc2.forwardND(relu_output);
        Eigen::MatrixXd softmax_output = softmax.forwardND(fc2_output);
        return softmax_output;
    }
    
    void backwardND(const Eigen::MatrixXd& target) {
        // Compute initial gradient (softmax output - target)
        Eigen::MatrixXd softmax_grad = softmax.getOutputND() - target;
        Eigen::MatrixXd fc2_grad = fc2.backwardND(softmax_grad, learning_rate);
        Eigen::MatrixXd relu_grad = relu.backwardND(fc2_grad, learning_rate);
        Eigen::MatrixXd fc1_grad = fc1.backwardND(relu_grad, learning_rate);
    }
    
    double calculateLossND(const Eigen::MatrixXd& target) {
        return softmax.calculateLossND(target);
    }
    
    Eigen::VectorXi predictND(const Eigen::MatrixXd& input) {
        Eigen::MatrixXd output = forwardND(input);
        Eigen::VectorXi predictions(output.rows());
        for (int i = 0; i < output.rows(); ++i) {
            Eigen::MatrixXd::Index maxRow, maxCol;
            output.row(i).maxCoeff(&maxRow, &maxCol);
            predictions(i) = maxCol;
        }
        return predictions;
    }
};

void run_vectorized(const NetworkConfig& config) {
    // Function to train and test using vectorized implementation
    Timer total_timer("Total vectorized execution");
    NeuralNetwork nn(784, config.hidden_size, 10, config.learning_rate, 42);
    Dataloader train_images_loader(config.batch_size);
    Dataloader train_labels_loader(config.batch_size);
    std::cout << "Loading training data...\n";
    train_images_loader.read_image_data(config.train_images_path);
    train_labels_loader.read_labels_data(config.train_labels_path);
    
    size_t num_batches = train_images_loader.get_num_batches();
    std::cout << "Loaded " << num_batches << " batches of training data\n";
    std::cout << "Training started (vectorized)...\n";
    // Training loop
    for(int epoch = 0; epoch < config.num_epochs; ++epoch) {
        Timer epoch_timer("Epoch " + std::to_string(epoch + 1));
        size_t correct = 0;
        size_t total = 0;
        double epoch_loss = 0.0;
        for(size_t batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
            if (batch_idx % 100 == 0) {
                std::cout << "Epoch " << epoch + 1 << " Batch: " << batch_idx << "/" << num_batches << "\n";
            }
            // Get batch data and convert to vector format
            auto batch_data = train_images_loader.get_batch_data_as_vec(batch_idx);
            auto label_batch = train_labels_loader.get_data_batch(batch_idx);
            for(size_t i = 0; i < batch_data.size(); ++i) {
                const auto& image = batch_data[i];
                // Get one-hot encoded label
                std::vector<double> target(10, 0.0);
                for (int j = 0; j < 10; ++j) {
                    target[j] = label_batch(i, j);
                }
                std::vector<double> predicted = nn.forward(image);
                epoch_loss += nn.calculateLoss(target);
                nn.backward(target);
                // Calculate accuracy
                size_t pred_label = nn.predict(image);
                size_t true_label = 0;
                for (size_t j = 0; j < 10; ++j) {
                    if (label_batch(i, j) > 0.5) {
                        true_label = j;
                        break;
                    }
                }
                if(pred_label == true_label) correct++;
                total++;
            }
        }
        std::cout << "Epoch " << epoch + 1 << "/" << config.num_epochs 
                  << ", Loss: " << epoch_loss / total 
                  << ", Accuracy: " << (100.0 * correct / total) << "%" << std::endl;
    }
    // Clear training data to free memory
    train_images_loader.clear();
    train_labels_loader.clear();

    // Testing phase
    Timer test_timer("Testing (vectorized)");
    Dataloader test_images_loader(config.batch_size);
    Dataloader test_labels_loader(config.batch_size);
    
    std::cout << "Loading test data...\n";
    test_images_loader.read_image_data(config.test_images_path);
    test_labels_loader.read_labels_data(config.test_labels_path);
    size_t test_num_batches = test_images_loader.get_num_batches();
    std::cout << "Loaded " << test_num_batches << " batches of test data\n";
    std::ofstream log_file(config.log_file_path);
    if(!log_file) {
        throw std::runtime_error("Error opening log file");
    }
    
    size_t test_correct = 0;
    size_t test_total = 0;
    for(size_t batch_idx = 0; batch_idx < test_num_batches; ++batch_idx) {
        auto batch_data = test_images_loader.get_batch_data_as_vec(batch_idx);
        auto label_batch = test_labels_loader.get_data_batch(batch_idx);
        log_file << "Current batch: " << batch_idx << "\n";
        for(size_t i = 0; i < batch_data.size(); ++i) {
            const auto& image = batch_data[i];
            // Get true label
            size_t true_label = 0;
            for (size_t j = 0; j < 10; ++j) {
                if (label_batch(i, j) > 0.5) {
                    true_label = j;
                    break;
                }
            }
            size_t pred_label = nn.predict(image);
            if(pred_label == true_label) test_correct++;
            test_total++;
            log_file << " - image " << (batch_idx * config.batch_size + i) 
                     << ": Prediction=" << pred_label 
                     << ". Label=" << true_label << "\n";
        }
    }
    double test_accuracy = 100.0 * test_correct / test_total;
    std::cout << "Test Accuracy (vectorized): " << test_accuracy << "%" << std::endl;
}

void run_eigen(const NetworkConfig& config) {
    // Function to train and test using Eigen-based implementation
    Timer total_timer("Total Eigen execution");
    NeuralNetworkND nn(784, config.hidden_size, 10, config.learning_rate, 42);
    Dataloader train_images_loader(config.batch_size);
    Dataloader train_labels_loader(config.batch_size);
    
    std::cout << "Loading training data...\n";
    train_images_loader.read_image_data(config.train_images_path);
    train_labels_loader.read_labels_data(config.train_labels_path);
    size_t num_batches = train_images_loader.get_num_batches();
    std::cout << "Loaded " << num_batches << " batches of training data\n";
    std::cout << "Training started (Eigen)...\n";
    // Training loop
    for(int epoch = 0; epoch < config.num_epochs; ++epoch) {
        Timer epoch_timer("Epoch " + std::to_string(epoch + 1));
        size_t correct = 0;
        size_t total = 0;
        double epoch_loss = 0.0;
        
        for(size_t batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
            if (batch_idx % 100 == 0) {
                std::cout << "Epoch " << epoch + 1 << " Batch: " << batch_idx << "/" << num_batches << "\n";
            }
            Eigen::MatrixXd image_batch = train_images_loader.get_data_batch(batch_idx);
            Eigen::MatrixXd label_batch = train_labels_loader.get_data_batch(batch_idx);
            Eigen::MatrixXd predicted = nn.forwardND(image_batch);
            epoch_loss += nn.calculateLossND(label_batch);
            nn.backwardND(label_batch);
            // Calculate accuracy
            Eigen::VectorXi predictions = nn.predictND(image_batch);
            for (int i = 0; i < predictions.size(); ++i) {
                // Find true label from one-hot encoding
                int true_label = -1;
                for (int j = 0; j < label_batch.cols(); ++j) {
                    if (label_batch(i, j) > 0.5) {
                        true_label = j;
                        break;
                    }
                }
                if (predictions(i) == true_label) {
                    correct++;
                }
                total++;
            }
        }
        std::cout << "Epoch " << epoch + 1 << "/" << config.num_epochs 
                  << ", Loss: " << epoch_loss / total 
                  << ", Accuracy: " << (100.0 * correct / total) << "%" << std::endl;
    }
    // Clear training data to free memory
    train_images_loader.clear();
    train_labels_loader.clear();
    
    // Testing phase
    Timer test_timer("Testing (Eigen)");
    Dataloader test_images_loader(config.batch_size);
    Dataloader test_labels_loader(config.batch_size);
    
    std::cout << "Loading test data...\n";
    test_images_loader.read_image_data(config.test_images_path);
    test_labels_loader.read_labels_data(config.test_labels_path);

    size_t test_num_batches = test_images_loader.get_num_batches();
    std::cout << "Loaded " << test_num_batches << " batches of test data\n";
    std::ofstream log_file(config.log_file_path);
    if(!log_file) {
        throw std::runtime_error("Error opening log file");
    }
    
    size_t test_correct = 0;
    size_t test_total = 0;
    
    for(size_t batch_idx = 0; batch_idx < test_num_batches; ++batch_idx) {
        Eigen::MatrixXd image_batch = test_images_loader.get_data_batch(batch_idx);
        Eigen::MatrixXd label_batch = test_labels_loader.get_data_batch(batch_idx);
        Eigen::VectorXi predictions = nn.predictND(image_batch);
        log_file << "Current batch: " << batch_idx << "\n";
        for(int i = 0; i < predictions.size(); ++i) {
            // Find true label from one-hot encoding
            int true_label = -1;
            for (int j = 0; j < label_batch.cols(); ++j) {
                if (label_batch(i, j) > 0.5) {
                    true_label = j;
                    break;
                }
            }
            if (predictions(i) == true_label) {
                test_correct++;
            }
            test_total++;
            log_file << " - image " << (batch_idx * config.batch_size + i) 
                     << ": Prediction=" << predictions(i) 
                     << ". Label=" << true_label << "\n";
        }
    }
    double test_accuracy = 100.0 * test_correct / test_total;
    std::cout << "Test Accuracy (Eigen): " << test_accuracy << "%" << std::endl;
}

int main(int argc, char* argv[]) {
    if(argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <config_file>" << std::endl;
        return 1;
    }
    try {
        // Read configuration
        NetworkConfig config = read_config(argv[1]);
        // training mode
        int mode = 0; //std::stoi(argv[2]);
        if (mode == 0) {
            std::cout << "Running vectorized implementation..." << std::endl;
            run_vectorized(config);
            std::cout << "Done vectorized implementation!" << std::endl;
        } else if (mode == 1) {
            std::cout << "Running Eigen implementation..." << std::endl;
            run_eigen(config);
            std::cout << "Done Eigen implementation!" << std::endl;
        } else {
            std::cerr << "Invalid mode: " << mode << std::endl;
            std::cerr << "  mode: 0 for vectorized, 1 for Eigen" << std::endl;
            return 1;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}