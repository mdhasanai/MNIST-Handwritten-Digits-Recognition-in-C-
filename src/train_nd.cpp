#include <iostream>
#include <fstream>
#include <random>
#include <cmath>
#include <string>
#include <vector>
#include <chrono>
// #include <omp.h>
#include "FCLayer.hpp"
#include "ReLULayer.hpp"
#include "SoftmaxLayer.hpp"
#include "dataset_utils.hpp"
#include "NetworkConfig.hpp"

// Performance monitoring class
class Timer {
private:
// Using chrono for high resolution time measurement to track time for clocks, time_point, duration
    std::chrono::high_resolution_clock::time_point start_time;
    std::string name;
public:
    Timer(const std::string& name) : name(name) {
        start_time = std::chrono::high_resolution_clock::now();
    }
    ~Timer() {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
        // std::cout << name << " took " << duration << " ms" << std::endl;
    }
};

// Neural Network with Eigen
class NeuralNetwork {
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
        
        Eigen::MatrixXd forwardND(const Eigen::MatrixXd& input) {
            // Forward pass through each layer
            Eigen::MatrixXd fc1_output = fc1.forwardND(input);
            Eigen::MatrixXd relu_output = relu.forwardND(fc1_output);
            Eigen::MatrixXd fc2_output = fc2.forwardND(relu_output);
            Eigen::MatrixXd softmax_output = softmax.forwardND(fc2_output);
            return softmax_output;
        }
        
        void backwardND(const Eigen::MatrixXd& target) {
            // Compute initial gradient (softmax output - target)
            Eigen::MatrixXd softmax_grad = softmax.getOutputND() - target;
            
            // Backward pass through each layer
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
    
    int main(int argc, char* argv[]) {
        if(argc != 2) {
            std::cerr << "Usage: " << argv[0] << " <config_file>" << std::endl;
            return 1;
        }
        
        try {
            Timer total_timer("Total execution");
            
            // Read configuration
            NetworkConfig config = read_config(argv[1]);
            
            // Initialize neural network
            NeuralNetwork nn(784, config.hidden_size, 10, config.learning_rate, 42);
            
            // Initialize dataloaders for training
            Dataloader train_images_loader(config.batch_size);
            Dataloader train_labels_loader(config.batch_size);
            
            std::cout << "Loading training data...\n";
            train_images_loader.read_image_data(config.train_images_path);
            train_labels_loader.read_labels_data(config.train_labels_path);
            
            size_t num_batches = train_images_loader.get_num_batches();
            std::cout << "Loaded " << num_batches << " batches of training data\n";
            
            std::cout << "Training started...\n";
            
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
                    
                    // Get batch data
                    Eigen::MatrixXd image_batch = train_images_loader.get_data_batch(batch_idx);
                    Eigen::MatrixXd label_batch = train_labels_loader.get_data_batch(batch_idx);
                    
                    // Forward pass
                    Eigen::MatrixXd predicted = nn.forwardND(image_batch);
                    
                    // Calculate loss
                    epoch_loss += nn.calculateLossND(label_batch);
                    
                    // Backward pass
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
            Timer test_timer("Testing");
            
            // Initialize dataloaders for testing
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
                // Get batch data
                Eigen::MatrixXd image_batch = test_images_loader.get_data_batch(batch_idx);
                Eigen::MatrixXd label_batch = test_labels_loader.get_data_batch(batch_idx);
                
                // Predict
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
            std::cout << "Test Accuracy: " << test_accuracy << "%" << std::endl;
            
        } catch (const std::exception& e) {
            std::cerr << "Error: " << e.what() << std::endl;
            return 1;
        }
        
        return 0;
    }