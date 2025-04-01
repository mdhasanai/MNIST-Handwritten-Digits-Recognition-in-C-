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

// Neural Network
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
    
        std::vector<double> forward(const std::vector<double>& input) {
            // Forward pass through each layer
            std::vector<double> fc1_output = fc1.forward(input);
            std::vector<double> relu_output = relu.forward(fc1_output);
            std::vector<double> fc2_output = fc2.forward(relu_output);
            std::vector<double> softmax_output = softmax.forward(fc2_output);
            return softmax_output;
        }
        
        void backward(const std::vector<double>& target) {
            // Backward pass through each layer
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
                
                // Get batch data and convert to vector format
                auto batch_data = train_images_loader.get_batch_data_as_vec(batch_idx);
                auto label_batch = train_labels_loader.get_data_batch(batch_idx);
                
                for(size_t i = 0; i < batch_data.size(); ++i) {
                    // Get image
                    const auto& image = batch_data[i];
                    
                    // Get one-hot encoded label
                    std::vector<double> target(10, 0.0);
                    for (int j = 0; j < 10; ++j) {
                        target[j] = label_batch(i, j);
                    }
                    
                    // Forward pass
                    std::vector<double> predicted = nn.forward(image);
                    
                    // Calculate loss
                    epoch_loss += nn.calculateLoss(target);
                    
                    // Backward pass
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
            
            // Early stopping for debugging
            // if (epoch == 1) {
            //     break;
            // }
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
            // Get batch data and convert to vector format
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
        std::cout << "Test Accuracy: " << test_accuracy << "%" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}