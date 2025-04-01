#pragma once
#include <vector>
#include <iostream>
#include "tensor.hpp"
#include "Eigen/Dense"
#include <vector>
#include <iostream>
#include "tensor.hpp"
#include "Eigen/Dense"

class Dataloader {
private:
    size_t batch_size;
    size_t class_count;
    size_t dim_y;
    size_t dim_x;
    std::vector<Eigen::MatrixXd> dataset;

    int decode_int32(std::ifstream& src) {
        char bytes[4];
        src.read(bytes, 4);
        if (src.gcount() != 4) {
            throw std::runtime_error("Failed reading 32-bit value");
        }
        std::reverse(bytes, bytes + 4);
        int value;
        std::memcpy(&value, bytes, sizeof(int));
        return value;
    }

    std::ifstream load_file(const std::string& path) {
        std::ifstream file(path, std::ios::binary);
        if (!file) {
            throw std::runtime_error("Cannot access: " + path);
        }
        return file;
    }

public:
    Dataloader(): batch_size(1) {}
    Dataloader(size_t size): batch_size(size) {}

    void read_image_data(const std::string& path) {
        auto file = load_file(path);
        decode_int32(file); // Skip magic
        size_t sample_count = decode_int32(file); // Number of images
        dim_y = decode_int32(file); // Height
        dim_x = decode_int32(file); // Width
        size_t pixels = dim_y * dim_x;
        size_t remainder = sample_count % batch_size;
        std::vector<unsigned char> raw(pixels);
        std::vector<double> normalized(pixels);
        Eigen::MatrixXd batch(batch_size, pixels);

        for (size_t i = 0; i < sample_count; ++i) {
            file.read(reinterpret_cast<char*>(raw.data()), pixels);
            for (size_t j = 0; j < pixels; ++j) {
                normalized[j] = static_cast<double>(raw[j]) / 255.0;
            }
            batch.row(i % batch_size) = Eigen::Map<Eigen::VectorXd>(normalized.data(), pixels);
            if ((i + 1) % batch_size == 0) {
                dataset.push_back(batch);
            }
            else if (i == sample_count - 1) {
                size_t last_size = remainder;
                dataset.push_back(batch.block(0, 0, last_size, pixels));
            }
        }
    }

    void read_labels_data(const std::string& path) {
        auto file = load_file(path);
        decode_int32(file); // Skip magic
        class_count = decode_int32(file); // Number of labels
        Eigen::MatrixXd one_hot(batch_size, 10);
        one_hot.setZero();
        
        for (size_t i = 0; i < class_count; ++i) {
            uint8_t label_val;
            file.read(reinterpret_cast<char*>(&label_val), 1);
            one_hot(i % batch_size, static_cast<int>(label_val)) = 1;
            if ((i + 1) % batch_size == 0 || i == class_count - 1) {
                dataset.push_back(one_hot);
                one_hot.setZero();
            }
        }
    }

    Eigen::MatrixXd get_data_batch(const size_t& idx) {
        return dataset[idx];
    }

    size_t get_num_batches() const {
        return dataset.size();
    }

    std::vector<std::vector<double>> get_batch_data_as_vec(size_t batch_idx) const {
        const Eigen::MatrixXd& batch = dataset.at(batch_idx);
        std::vector<std::vector<double>> batch_data;
        for (int i = 0; i < batch.rows(); ++i) {
            std::vector<double> row(batch.row(i).data(), batch.row(i).data() + batch.row(i).size());
            batch_data.push_back(row);
        }
        return batch_data;
    }

    // Get raw label values from one-hot encoded matrix
    std::vector<int> get_raw_labels() const {
        std::vector<int> labels;
        for (size_t batch_idx = 0; batch_idx < dataset.size(); ++batch_idx) {
            const Eigen::MatrixXd& batch = dataset[batch_idx];
            for (int i = 0; i < batch.rows(); ++i) {
                for (int j = 0; j < batch.cols(); ++j) {
                    if (batch(i, j) > 0.5) { // Threshold for one-hot encoding
                        labels.push_back(j);
                        break;
                    }
                }
            }
        }
        return labels;
    }

    void write_images_to_file(const std::string& path, const size_t& idx) const {
        size_t batch_idx = idx / batch_size;
        size_t item_idx = idx % batch_size;
        std::ofstream out(path);
        std::cout << "Writing to: " << path << std::endl;
        if (!out) {
            std::cerr << "Failed to write: " << path << std::endl;
            return;
        }
        out << 2 << "\n" << dim_y << "\n" << dim_x << "\n";
        size_t pixels = dim_y * dim_x;
        std::cout << "Total pixels: " << pixels << std::endl;
        for (size_t i = 0; i < pixels; ++i) {
            out << dataset.at(batch_idx)(item_idx, i) << "\n";
        }
    }

    void write_labels_to_file(const std::string& path, const size_t& idx) {
        std::ofstream out(path);
        if (!out) {
            std::cerr << "Failed to write: " << path << std::endl;
            return;
        }
        out << 1 << "\n" << 10 << "\n";
        size_t batch_idx = idx / batch_size;
        size_t item_idx = idx % batch_size;
        for (size_t i = 0; i < 10; ++i) {
            out << dataset[batch_idx](item_idx, i) << "\n";
        }
    }

    void clear() {
        dataset.clear();
    }
};


// Helper function to read MNIST images
std::vector<double> read_mnist_image(std::ifstream& file) {
    std::vector<double> image_data(784);
    unsigned char pixel;
    for(size_t i = 0; i < 784; ++i) {
        file.read(reinterpret_cast<char*>(&pixel), 1);
        image_data[i] = static_cast<double>(pixel) / 255.0;  // Normalize to [0,1]
    }
    return image_data;
}

// Function to write tensor to file
void write_tensor_to_file(const std::string& file_path, const Tensor<double>& tensor) {
    std::ofstream file(file_path);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + file_path);
    }
    file << tensor;
}

// Function to read MNIST label data
int read_mnist_label(const std::string& file_path, int label_index) {
    std::ifstream file(file_path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + file_path);
    }
    file.seekg(8 + label_index); //Skip the header
    unsigned char label;
    file.read(reinterpret_cast<char*>(&label), sizeof(label));
    return static_cast<int>(label);
}

// Helper function to create one-hot encoded vector
std::vector<double> create_one_hot(unsigned char label) {
    std::vector<double> one_hot(10, 0.0);
    one_hot[label] = 1.0;
    return one_hot;
}

std::vector<double> read_mnist_image(const std::string& file_path, int image_index) {
    std::ifstream file(file_path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + file_path);
    }
    // Skip the header. MNIST files have a 16-byte header. Seeks directly to the start of the requested image
    file.seekg(16 + image_index * 28 * 28);
    std::vector<double> image(28 * 28); // Creates a vector to hold 784 normalized pixel values
    for (int i = 0; i < 28 * 28; ++i) {
        unsigned char pixel;
        file.read(reinterpret_cast<char*>(&pixel), sizeof(pixel)); // Uses reinterpret_cast to read raw bytes safely
        image[i] = static_cast<double>(pixel) / 255.0;
    }
    return image;
}

// Matrix-vector multiplication
std::vector<double> matvec(const std::vector<std::vector<double>>& weights, const std::vector<double>& input) {
    size_t output_size = weights.size();
    size_t input_size = input.size();
    std::vector<double> result(output_size, 0.0);
    // #pragma omp parallel for
    for (size_t i = 0; i < output_size; ++i) {
        double sum = 0.0;
        for (size_t j = 0; j < input_size; ++j) {
            sum += weights[i][j] * input[j];
        }
        result[i] = sum;
    }
    return result;
}