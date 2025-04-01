#include <iostream>
#include <fstream>
#include <vector>
#include "dataset_utils.hpp"

int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <label_dataset_input> <label_tensor_output> <label_index>" << std::endl;
        return 1;
    }
    std::string label_dataset_input = argv[1];
    std::string label_tensor_output = argv[2];
    int label_index = std::stoi(argv[3]);
    Dataloader dataloader(100);
    try {
        dataloader.read_labels_data(label_dataset_input);
        dataloader.write_labels_to_file(label_tensor_output, label_index);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}