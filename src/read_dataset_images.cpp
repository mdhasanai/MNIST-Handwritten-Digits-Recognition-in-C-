#include <iostream>
#include <fstream>
#include <vector>
#include "dataset_utils.hpp"

int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <image_dataset_input> <image_tensor_output> <image_index>" << std::endl;
        return 1;
    }
    std::string image_dataset_input = argv[1];
    std::string image_tensor_output = argv[2];
    int image_index = std::stoi(argv[3]);

    // Read the image data from the dataset and write it to a tensor file
    Dataloader dataloader(100);
    dataloader.read_image_data(image_dataset_input);
    dataloader.write_images_to_file(image_tensor_output, image_index);
    try {
        dataloader.read_image_data(image_dataset_input);
        dataloader.write_images_to_file(image_tensor_output, image_index);
        // Get a specific batch
        // Eigen::MatrixXd batch = dataloader.get_data_batch(0);
        // // Get a specific row
        // Eigen::VectorXd row = batch.row(0);
        // // Convert the Eigen::VectorXd to std::vector<double>
        // std::vector<double> image_data(row.data(), row.data() + row.size());
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}