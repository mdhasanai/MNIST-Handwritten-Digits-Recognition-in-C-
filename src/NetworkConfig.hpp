#include <fstream>
#include <string>
#include <iostream>

struct NetworkConfig {
    std::string train_images_path;
    std::string train_labels_path;
    std::string test_images_path;
    std::string test_labels_path;
    std::string log_file_path;
    int num_epochs;
    int batch_size;
    int hidden_size;
    double learning_rate;
};

NetworkConfig read_config(const std::string& config_file) {
    NetworkConfig config;
    std::ifstream file(config_file);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open config file: " + config_file);
    }
    
    std::string line, key, value;
    while(std::getline(file, line)) {
        // Skip empty lines and comments
        if(line.empty() || line[0] == '/' || line[0] == '#') continue;
        size_t pos = line.find('=');
        if(pos == std::string::npos) continue;
        key = line.substr(0, pos);
        value = line.substr(pos + 1);
        // Remove whitespace
        key.erase(0, key.find_first_not_of(" \t"));
        key.erase(key.find_last_not_of(" \t") + 1);
        value.erase(0, value.find_first_not_of(" \t"));
        value.erase(value.find_last_not_of(" \t") + 1);
        if(key == "rel_path_train_images") config.train_images_path = value;
        else if(key == "rel_path_train_labels") config.train_labels_path = value;
        else if(key == "rel_path_test_images") config.test_images_path = value;
        else if(key == "rel_path_test_labels") config.test_labels_path = value;
        else if(key == "rel_path_log_file") config.log_file_path = value;
        else if(key == "num_epochs") config.num_epochs = std::stoi(value);
        else if(key == "batch_size") config.batch_size = std::stoi(value);
        else if(key == "hidden_size") config.hidden_size = std::stoi(value);
        else if(key == "learning_rate") config.learning_rate = std::stod(value);
    }
    return config;
}