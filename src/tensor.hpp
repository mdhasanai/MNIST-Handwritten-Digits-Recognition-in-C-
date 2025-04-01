#pragma once // include guard to prevent multiple inclusions of the same header file

#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>
#include <fstream>
#include <type_traits>

template<class T>
constexpr bool is_arithmetic_v = std::is_arithmetic<T>::value; 

template<class T>
struct is_arithmetic {
    // This is a compile-time type checker
    static_assert(is_arithmetic_v<T>, "Template parameter must be an arithmetic type.");
};

// (using Arithmetic ) Creates a type alias named Arithmetic
// SFINAE Principle: Substitution Failure Is Not An Error
// If the condition is true, the type alias is created
// If the condition is false, the type alias is not created
template<class T>
using Arithmetic = typename std::enable_if<is_arithmetic_v<T>, T>::type;

template<typename ComponentType, typename = Arithmetic<ComponentType>>
class Tensor {
    public:
        Tensor() : shape_({}), data_({0}) {}
        // explicit forces direct initialization instead of copy initialization
        explicit Tensor(const std::vector<size_t>& shape) : shape_(shape) {
            size_t num_elements = 1;
            for (auto dim : shape) num_elements *= dim;
            data_.resize(num_elements, 0);
        }

        Tensor(const std::vector<size_t>& shape, const ComponentType& fillValue) : shape_(shape) {
            size_t num_elements = 1;
            for (auto dim : shape) num_elements *= dim;
            data_.resize(num_elements, fillValue); // with filled values
        }

        Tensor(const std::vector<size_t>& shape, const std::vector<ComponentType>& data) : shape_(shape), data_(data) {
            size_t num_elements = 1;
            for (auto dim : shape) num_elements *= dim;
            if (data.size() != num_elements) {
                throw std::invalid_argument("size not match.");
            }
        }
        // Copy constructor
        Tensor(const Tensor<ComponentType>& other) : shape_(other.shape_), data_(other.data_) {}

        // Move constructor
        Tensor(Tensor<ComponentType>&& other) noexcept
            : shape_(std::move(other.shape_)), data_(std::move(other.data_)) {
            other.shape_ = {};
            other.data_ = {0};
        }
        // Copy assignment: t1 = t2; -> Calls the this operator
        Tensor& operator=(const Tensor& other) {
            if (this != &other) {
                shape_ = other.shape_;
                data_ = other.data_;
            }
            return *this;
        }
        // Move assignment:
        Tensor& operator=(Tensor<ComponentType>&& other) noexcept {
            if (this != &other) {
                shape_ = std::move(other.shape_);
                data_ = std::move(other.data_);
                other.shape_ = {};
                other.data_ = {0};
            }
            return *this;
        }
        // Destructor
        ~Tensor() = default;
        // Warns if return value is ignored.
        [[nodiscard]] size_t rank() const { return shape_.size(); }
        // Returns shape
        [[nodiscard]] std::vector<size_t> shape() const { return shape_; }
        // Returns total number of elements
        [[nodiscard]] size_t numElements() const { return data_.size(); }
        // Access element (const)
        const ComponentType& operator()(const std::vector<size_t>& idx) const {
            return data_[computeIndex(idx)];
        }
        // Access element (non-const)
        ComponentType& operator()(const std::vector<size_t>& idx) {
            return data_[computeIndex(idx)];
        }
        // Sets data manually (for file I/O compatibility)
        void setData(const std::vector<ComponentType>& new_data) {
            if (new_data.size() != data_.size()) {
                throw std::invalid_argument("New data size does not match the current data size.");
            }
            data_ = new_data;
        }
        // Equality operator: Friend: Mechanism by which a class grants access to its nonpublic.is has access to all private and protected members of the class.
        friend bool operator==(const Tensor<ComponentType>& a, const Tensor<ComponentType>& b) {
            return a.shape_ == b.shape_ && a.data_ == b.data_;
        }
    public:
        // Returns a const reference to the tensor data
        const std::vector<ComponentType>& getData() const {
            return data_;
        }
    private:
        std::vector<size_t> shape_;
        std::vector<ComponentType> data_;
        // Compute flat index. function behaves as a "read-only" operation.
        size_t computeIndex(const std::vector<size_t>& idx) const {
            size_t flat_index = 0, stride = 1;
            for (int i = static_cast<int>(idx.size()) - 1; i >= 0; --i) {
                flat_index += idx[i] * stride;
                stride *= shape_[i];
            }
            return flat_index;
        }
};
// Reads a tensor from file
template<typename ComponentType>
Tensor<ComponentType> readTensorFromFile(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }
    size_t rank;
    file >> rank; // First Rank (# of Dimension) is Input
    std::vector<size_t> shape(rank);
    for (size_t i = 0; i < rank; ++i) file >> shape[i]; // Shape of each dimension.
    Tensor<ComponentType> tensor(shape);
    std::vector<ComponentType> temp_data(tensor.numElements());
    for (size_t i = 0; i < tensor.numElements(); ++i) file >> temp_data[i]; // elements of the row
    tensor.setData(temp_data);
    return tensor;
}
// Writes a tensor to file
template<typename ComponentType>
void writeTensorToFile(const Tensor<ComponentType>& tensor, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }
    file << tensor.rank() << " "; // Writes rank (# of dimensions)
    for (auto dim : tensor.shape()) file << dim << " "; // Writes Shape (Shape of each dimension)
    for (auto elem : tensor.getData()) file << elem << " "; // Write Elements 
}

// Overload operator<< for Tensor template
template<typename ComponentType>
std::ostream& operator<<(std::ostream& out, const Tensor<ComponentType>& tensor) {
    out << tensor.rank() << "\n";
    for (size_t i = 0; i < tensor.rank(); ++i) {
        out << tensor.shape()[i] << "\n";
    }
    for (size_t i = 0; i < tensor.numElements(); ++i) {
        out << tensor.getData()[i] << "\n";
    }
    return out;
}