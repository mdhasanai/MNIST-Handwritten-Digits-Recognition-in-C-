#pragma once

#include "tensor.hpp"

template< typename ComponentType >
class Vector
{
public:
    // Default-constructor.
    Vector() = default;

    // Constructor for vector of certain size.
    //explicit Vector(size_t size);
    explicit Vector(size_t size) : tensor_({size}) {}

    // Constructor for vector of certain size with constant fill-value.
    //Vector(size_t size, const ComponentType& fillValue);
    Vector(size_t size, const ComponentType& fillValue) : tensor_({size}, fillValue) {}

    // Constructing vector from file.
    //Vector(const std::string& filename);
    Vector(const std::string& filename) : tensor_(readTensorFromFile<ComponentType>(filename)) {}

    // Number of elements in this vector.
    //[[nodiscard]] size_t size() const;
    [[nodiscard]] size_t size() const { return tensor_.numElements(); }

    // Element access function
    //const ComponentType&
    //operator()(size_t idx) const;
    const ComponentType& operator()(size_t idx) const { return tensor_({idx}); }

    // Element mutation function
    //ComponentType&
    //operator()(size_t idx);
    ComponentType& operator()(size_t idx) { return tensor_({idx}); }

    // Reference to internal tensor.
    //Tensor< ComponentType >& tensor();
    Tensor<ComponentType>& tensor() { return tensor_; }
    const Tensor<ComponentType>& tensor() const { return tensor_; }

private:
    Tensor< ComponentType > tensor_;
};

template< typename ComponentType >
class Matrix
{
public:
    // Default-constructor.
    Matrix() = default;

    // Constructor for matrix of certain size.
    //explicit Matrix(size_t rows, size_t cols);
    explicit Matrix(size_t rows, size_t cols) : tensor_({rows, cols}) {}

    // Constructor for matrix of certain size with constant fill-value.
    //Matrix(size_t rows, size_t cols, const ComponentType& fillValue);
    Matrix(size_t rows, size_t cols, const ComponentType& fillValue) : tensor_({rows, cols}, fillValue) {}

    // Constructing matrix from file.
    //Matrix(const std::string& filename);
    Matrix(const std::string& filename) : tensor_(readTensorFromFile<ComponentType>(filename)) {}

    // Number of rows.
    //[[nodiscard]] size_t rows() const;
    [[nodiscard]] size_t rows() const { return tensor_.shape()[0]; }

    // Number of columns.
    //[[nodiscard]] size_t cols() const;
    [[nodiscard]] size_t cols() const { return tensor_.shape()[1]; }

    // Element access function
    const ComponentType&
    //operator()(size_t row, size_t col) const;
    operator()(size_t row, size_t col) const { return tensor_({row, col}); }


    // Element mutation function
    ComponentType&
    //operator()(size_t row, size_t col);
    operator()(size_t row, size_t col) { return tensor_({row, col}); }

    // Reference to internal tensor.
    //Tensor< ComponentType >& tensor();
    Tensor<ComponentType>& tensor() { return tensor_; }
    const Tensor<ComponentType>& tensor() const { return tensor_; }

private:
    Tensor< ComponentType > tensor_;
};

// TODO: Implement all methods.


// Performs a matrix-vector multiplication.
template< typename ComponentType >
Vector< ComponentType > matvec(const Matrix< ComponentType >& mat, const Vector< ComponentType >& vec)
{
    // TODO: Implement this.

    if (mat.cols() != vec.size()) {
        throw std::runtime_error("Matrix columns must match vector size for multiplication");
    }
    
    Vector<ComponentType> result(mat.rows(), 0);
    for (size_t i = 0; i < mat.rows(); ++i) {
        ComponentType sum = 0;
        for (size_t j = 0; j < mat.cols(); ++j) {
            sum += mat(i, j) * vec(j);
        }
        result(i) = sum;
    }
    return result;

}

