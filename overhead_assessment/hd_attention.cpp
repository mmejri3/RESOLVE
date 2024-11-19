/*#include <chrono> 
#include <vector>
#include <cstdint>
#include <algorithm>
#include <random>
#include <iostream>
#include <cstdint>  // For uint32_t and bit manipulation
#include <cstring>  // For memcpy
#include <cstdlib>  // For std::atoi

using namespace std;

// Fast sign function using bit manipulation to extract the sign bit from a float
inline uint32_t get_sign_bit(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(value));  // Reinterpret float as uint32_t
    return (bits >> 31) & 1;  // Extract the sign bit (0 for positive, 1 for negative)
}

// Fast sign multiplication using XOR directly on sign bits
inline int sign_multiply(float a, float b) {
    return 1 - 2 * (get_sign_bit(a) ^ get_sign_bit(b));
}

// Multiply two sign terms using XOR logic for optimization
inline int xor_sign_multiply(int sign1, int sign2) {
    return 1 - 2 * (sign1 ^ sign2);
}


std::vector<float> compute_final_output(const std::vector<float>& output, int output_rows, int W_cols) {
    std::vector<float> S(output_rows * output_rows, 0.0f);  // S is a output_rows x output_rows matrix

    for (int i = 0; i < output_rows; ++i) {
        for (int j = 0; j < output_rows; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < W_cols; ++k) {
                sum += sign_multiply(output[i * W_cols + k], output[i * W_cols + k] + output[j * W_cols + k]);
            }
            S[i * output_rows + j] = sum / W_cols;  // Normalize by W_cols
        }
    }
    std::vector<float> final_output(output_rows * W_cols, 0.0f);

    for (int i = 0; i < output_rows; ++i) {
        for (int j = 0; j < W_cols; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < output_rows; ++k) {
                sum += S[i * output_rows + k] * output[k * W_cols + j];
            }
            final_output[i * W_cols + j] = sum;
        }
    }

    return final_output;
}

void efficient_matrix_multiply(const std::vector<float>& X, 
                               const std::vector<int8_t>& W, 
                               std::vector<float>& output,
                               int X_rows, int X_cols, int W_cols, int output_rows) {

    output.resize(output_rows * W_cols, 0.0f);

    std::vector<int8_t> W_T(W_cols * X_cols);
    for (int k = 0; k < X_cols; ++k) {
        for (int j = 0; j < W_cols; ++j) {
            W_T[j * X_cols + k] = W[k * W_cols + j];
        }
    }

    for (int i = 0; i < output_rows; ++i) {
        int base_X_index = i * 5 * X_cols;  // Precompute base X index for this row
        for (int j = 0; j < W_cols; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < X_cols; ++k) {
                float X_val_0 = X[base_X_index + k];           // X index for m = 0
                float X_val_1 = X[base_X_index + X_cols + k];  // X index for m = 1
                float X_val_2 = X[base_X_index + 2 * X_cols + k];  // X index for m = 2
                float X_val_3 = X[base_X_index + 3 * X_cols + k];  // X index for m = 3
                float X_val_4 = X[base_X_index + 4 * X_cols + k];  // X index for m = 4

                int8_t W_sign = W_T[j * X_cols + k];  // W_T stores +1 or -1
                sum += W_sign * (X_val_0 + X_val_1 + X_val_2 + X_val_3 + X_val_4);
            }
            output[i * W_cols + j] = sum;
        }
    }
}

// Example usage
int main(int argc, char* argv[]) {
    if (argc != 5) {
        cerr << "Usage: " << argv[0] << " <X_rows> <X_cols> <W_cols> <output_rows>" << endl;
        return 1;
    }

    int X_rows = std::atoi(argv[1]);
    int X_cols = std::atoi(argv[2]);
    int W_cols = std::atoi(argv[3]);
    int output_rows = std::atoi(argv[4]);

    std::vector<float> X(X_rows * X_cols, 1.0f);  // Initialize X with all 1's for simplicity
    std::vector<int8_t> W(X_cols * W_cols);  // Initialize W with appropriate dimensions
    std::vector<float> output;

    for (int i = 0; i < 100; i++) {
        efficient_matrix_multiply(X, W, output, X_rows, X_cols, W_cols, output_rows);
        std::vector<float> final_output = compute_final_output(output, output_rows, W_cols);
    }

    return 0;
}*/



#include <iostream>
#include <vector>
#include <cstdint>
#include <cstring>  // For memcpy
#include <cstdlib>  // For std::atoi
#include <cmath>    // For log2

using namespace std;

// Function to compute XOR-based sign product for bipolar values
int xor_sign_product(int x_bits, int y_bits) {
    // XOR between corresponding bits and count matching bits
    int xor_result = ~(x_bits ^ y_bits);
    return __builtin_popcount(xor_result);  // Count the number of matching bits (i.e., 1s in the xor_result)
}

// Main function to handle matrix multiplication and output computation
void process_matrices(const std::vector<int>& X_packed, const std::vector<int>& W_packed, 
                      int X_rows, int X_cols, int W_cols, int output_rows) {

    std::vector<int> output(output_rows * W_cols, 0);  // Fixed-point representation for output
    int bits_per_entry = 4;  // Packing 8 bipolar values in one byte
    int val = (int)(X_rows/output_rows);
    // Compute output matrix using XOR-based sign product
    for (int i = 0; i < output_rows; ++i) {
        int base_X_index = i * val * (X_cols / bits_per_entry);
        for (int j = 0; j < W_cols; ++j) {
            int sum = 0;
            for (int k = 0; k < X_cols / bits_per_entry; ++k) {
                sum += xor_sign_product(X_packed[base_X_index + k], W_packed[j * (X_cols / bits_per_entry) + k]);
            }
            output[i * W_cols + j] = sum;  // Accumulated count of matching bits (indicating sign matches)
        }
    }

    // Calculate the S matrix and apply fixed-point normalization
    int log_dim_plus_2 = (int)(log2(W_cols)) + 2;
    int max_value = (1 << log_dim_plus_2);  // Max value is 2^(log_dim + 2)
    std::vector<int> S(output_rows * output_rows, 0);

    for (int i = 0; i < output_rows; ++i) {
        for (int j = 0; j < output_rows; ++j) {
            int sum = 0;
            for (int k = 0; k < W_cols; ++k) {
                // Combine the sign products using XOR
                sum += xor_sign_product(output[i * W_cols + k], output[j * W_cols + k]);
            }
            S[i * output_rows + j] = sum / W_cols;  // Normalize by dividing by W_cols
        }
    }

    // Compute the final matrix multiplication result
    std::vector<int> final_output(output_rows * W_cols, 0);
    for (int i = 0; i < output_rows; ++i) {
        for (int j = 0; j < W_cols; ++j) {
            int sum = 0;
            for (int k = 0; k < output_rows; ++k) {
                sum += S[i * output_rows + k] * output[k * W_cols + j];
            }
            final_output[i * W_cols + j] = sum;  // Scale down by max_value for fixed-point precision
        }
    }

}

// Utility to pack bipolar values (-1 and 1) into a bitfield
void pack_bipolar_values(const std::vector<int>& source, std::vector<int>& destination, int size) {
    for (int i = 0; i < size; i += 8) {
        int packed = 0;
        for (int bit = 0; bit < 8; ++bit) {
            if (i + bit < size) {
                packed |= ((source[i + bit] > 0) << bit);
            }
        }
        destination[i / 8] = packed;
    }
}

int main(int argc, char* argv[]) {
    if (argc != 5) {
        cerr << "Usage: " << argv[0] << " <X_rows> <X_cols> <W_cols> <output_rows>" << endl;
        return 1;
    }

    int X_rows = std::atoi(argv[1]);
    int X_cols = std::atoi(argv[2]);
    int W_cols = std::atoi(argv[3]);
    int output_rows = std::atoi(argv[4]);

    std::vector<int> X(X_rows * X_cols, 1);  // Initialize X with all 1's (bipolar values)
    std::vector<int> W(X_cols * W_cols, 1);  // Initialize W

    // Pack bipolar values into bitfields
    std::vector<int> X_packed(X_rows * X_cols / 8, 0);
    std::vector<int> W_packed(X_cols * W_cols / 8, 0);
    pack_bipolar_values(X, X_packed, X_rows * X_cols);
    pack_bipolar_values(W, W_packed, X_cols * W_cols);

    for (int i = 0; i < 100; i++) {
        process_matrices(X_packed, W_packed, X_rows, X_cols, W_cols, output_rows);
    }

    return 0;
}

