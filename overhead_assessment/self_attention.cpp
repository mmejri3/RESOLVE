/*#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <algorithm>
#include <cstdlib>  // For std::atoi

using namespace std;

// Function to compute matrix multiplication
vector<vector<float>> matmul(const vector<vector<float>>& A, const vector<vector<float>>& B) {
    int n = A.size(); // rows of A
    int m = B[0].size(); // columns of B
    int k = A[0].size(); // columns of A, rows of B

    vector<vector<float>> C(n, vector<float>(m, 0));

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            for (int x = 0; x < k; ++x) {
                C[i][j] += A[i][x] * B[x][j];
            }
        }
    }

    return C;
}

// Function to compute the transpose of a matrix
vector<vector<float>> transpose(const vector<vector<float>>& A) {
    int n = A.size();    // rows of A
    int m = A[0].size(); // columns of A

    vector<vector<float>> T(m, vector<float>(n, 0));

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            T[j][i] = A[i][j];
        }
    }

    return T;
}

// Softmax function applied row-wise
vector<vector<float>> softmax(const vector<vector<float>>& A) {
    vector<vector<float>> S = A;
    int n = A.size();
    int m = A[0].size();

    for (int i = 0; i < n; ++i) {
        float maxVal = *max_element(A[i].begin(), A[i].end());
        float sum = 0.0;

        for (int j = 0; j < m; ++j) {
            S[i][j] = exp(A[i][j] - maxVal);  // subtract max for numerical stability
            sum += S[i][j];
        }

        for (int j = 0; j < m; ++j) {
            S[i][j] /= sum;
        }
    }

    return S;
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        cerr << "Usage: " << argv[0] << " <matrix_size> <vector_size>" << endl;
        return 1;
    }

    int matrix_size = std::atoi(argv[1]);
    int vector_size = std::atoi(argv[2]);

    // Input X of shape (matrix_size, vector_size)
    vector<vector<float>> X(matrix_size, vector<float>(vector_size, 1.0));  // Example values

    // Weights W_Q, W_K, W_V of shape (vector_size, vector_size)
    vector<vector<float>> W_Q(vector_size, vector<float>(vector_size, 0.5)); // Example values
    vector<vector<float>> W_K(vector_size, vector<float>(vector_size, 0.5)); // Example values
    vector<vector<float>> W_V(vector_size, vector<float>(vector_size, 0.5)); // Example values


    for (int i(0); i<100; ++i) {
        // Step 1: Compute Q = X * W_Q, K = X * W_K, V = X * W_V
        vector<vector<float>> Q = matmul(X, W_Q);
        vector<vector<float>> K = matmul(X, W_K);
        vector<vector<float>> V = matmul(X, W_V);

        // Step 2: Compute K^T (transpose of K)
        vector<vector<float>> K_T = transpose(K);

        // Step 3: Compute S = softmax(Q * K^T)
        vector<vector<float>> S = softmax(matmul(Q, K_T));

        // Step 4: Compute O = S * V
        vector<vector<float>> O = matmul(S, V);    
    
    }
    return 0;
}*/


#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <algorithm>
#include <cstdlib>  // For std::atoi

using namespace std;

void matrix_operations(int matrix_size, int vector_size) {
    // Function to compute matrix multiplication
    auto matmul = [](const vector<vector<float>>& A, const vector<vector<float>>& B) {
        int n = A.size();  // rows of A
        int m = B[0].size();  // columns of B
        int k = A[0].size();  // columns of A, rows of B

        vector<vector<float>> C(n, vector<float>(m, 0));

        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < m; ++j) {
                for (int x = 0; x < k; ++x) {
                    C[i][j] += A[i][x] * B[x][j];
                }
            }
        }

        return C;
    };

    // Function to compute the transpose of a matrix
    auto transpose = [](const vector<vector<float>>& A) {
        int n = A.size();  // rows of A
        int m = A[0].size();  // columns of A

        vector<vector<float>> T(m, vector<float>(n, 0));

        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < m; ++j) {
                T[j][i] = A[i][j];
            }
        }

        return T;
    };

    // Input X of shape (matrix_size, vector_size)
    vector<vector<float>> X(matrix_size, vector<float>(vector_size, 1.0));  // Example values

    // Weights W_Q, W_K, W_V of shape (vector_size, vector_size)
    vector<vector<float>> W_Q(vector_size, vector<float>(vector_size, 0.5));  // Example values
    vector<vector<float>> W_K(vector_size, vector<float>(vector_size, 0.5));  // Example values
    vector<vector<float>> W_V(vector_size, vector<float>(vector_size, 0.5));  // Example values

    for (int i = 0; i < 100; ++i) {
        // Step 1: Compute Q = X * W_Q, K = X * W_K, V = X * W_V
        vector<vector<float>> Q = matmul(X, W_Q);
        vector<vector<float>> K = matmul(X, W_K);
        vector<vector<float>> V = matmul(X, W_V);

        // Step 2: Compute K^T (transpose of K)
        vector<vector<float>> K_T = transpose(K);

        // Step 3: Compute S = Q * K^T (no softmax function)
        vector<vector<float>> S = matmul(Q, K_T);

        // Step 4: Compute O = S * V
        vector<vector<float>> O = matmul(S, V);
    }
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        cerr << "Usage: " << argv[0] << " <matrix_size> <vector_size>" << endl;
        return 1;
    }

    int matrix_size = std::atoi(argv[1]);
    int vector_size = std::atoi(argv[2]);

    // Call the function to perform matrix operations
    matrix_operations(matrix_size, vector_size);

    return 0;
}


