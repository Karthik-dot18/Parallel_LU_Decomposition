#include <iostream>
#include <vector>
#include <cstdlib>
#include <iomanip>
#include <algorithm>
#include <omp.h>

void lu_blocked(std::vector<double>& A, int N, int bs) {
    int nb = (N + bs - 1) / bs;
    
    for (int kk = 0; kk < nb; kk++) {
        int k0 = kk * bs;
        int k1 = std::min(k0 + bs, N);

        // Step 1: Sequential Diagonal Tile Factorization
        for (int k = k0; k < k1; k++) {
            double inv = 1.0 / A[k*N + k];
            for (int i = k + 1; i < k1; i++) {
                A[i*N + k] *= inv;
            }
            for (int i = k + 1; i < k1; i++) {
                for (int j = k + 1; j < k1; j++) {
                    A[i*N + j] -= A[i*N + k] * A[k*N + j];
                }
            }
        }

        // Step 2: Parallel Column Panel Update
        #pragma omp parallel for schedule(dynamic, 1)
        for (int ii = kk + 1; ii < nb; ii++) {
            int i0 = ii * bs;
            int i1 = std::min(i0 + bs, N);
            for (int k = k0; k < k1; k++) {
                double inv = 1.0 / A[k*N + k];
                for (int i = i0; i < i1; i++) A[i*N + k] *= inv;
                for (int i = i0; i < i1; i++) {
                    for (int j = k + 1; j < k1; j++) {
                        A[i*N + j] -= A[i*N + k] * A[k*N + j];
                    }
                }
            }
        }

        // Step 3: Parallel Row Panel Update
        #pragma omp parallel for schedule(dynamic, 1)
        for (int jj = kk + 1; jj < nb; jj++) {
            int j0 = jj * bs;
            int j1 = std::min(j0 + bs, N);
            for (int k = k0; k < k1; k++) {
                for (int i = k + 1; i < k1; i++) {
                    for (int j = j0; j < j1; j++) {
                        A[i*N + j] -= A[i*N + k] * A[k*N + j];
                    }
                }
            }
        }

        // Step 4: Parallel Trailing Submatrix Update
        #pragma omp parallel for schedule(dynamic, 1) collapse(2)
        for (int ii = kk + 1; ii < nb; ii++) {
            for (int jj = kk + 1; jj < nb; jj++) {
                int i0 = ii * bs;
                int i1 = std::min(i0 + bs, N);
                int j0 = jj * bs;
                int j1 = std::min(j0 + bs, N);
                
                for (int k = k0; k < k1; k++) {
                    for (int i = i0; i < i1; i++) {
                        double lik = A[i*N + k];
                        for (int j = j0; j < j1; j++) {
                            A[i*N + j] -= lik * A[k*N + j];
                        }
                    }
                }
            }
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <matrix_N> <threads> <block_size>\n";
        return 1;
    }
    int N = std::atoi(argv[1]);
    int num_t = std::atoi(argv[2]);
    int bs = std::atoi(argv[3]);
    omp_set_num_threads(num_t);

    std::vector<double> A(N * N, 1.0);
    for(int i=0; i<N; ++i) A[i*N + i] = N * 2.0; 

    double start = omp_get_wtime();
    lu_blocked(A, N, bs);
    double end = omp_get_wtime();

    std::cout << fixed << setprecision(2);
    std::cout << "Blocked Parallel LU Execution Time (" << num_t << " threads, BS=" << bs << "): " 
              << (end - start) * 1000.0 << " ms\n";
    return 0;
}
