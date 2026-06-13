#include <iostream>
#include <vector>
#include <cstdlib>
#include <iomanip>
#include <omp.h>

void lu_naive(std::vector<double>& A, int N) {
    for (int k = 0; k < N; k++) {
        double inv = 1.0 / A[k*N + k];
        
        #pragma omp parallel for schedule(static)
        for (int i = k + 1; i < N; i++) {
            A[i*N + k] *= inv;
        }
        
        #pragma omp parallel for schedule(static) collapse(2)
        for (int i = k + 1; i < N; i++) {
            for (int j = k + 1; j < N; j++) {
                A[i*N + j] -= A[i*N + k] * A[k*N + j];
            }
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <matrix_N> <threads>\n";
        return 1;
    }
    int N = std::atoi(argv[1]);
    int num_t = std::atoi(argv[2]);
    omp_set_num_threads(num_t);

    // Mock full dense matrix initialized for raw benchmark
    std::vector<double> A(N * N, 1.0);
    for(int i=0; i<N; ++i) A[i*N + i] = N * 2.0; 

    double start = omp_get_wtime();
    lu_naive(A, N);
    double end = omp_get_wtime();

    std::cout << fixed << setprecision(2);
    std::cout << "Naive Parallel LU Execution Time (" << num_t << " threads): " 
              << (end - start) * 1000.0 << " ms\n";
    return 0;
}
