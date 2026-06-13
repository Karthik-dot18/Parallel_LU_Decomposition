#include <iostream>
#include <vector>
#include <cstdlib>

void assemble_heat_matrix(std::vector<double>& A, int grid_n) {
    int N = grid_n * grid_n;
    double lam = 0.25;
    
    for(int i = 0; i < N * N; i++) {
        A[i] = 0.0;
    }

    for (int i = 0; i < grid_n; i++) {
        for (int j = 0; j < grid_n; j++) {
            int p = i * grid_n + j;
            A[p*N + p] = 1.0 + 4.0 * lam; 
            
            if (j > 0)          A[p*N + (p-1)]      = -lam;
            if (j < grid_n-1)   A[p*N + (p+1)]      = -lam;
            if (i > 0)          A[p*N + (p-grid_n)] = -lam;
            if (i < grid_n-1)   A[p*N + (p+grid_n)] = -lam;
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <grid_n>\n";
        return 1;
    }
    int grid_n = std::atoi(argv[1]);
    int N = grid_n * grid_n;
    
    std::vector<double> A(N * N);
    assemble_heat_matrix(A, grid_n);
    
    std::cout << "Successfully generated a "<<N<<"x"<<N<<" system matrix for a "<<grid_n<<"x"<<grid_n<<" heat grid.\n";
    return 0;
}
