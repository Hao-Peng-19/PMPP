#include "datatypes.h"
#include "matrix_reader.h"
#include "cpu_brute_force.h"
#include "timer.h"

#include <set>
#include <iostream>
#include <filesystem>

#define MAX_QUBO_SIZE 30

// ======== GPU only includes ========
#ifdef HAVE_CUDA
#include "cuda_debug.h"
#include "cuda_timer.h"
#include "gpu_brute_force.h"
#endif
// ===================================

int main() {

#ifdef HAVE_CUDA
    auto gpus = get_gpu_info();
    for (const auto & gpu : gpus)
    {
        print_gpu_info(gpu);
    }
#else
    std::cout << "[INFO] CUDA not available → running CPU-only version.\n";
#endif

    // scan directory for matrix files:
    std::string matrix_path = "./data";

    bool exists = false;
    try {
        exists = std::filesystem::exists(matrix_path);
    } catch (...) {}

    if (!exists)
    {
        matrix_path = "../data";
        try {
            exists = std::filesystem::exists(matrix_path);
        } catch (...) { exists = false; }
    }

    if (!exists)
    {
        std::cerr << "Data directory not found (./data or ../data)!\n";
        return -1;
    }

    std::vector<std::string> matrix_files;
    for (auto & entry : std::filesystem::directory_iterator(matrix_path))
        if (entry.path().extension() == ".mtx")
            matrix_files.push_back(entry.path().string());

    // Brute forcers
    auto dense_brute_forcer  = CPUQUBOBruteForcer<IndexType, ValueType, StateType, DenseMatrix<ValueType>>();
    auto sparse_brute_forcer = CPUQUBOBruteForcer<IndexType, ValueType, StateType, SparseMatrix<ValueType, IndexType>>();

#ifdef HAVE_CUDA
    auto cuda_dense_brute_forcer  = GPUQUBOBruteForcer<IndexType, ValueType, StateType, DenseMatrix<ValueType>>();
    auto cuda_sparse_brute_forcer = GPUQUBOBruteForcer<IndexType, ValueType, StateType, SparseMatrix<ValueType, IndexType>>();
#endif

    for (auto & file : matrix_files)
    {
        auto sparse_matrix = readMatrixMarket<ValueType, IndexType>(file);
        if (sparse_matrix.rows > MAX_QUBO_SIZE)
            continue;

        std::cout << "############################################################\n";
        std::cout << "Matrix: " << file << "\n";
        std::cout << sparse_matrix.rows << " x " << sparse_matrix.cols 
                  << ", nnz = " << sparse_matrix.nnz << "\n\n";

        double density = double(sparse_matrix.nnz) / double(sparse_matrix.rows * sparse_matrix.cols);

        std::vector<std::vector<StateType>> cpu_results;
        std::set<size_t> cpu_binary_results;

        std::vector<std::vector<StateType>> gpu_results; // always defined

        if (density >= 0.5)
        {
            std::cout << "**CPU (dense):\n";
            auto dense_matrix = sparse_to_dense(sparse_matrix);

            {
                Timer t("CPU dense brute force");
                cpu_results = dense_brute_forcer.brute_force_optima(dense_matrix);
            }

#ifdef HAVE_CUDA
            std::cout << "\n**GPU (dense):\n";
            {
                CudaTimer timer("GPU dense brute force");
                gpu_results = cuda_dense_brute_forcer.brute_force_optima(dense_matrix);
                timer.stop();
                timer.wait_for_time();
            }
#else
            std::cout << "[GPU disabled - skipped]\n";
#endif
        }
        else
        {
            std::cout << "**CPU (sparse):\n";
            {
                Timer t("CPU sparse brute force");
                cpu_results = sparse_brute_forcer.brute_force_optima(sparse_matrix);
            }

#ifdef HAVE_CUDA
            std::cout << "\n**GPU (sparse):\n";
            {
                CudaTimer timer("GPU sparse brute force");
                gpu_results = cuda_sparse_brute_forcer.brute_force_optima(sparse_matrix);
                timer.stop();
                timer.wait_for_time();
            }
#else
            std::cout << "[GPU disabled - skipped]\n";
#endif
        }

        // Results printing (CPU always present)
        std::cout << "\nCPU Optimal states:\n";
        if (cpu_results.empty()) std::cout << "<Empty>\n";

        for (int i = 0; i < cpu_results.size(); i++)
        {
            const auto & state = cpu_results[i];
            std::cout << i << ": ";
            for (auto bit : state) std::cout << int(bit) << " ";

            std::cout << "Energy: " << compute_energy<ValueType>(sparse_matrix, state.data());

            auto repr = state_vector_to_binary_reprensentation(state);
            if (cpu_binary_results.count(repr))
                std::cout << " - DUPLICATE!";
            cpu_binary_results.insert(repr);

            std::cout << "\n";
        }

#ifdef HAVE_CUDA
        std::cout << "\nGPU Optimal states:\n";
        if (gpu_results.empty()) std::cout << "<Empty>\n";

        int identical = 0;
        for (int i = 0; i < gpu_results.size(); i++)
        {
            const auto & state = gpu_results[i];
            std::cout << i << ": ";
            for (auto bit : state) std::cout << int(bit) << " ";

            std::cout << "Energy: " << compute_energy<ValueType>(sparse_matrix, state.data());

            auto repr = state_vector_to_binary_reprensentation(state);
            if (!cpu_binary_results.count(repr))
                std::cout << " [WRONG! NOT FOUND IN CPU RESULTS]";
            else
                identical++;

            std::cout << "\n";
        }
        std::cout << "Identical CPU/GPU states: " << identical << " / " << cpu_results.size() << "\n";
#endif

        std::cout << "############################################################\n\n";
    }

    return 0;
}
