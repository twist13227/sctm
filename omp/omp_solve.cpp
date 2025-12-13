#include <iostream>
#include <vector>
#include <cmath>
#include <omp.h>
#include <algorithm>
#include <chrono>
#include <cstring>

int N_PLUS_1 = 0;

double u_analytical(double x, double Lx, double y, double Ly, double z, double Lz, double a_t, double t) {
    return sin(3.0 * M_PI * x / Lx) * sin(2.0 * M_PI * y / Ly) * sin(2.0 * M_PI * z / Lz) * cos(a_t * t + 4.0 * M_PI);
}

inline int index(int i, int j, int k) {
    return i * N_PLUS_1 * N_PLUS_1 + j * N_PLUS_1 + k;
}

double laplacian_x_component(int i, int j, int k, const std::vector<double> &u, double hx_sq, int N) {
    if (i == 0) return (u[index(1, j, k)] - 2.0 * u[index(0, j, k)]) / hx_sq;
    if (i == N - 1) return (-2.0 * u[index(N-1, j, k)] + u[index(N-2, j, k)]) / hx_sq;
    return (u[index(i+1, j, k)] - 2.0 * u[index(i, j, k)] + u[index(i-1, j, k)]) / hx_sq;
}

double laplacian_y_component(int i, int j, int k, const std::vector<double> &u, double hy_sq, int N) {
    int j_prev = (j - 1 + N) % N;
    int j_next = (j + 1) % N;
    return (u[index(i, j_next, k)] - 2.0 * u[index(i, j, k)] + u[index(i, j_prev, k)]) / hy_sq;
}

double laplacian_z_component(int i, int j, int k, const std::vector<double> &u, double hz_sq, int N) {
    int k_prev = (k - 1 + N) % N;
    int k_next = (k + 1) % N;
    return (u[index(i, j, k_next)] - 2.0 * u[index(i, j, k)] + u[index(i, j, k_prev)]) / hz_sq;
}

double compute_full_laplacian(int i, int j, int k, const std::vector<double> &u, double hx_sq, double hy_sq, double hz_sq, int N) {
    double lap_x = laplacian_x_component(i, j, k, u, hx_sq, N);
    double lap_y = laplacian_y_component(i, j, k, u, hy_sq, N);
    double lap_z = laplacian_z_component(i, j, k, u, hz_sq, N);
    return lap_x + lap_y + lap_z;
}

void time_step_evolution(
    const std::vector<double> &u_old,
    const std::vector<double> &u_current,
    std::vector<double> &u_new,
    int N,
    double hx_sq, double hy_sq, double hz_sq,
    double tau, double a2,
    bool is_first_step
) {
    #pragma omp parallel for collapse(3)
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            for (int k = 0; k < N; ++k) {
                if (i == 0 || i == N - 1) {
                    u_new[index(i, j, k)] = 0.0;
                    continue;
                }
                
                double lap = compute_full_laplacian(i, j, k, u_current, hx_sq, hy_sq, hz_sq, N);
                double coef = is_first_step ? 0.5 : 1.0;
                double correction = a2 * tau * tau * lap;
                
                if (is_first_step) {
                    u_new[index(i, j, k)] = u_old[index(i, j, k)] + coef * correction;
                } else {
                    u_new[index(i, j, k)] = 2.0 * u_current[index(i, j, k)] - u_old[index(i, j, k)] + correction;
                }
            }
        }
    }
}

double calculate_max_error(const std::vector<double> &u_curr, int N, double hx, double hy, double hz, 
                          int TIME_STEPS, double tau, double Lx, double Ly, double Lz, double a_t) {
    double error = 0.0;
    #pragma omp parallel for collapse(3) reduction(max : error)
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            for (int k = 0; k < N; ++k) {
                double x = i * hx;
                double y = j * hy;
                double z = k * hz;
                double t = TIME_STEPS * tau;
                double u_exact = u_analytical(x, Lx, y, Ly, z, Lz, a_t, t);
                double diff = fabs(u_curr[index(i, j, k)] - u_exact);
                error = std::max(error, diff);
            }
        }
    }
    return error;
}

int main(int argc, char *argv[])
{
    auto start = std::chrono::high_resolution_clock::now();

    if (argc < 3) {
        std::cerr << "Expected 2 arguments: N and L (1 or PI)\n";
        return 1;
    }

    int N = std::atoi(argv[1]);
    std::string L_str = argv[2];
    double L;

    if (L_str == "1") {
        L = 1.0;
    } else if (L_str == "PI") {
        L = M_PI;
    } else {
        std::cerr << "Error: L must be either 1 or PI.\n";
        return 1;
    }

    std::cout << "Using parameters:\n"
            << "  N = " << N << "\n"
            << "  L = " << L << " (" << L_str << ")\n";

    const double Lx = L;
    const double Ly = L;
    const double Lz = L;

    const int TIME_STEPS = 20;
    const double hx = Lx / N;
    const double hy = Ly / N;
    const double hz = Lz / N;
    const double tau = hx / 100.0;

    N += 1;
    N_PLUS_1 = N;
    const double a_t = 2.0 * M_PI;
    const double a2 = 4.0 / (9.0 * Lx * Lx) + 4.0 / (Ly * Ly) + 4.0 / (Lz * Lz);

    const double hx_sq = hx * hx;
    const double hy_sq = hy * hy;
    const double hz_sq = hz * hz;

    std::vector<double> u_prev(N * N * N, 0.0);
    std::vector<double> u_curr(N * N * N, 0.0);
    std::vector<double> u_next(N * N * N, 0.0);

    #pragma omp parallel for collapse(3)
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            for (int k = 0; k < N; ++k) {
                double x = i * hx;
                double y = j * hy;
                double z = k * hz;
                u_prev[index(i, j, k)] = u_analytical(x, Lx, y, Ly, z, Lz, a_t, 0.0);
            }
        }
    }

    time_step_evolution(u_prev, u_prev, u_curr, N, hx_sq, hy_sq, hz_sq, tau, a2, true);

    for (int n = 1; n < TIME_STEPS; ++n) {
        time_step_evolution(u_prev, u_curr, u_next, N, hx_sq, hy_sq, hz_sq, tau, a2, false);

        std::swap(u_prev, u_curr);
        std::swap(u_curr, u_next);

        double max_error = calculate_max_error(u_curr, N, hx, hy, hz, n, tau, Lx, Ly, Lz, a_t);
        if (n < TIME_STEPS - 1) {
            std::cout << "Max error at current step: " << max_error << std::endl;
        } else {
            std::cout << "Final error: " << max_error << std::endl;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Elapsed time: " << elapsed.count() << std::endl;

    return 0;
}
