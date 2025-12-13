#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <mpi.h>
#include <cstring>

int ALLOC_NJ = 0;
int ALLOC_NK = 0;


double u_analytical(double x, double Lx, double y, double Ly, double z, double Lz, double a_t, double t) {
    return sin(3.0 * M_PI * x / Lx) * sin(2.0 * M_PI * y / Ly) * sin(2.0 * M_PI * z / Lz) * cos(a_t * t + 4.0 * M_PI);
}
inline int local_index(int i, int j, int k) {
    return i * ALLOC_NJ * ALLOC_NK + j * ALLOC_NK + k;
}

double compute_full_laplacian(int i, int j, int k,
                         const std::vector<double> &u,
                         double hx, double hy, double hz)
{
    int i_prev = i - 1;
    int i_next = i + 1;
    int j_prev = j - 1;
    int j_next = j + 1;
    int k_prev = k - 1;
    int k_next = k + 1;

    double u_center = u[local_index(i, j, k)];

    double u_i_prev = u[local_index(i_prev, j, k)];
    double u_i_next = u[local_index(i_next, j, k)];

    double u_j_prev = u[local_index(i, j_prev, k)];
    double u_j_next = u[local_index(i, j_next, k)];

    double u_k_prev = u[local_index(i, j, k_prev)];
    double u_k_next = u[local_index(i, j, k_next)];

    double laplacian =
        (u_i_next - 2.0 * u_center + u_i_prev) / (hx * hx) +
        (u_j_next - 2.0 * u_center + u_j_prev) / (hy * hy) +
        (u_k_next - 2.0 * u_center + u_k_prev) / (hz * hz);

    return laplacian;
}


int compute_local_size(int axis_size, int dims_i, int coord_i, int *start_i)
{
    int local_size;
    
    if (coord_i < axis_size % dims_i)
    {
        local_size = axis_size / dims_i + 1;
        *start_i = coord_i * local_size;
    }
    else
    {
        local_size = axis_size / dims_i;
        *start_i = coord_i * local_size + axis_size % dims_i;
    }
    return local_size;
}

void change_bounds(std::vector<double> &u,
                         int local_ni, int local_nj, int local_nk,
                         MPI_Comm cart_comm)
{
    int dims[3], periods[3], coords[3];
    MPI_Cart_get(cart_comm, 3, dims, periods, coords);

    if (dims[1] == 1)
    {
        for (int i = 0; i < local_ni + 2; ++i)
        {
            for (int k = 0; k < local_nk + 2; ++k)
            {
                u[local_index(i, 0, k)] = u[local_index(i, local_nj, k)];
                u[local_index(i, local_nj + 1, k)] = u[local_index(i, 1, k)];
            }
        }
    }
    
    if (dims[2] == 1)
    {
        for (int i = 0; i < local_ni + 2; ++i)
        {
            for (int j = 0; j < local_nj + 2; ++j)
            {
                u[local_index(i, j, 0)] = u[local_index(i, j, local_nk)];
                u[local_index(i, j, local_nk + 1)] = u[local_index(i, j, 1)];
            }
        }
    }
    
    if (dims[0] <= 1 && dims[1] <= 1 && dims[2] <= 1)
        return;
    
    int neighbor_left, neighbor_right;
    int neighbor_down, neighbor_up;
    int neighbor_back, neighbor_front;
    
    MPI_Cart_shift(cart_comm, 0, 1, &neighbor_left, &neighbor_right);
    MPI_Cart_shift(cart_comm, 1, 1, &neighbor_down, &neighbor_up);
    MPI_Cart_shift(cart_comm, 2, 1, &neighbor_back, &neighbor_front);
    
    MPI_Request requests[12];
    int request_count = 0;

    std::vector<double> send_buffer_left, recv_buffer_left, send_buffer_right, recv_buffer_right;
    if (dims[0] > 1)
    {
        int nj_k_size = (local_nj + 2) * (local_nk + 2);
        send_buffer_left.resize(nj_k_size);
        recv_buffer_left.resize(nj_k_size);
        send_buffer_right.resize(nj_k_size);
        recv_buffer_right.resize(nj_k_size);

        int idx = 0;
        for (int j = 0; j < local_nj + 2; ++j)
        {
            for (int k = 0; k < local_nk + 2; ++k)
            {
                send_buffer_left[idx] = u[local_index(1, j, k)];
                send_buffer_right[idx] = u[local_index(local_ni, j, k)];
                ++idx;
            }
        }

        MPI_Isend(send_buffer_left.data(), nj_k_size, MPI_DOUBLE, 
                  neighbor_left, 0, cart_comm, &requests[request_count++]);
        MPI_Irecv(recv_buffer_right.data(), nj_k_size, MPI_DOUBLE, 
                  neighbor_right, 0, cart_comm, &requests[request_count++]);

        MPI_Isend(send_buffer_right.data(), nj_k_size, MPI_DOUBLE, 
                  neighbor_right, 1, cart_comm, &requests[request_count++]);
        MPI_Irecv(recv_buffer_left.data(), nj_k_size, MPI_DOUBLE, 
                  neighbor_left, 1, cart_comm, &requests[request_count++]);
    }

    std::vector<double> send_buffer_down, recv_buffer_down, send_buffer_up, recv_buffer_up;
    if (dims[1] > 1)
    {
        int ni_k_size = (local_ni + 2) * (local_nk + 2);
        send_buffer_down.resize(ni_k_size);
        recv_buffer_down.resize(ni_k_size);
        send_buffer_up.resize(ni_k_size);
        recv_buffer_up.resize(ni_k_size);

        int idx = 0;
        for (int i = 0; i < local_ni + 2; ++i)
        {
            for (int k = 0; k < local_nk + 2; ++k)
            {
                send_buffer_down[idx] = u[local_index(i, 1, k)];
                send_buffer_up[idx] = u[local_index(i, local_nj, k)];
                ++idx;
            }
        }

        MPI_Isend(send_buffer_down.data(), ni_k_size, MPI_DOUBLE, 
                  neighbor_down, 2, cart_comm, &requests[request_count++]);
        MPI_Irecv(recv_buffer_up.data(), ni_k_size, MPI_DOUBLE, 
                  neighbor_up, 2, cart_comm, &requests[request_count++]);

        MPI_Isend(send_buffer_up.data(), ni_k_size, MPI_DOUBLE, 
                  neighbor_up, 3, cart_comm, &requests[request_count++]);
        MPI_Irecv(recv_buffer_down.data(), ni_k_size, MPI_DOUBLE, 
                  neighbor_down, 3, cart_comm, &requests[request_count++]);
    }

    std::vector<double> send_buffer_back, recv_buffer_back, send_buffer_front, recv_buffer_front;
    if (dims[2] > 1)
    {
        int ni_j_size = (local_ni + 2) * (local_nj + 2);
        send_buffer_back.resize(ni_j_size);
        recv_buffer_back.resize(ni_j_size);
        send_buffer_front.resize(ni_j_size);
        recv_buffer_front.resize(ni_j_size);

        int idx = 0;
        for (int i = 0; i < local_ni + 2; ++i)
        {
            for (int j = 0; j < local_nj + 2; ++j)
            {
                send_buffer_back[idx] = u[local_index(i, j, 1)];
                send_buffer_front[idx] = u[local_index(i, j, local_nk)];
                ++idx;
            }
        }
        
        MPI_Isend(send_buffer_back.data(), ni_j_size, MPI_DOUBLE, 
                  neighbor_back, 4, cart_comm, &requests[request_count++]);
        MPI_Irecv(recv_buffer_front.data(), ni_j_size, MPI_DOUBLE, 
                  neighbor_front, 4, cart_comm, &requests[request_count++]);

        MPI_Isend(send_buffer_front.data(), ni_j_size, MPI_DOUBLE, 
                  neighbor_front, 5, cart_comm, &requests[request_count++]);
        MPI_Irecv(recv_buffer_back.data(), ni_j_size, MPI_DOUBLE, 
                  neighbor_back, 5, cart_comm, &requests[request_count++]);
    }
    
    if (request_count > 0)
    {
        MPI_Waitall(request_count, requests, MPI_STATUSES_IGNORE);
    }

    if (dims[0] > 1)
    {
        int idx = 0;
        for (int j = 0; j < local_nj + 2; ++j)
        {
            for (int k = 0; k < local_nk + 2; ++k)
            {
                u[local_index(local_ni + 1, j, k)] = recv_buffer_right[idx];
                u[local_index(0, j, k)] = recv_buffer_left[idx];
                ++idx;
            }
        }
    }
    
    if (dims[1] > 1)
    {
        int idx = 0;
        for (int i = 0; i < local_ni + 2; ++i)
        {
            for (int k = 0; k < local_nk + 2; ++k)
            {
                u[local_index(i, local_nj + 1, k)] = recv_buffer_up[idx];
                u[local_index(i, 0, k)] = recv_buffer_down[idx];
                ++idx;
            }
        }
    }
    
    if (dims[2] > 1)
    {
        int idx = 0;
        for (int i = 0; i < local_ni + 2; ++i)
        {
            for (int j = 0; j < local_nj + 2; ++j)
            {
                u[local_index(i, j, local_nk + 1)] = recv_buffer_front[idx];
                u[local_index(i, j, 0)] = recv_buffer_back[idx];
                ++idx;
            }
        }
    }
}


double calculate_max_error(const std::vector<double> &u_curr,
                           int local_ni, int local_nj, int local_nk,
                           int i_start, int j_start, int k_start,
                           double hx, double hy, double hz,
                           int time_step, double tau,
                           double Lx, double Ly, double Lz, double a_t,
                           MPI_Comm cart_comm)
{
    double local_error = 0.0;

    for (int i = 1; i <= local_ni; ++i)
    {
        for (int j = 1; j <= local_nj; ++j)
        {
            for (int k = 1; k <= local_nk; ++k)
            {
                double x = (i_start + (i - 1)) * hx;
                double y = (j_start + (j - 1)) * hy;
                double z = (k_start + (k - 1)) * hz;
                double u_exact = u_analytical(x, Lx, y, Ly, z, Lz, a_t, time_step * tau);
                double diff = fabs(u_curr[local_index(i, j, k)] - u_exact);
                local_error = std::max(local_error, diff);
            }
        }
    }

    double global_error = 0.0;
    MPI_Reduce(&local_error, &global_error, 1, MPI_DOUBLE, MPI_MAX, 0, cart_comm);
    return global_error;
}

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    auto start = std::chrono::high_resolution_clock::now();

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 3) {
        std::cerr << "Expected 2 arguments: N and L (1 or PI)\n";
        MPI_Finalize();
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
    if (rank == 0)
    {
        std::cout << "Running with N = " << N << " and L = " << L << ", processes = " << size << std::endl;
    }

    const double Lx = L;
    const double Ly = L;
    const double Lz = L;

    const int TIME_STEPS = 20;
    const double hx = Lx / N;
    const double hy = Ly / N;
    const double hz = Lz / N;
    const double tau = hx / 100.0;

    N += 1;

    const double a_t = 2.0 * M_PI;
    const double a2 = 4.0 / (9.0 * Lx * Lx) + 4.0 / (Ly * Ly) + 4.0 / (Lz * Lz);

    int dims[3] = {0, 0, 0};
    MPI_Dims_create(size, 3, dims);

    if (rank == 0)
    {
        std::cout << "3D Grid decomposition: " << dims[0] << " x " << dims[1] << " x " << dims[2] << std::endl;
    }

    int periods[3] = {0, 1, 1};
    MPI_Comm cart_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periods, 0, &cart_comm);

    int coords[3];
    MPI_Comm_rank(cart_comm, &rank);
    MPI_Cart_coords(cart_comm, rank, 3, coords);

    int i_start, j_start, k_start;
    int local_ni = compute_local_size(N, dims[0], coords[0], &i_start);
    int local_nj = compute_local_size(N, dims[1], coords[1], &j_start);
    int local_nk = compute_local_size(N, dims[2], coords[2], &k_start);

    ALLOC_NJ = local_nj + 2;
    ALLOC_NK = local_nk + 2;

    std::vector<double> u_prev((local_ni + 2) * (local_nj + 2) * (local_nk + 2), 0.0);
    std::vector<double> u_curr((local_ni + 2) * (local_nj + 2) * (local_nk + 2), 0.0);
    std::vector<double> u_next((local_ni + 2) * (local_nj + 2) * (local_nk + 2), 0.0);

    for (int i = 1; i <= local_ni; ++i)
    {
        for (int j = 1; j <= local_nj; ++j)
        {
            for (int k = 1; k <= local_nk; ++k)
            {
                int global_i = i_start + (i - 1);
                int global_j = j_start + (j - 1);
                int global_k = k_start + (k - 1);
                
                double x = global_i * hx;
                double y = global_j * hy;
                double z = global_k * hz;
                
                u_prev[local_index(i, j, k)] = 
                    u_analytical(x, Lx, y, Ly, z, Lz, a_t, 0.0);
            }
        }
    }

    change_bounds(u_prev, local_ni, local_nj, local_nk, cart_comm);

    for (int i = 1; i <= local_ni; ++i)
    {
        for (int j = 1; j <= local_nj; ++j)
        {
            for (int k = 1; k <= local_nk; ++k)
            {
                int global_i = i_start + (i - 1);
                
                if (global_i == 0 || global_i == N - 1)
                {
                    u_curr[local_index(i, j, k)] = 0.0;
                    continue;
                }

                double laplacian = compute_full_laplacian(i, j, k, u_prev, hx, hy, hz);
                u_curr[local_index(i, j, k)] = u_prev[local_index(i, j, k)] + 0.5 * a2 * tau * tau * laplacian;
            }
        }
    }

    
    for (int n = 1; n < TIME_STEPS; ++n)
    {
        change_bounds(u_curr, local_ni, local_nj, local_nk, cart_comm);

        for (int i = 1; i <= local_ni; ++i)
        {
            for (int j = 1; j <= local_nj; ++j)
            {
                for (int k = 1; k <= local_nk; ++k)
                {
                    int global_i = i_start + (i - 1);
                    
                    if (global_i == 0 || global_i == N - 1)
                    {
                        u_next[local_index(i, j, k)] = 0.0;
                        continue;
                    }

                    double laplacian = compute_full_laplacian(i, j, k, u_curr, hx, hy, hz);
                    u_next[local_index(i, j, k)] =
                        2.0 * u_curr[local_index(i, j, k)] - u_prev[local_index(i, j, k)] + a2 * tau * tau * laplacian;
                }
            }
        }

        std::swap(u_prev, u_curr);
        std::swap(u_curr, u_next);
        
        double max_error = calculate_max_error(u_curr, local_ni, local_nj, local_nk,
                                               i_start, j_start, k_start,
                                               hx, hy, hz, n, tau,
                                               Lx, Ly, Lz, a_t,
                                               cart_comm);
        
        if (rank == 0)
        {
            if (n < TIME_STEPS - 1) {
            std::cout << "Max error at current step: " << max_error << std::endl;
            } else {
                std::cout << "Final error: " << max_error << std::endl;
            }
        }
    }

    MPI_Barrier(cart_comm);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    
    if (rank == 0)
    {
        std::cout << "Elapsed time: " << elapsed.count() << std::endl;
    }

    MPI_Comm_free(&cart_comm);
    MPI_Finalize();
    return 0;
}
