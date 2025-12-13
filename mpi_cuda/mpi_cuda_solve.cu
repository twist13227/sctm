#include <mpi.h>
#include <cuda_runtime.h>

#include <vector>
#include <cmath>
#include <iostream>
#include <chrono>
#include <cassert>
#include <cstring>
#include <cstdlib>

#include <thrust/device_vector.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>


#define SAFE_CALL(call) do {                                                            \
    cudaError_t err = (call);                                                           \
    if (err != cudaSuccess) {                                                           \
        printf("Error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__);    \
        exit(1);                                                                        \
    }                                                                                   \
} while (0)



__host__ __device__ __forceinline__
double u_analytical(double x, double Lx, double y, double Ly, double z, double Lz, double a_t, double t) {
    return sin(3.0 * M_PI * x / Lx) * sin(2.0 * M_PI * y / Ly) * sin(2.0 * M_PI * z / Lz) * cos(a_t * t + 4.0 * M_PI);
}


__host__ __device__ __forceinline__ size_t IDX(int i, int j, int k, int nj, int nk) {
    return (size_t)i * nj * nk + (size_t)j * nk + (size_t)k;
}


__global__ void pack_i_layer(const double* __restrict__ d_u,
                             double* __restrict__ d_out,
                             int layer_i, int alloc_ni, int alloc_nj, int alloc_nk) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = alloc_nj * alloc_nk;
    if (tid >= total) return;
    int j = tid / alloc_nk;
    int k = tid % alloc_nk;
    d_out[tid] = d_u[IDX(layer_i, j, k, alloc_nj, alloc_nk)];
}


__global__ void pack_j_layer(const double* __restrict__ d_u,
                             double* __restrict__ d_out,
                             int layer_j, int alloc_ni, int alloc_nj, int alloc_nk) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = alloc_ni * alloc_nk;
    if (tid >= total) return;
    int i = tid / alloc_nk;
    int k = tid % alloc_nk;
    d_out[tid] = d_u[IDX(i, layer_j, k, alloc_nj, alloc_nk)];
}


__global__ void pack_k_layer(const double* __restrict__ d_u,
                             double* __restrict__ d_out,
                             int layer_k, int alloc_ni, int alloc_nj, int alloc_nk) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = alloc_ni * alloc_nj;
    if (tid >= total) return;
    int i = tid / alloc_nj;
    int j = tid % alloc_nj;
    d_out[tid] = d_u[IDX(i, j, layer_k, alloc_nj, alloc_nk)];
}


__global__ void unpack_i_layer(double* __restrict__ d_u,
                               const double* __restrict__ d_in,
                               int layer_i, int alloc_ni, int alloc_nj, int alloc_nk) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = alloc_nj * alloc_nk;
    if (tid >= total) return;
    int j = tid / alloc_nk;
    int k = tid % alloc_nk;
    d_u[IDX(layer_i, j, k, alloc_nj, alloc_nk)] = d_in[tid];
}


__global__ void unpack_j_layer(double* __restrict__ d_u,
                               const double* __restrict__ d_in,
                               int layer_j, int alloc_ni, int alloc_nj, int alloc_nk) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = alloc_ni * alloc_nk;
    if (tid >= total) return;
    int i = tid / alloc_nk;
    int k = tid % alloc_nk;
    d_u[IDX(i, layer_j, k, alloc_nj, alloc_nk)] = d_in[tid];
}


__global__ void unpack_k_layer(double* __restrict__ d_u,
                               const double* __restrict__ d_in,
                               int layer_k, int alloc_ni, int alloc_nj, int alloc_nk) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = alloc_ni * alloc_nj;
    if (tid >= total) return;
    int i = tid / alloc_nj;
    int j = tid % alloc_nj;
    d_u[IDX(i, j, layer_k, alloc_nj, alloc_nk)] = d_in[tid];
}


__global__ void core_kernel(const double* __restrict__ d_prev,
                            const double* __restrict__ d_curr,
                            double* __restrict__ d_next,
                            int local_ni, int local_nj, int local_nk,
                            int alloc_nj, int alloc_nk,
                            double hx, double hy, double hz,
                            double a2, double tau) {
    int ii = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int jj = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int kk = blockIdx.z * blockDim.z + threadIdx.z + 1;
    if (ii > local_ni || jj > local_nj || kk > local_nk) return;

    const size_t di = (size_t)alloc_nj * alloc_nk;
    const size_t dj = (size_t)alloc_nk;
    const size_t dk = 1;
    size_t idx = IDX(ii, jj, kk, alloc_nj, alloc_nk);
    
    // c=center, p=+1, m=-1
    double u_c  = d_curr[idx];
    double u_ip = d_curr[idx + di];
    double u_im = d_curr[idx - di];
    double u_jp = d_curr[idx + dj];
    double u_jm = d_curr[idx - dj];
    double u_kp = d_curr[idx + dk];
    double u_km = d_curr[idx - dk];

    double laplacian = (u_ip - 2.0 * u_c + u_im) / (hx * hx)
               + (u_jp - 2.0 * u_c + u_jm) / (hy * hy)
               + (u_kp - 2.0 * u_c + u_km) / (hz * hz);

    d_next[idx] = 2.0 * u_c - d_prev[idx] + a2 * tau * tau * laplacian;
}


__global__ void halo_kernel(const double* __restrict__ d_prev,
                            const double* __restrict__ d_curr,
                            double* __restrict__ d_next,
                            int local_ni, int local_nj, int local_nk,
                            int alloc_ni, int alloc_nj, int alloc_nk,
                            double hx, double hy, double hz,
                            double a2, double tau) {
    int ia = blockIdx.x * blockDim.x + threadIdx.x;
    int ja = blockIdx.y * blockDim.y + threadIdx.y;
    int ka = blockIdx.z * blockDim.z + threadIdx.z;

    if (ia >= alloc_ni || ja >= alloc_nj || ka >= alloc_nk) return;
    if (ia >= 1 && ia <= local_ni && ja >= 1 && ja <= local_nj && ka >= 1 && ka <= local_nk) return;
    
    int ip = min(ia + 1, alloc_ni - 1);
    int im = max(ia - 1, 0);
    int jp = min(ja + 1, alloc_nj - 1);
    int jm = max(ja - 1, 0);
    int kp = min(ka + 1, alloc_nk - 1);
    int km = max(ka - 1, 0);

    double u_c  = d_curr[IDX(ia, ja, ka, alloc_nj, alloc_nk)];
    double u_ip = d_curr[IDX(ip, ja, ka, alloc_nj, alloc_nk)];
    double u_im = d_curr[IDX(im, ja, ka, alloc_nj, alloc_nk)];
    double u_jp = d_curr[IDX(ia, jp, ka, alloc_nj, alloc_nk)];
    double u_jm = d_curr[IDX(ia, jm, ka, alloc_nj, alloc_nk)];
    double u_kp = d_curr[IDX(ia, ja, kp, alloc_nj, alloc_nk)];
    double u_km = d_curr[IDX(ia, ja, km, alloc_nj, alloc_nk)];

    double laplacian = (u_ip - 2.0 * u_c + u_im) / (hx * hx)
               + (u_jp - 2.0 * u_c + u_jm) / (hy * hy)
               + (u_kp - 2.0 * u_c + u_km) / (hz * hz);

    d_next[IDX(ia, ja, ka, alloc_nj, alloc_nk)] = 2.0 * u_c - d_prev[IDX(ia, ja, ka, alloc_nj, alloc_nk)] + a2 * tau * tau * laplacian;
}


__global__ void initial_step_kernel(
    const double* __restrict__ d_prev,
    double* __restrict__ d_curr,
    int local_ni, int local_nj, int local_nk,
    int alloc_nj, int alloc_nk,
    double hx, double hy, double hz,
    double a2, double tau) {
   
    int ii = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int jj = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int kk = blockIdx.z * blockDim.z + threadIdx.z + 1;
    if (ii > local_ni || jj > local_nj || kk > local_nk) return;
    const size_t di = (size_t)alloc_nj * alloc_nk;
    const size_t dj = (size_t)alloc_nk;
    const size_t dk = 1;
    size_t idx = IDX(ii, jj, kk, alloc_nj, alloc_nk);
   
    double center = d_prev[idx];
    double ip = d_prev[idx + di];
    double im = d_prev[idx - di];
    double jp = d_prev[idx + dj];
    double jm = d_prev[idx - dj];
    double kp = d_prev[idx + dk];
    double km = d_prev[idx - dk];
    double laplacian = (ip - 2.0 * center + im) / (hx * hx)
                + (jp - 2.0 * center + jm) / (hy * hy)
                + (kp - 2.0 * center + km) / (hz * hz);
   d_curr[idx] = center + 0.5 * a2 * tau * tau * laplacian;
}


__global__ void apply_periodic_y(double* __restrict__ d_u,
                                 int local_ni, int local_nj, int local_nk,
                                 int alloc_nj, int alloc_nk) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int k = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= local_ni + 2 || k >= local_nk + 2) return;
    d_u[IDX(i, 0, k, alloc_nj, alloc_nk)] = d_u[IDX(i, local_nj, k, alloc_nj, alloc_nk)];
    d_u[IDX(i, local_nj + 1, k, alloc_nj, alloc_nk)] = d_u[IDX(i, 1, k, alloc_nj, alloc_nk)];
}


__global__ void apply_periodic_z(double* __restrict__ d_u,
                                 int local_ni, int local_nj, int local_nk,
                                 int alloc_nj, int alloc_nk) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= local_ni + 2 || j >= local_nj + 2) return;
    d_u[IDX(i, j, 0, alloc_nj, alloc_nk)] = d_u[IDX(i, j, local_nk, alloc_nj, alloc_nk)];
    d_u[IDX(i, j, local_nk + 1, alloc_nj, alloc_nk)] = d_u[IDX(i, j, 1, alloc_nj, alloc_nk)];
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


void init_host_u(std::vector<double>& h_u_prev,
                 int local_ni, int local_nj, int local_nk,
                 int i_start, int j_start, int k_start,
                 double hx, double hy, double hz,
                 double Lx, double Ly, double Lz, double a_t) {
    for (int i = 1; i <= local_ni; ++i)
        for (int j = 1; j <= local_nj; ++j)
            for (int k = 1; k <= local_nk; ++k) {
                double x = (i_start + (i - 1)) * hx;
                double y = (j_start + (j - 1)) * hy;
                double z = (k_start + (k - 1)) * hz;
                h_u_prev[IDX(i, j, k, local_nj + 2, local_nk + 2)] = u_analytical(x, Lx, y, Ly, z, Lz, a_t, 0.0);
            }
}


struct max_error_functor {
    const double* u;
    int ni, nj, nk;
    int i_start, j_start, k_start;
    double hx, hy, hz;
    double Lx, Ly, Lz;
    double a_t;
    double t;

    __host__ __device__
    max_error_functor(const double* u_,
                      int ni_, int nj_, int nk_,
                      int i_start_, int j_start_, int k_start_,
                      double hx_, double hy_, double hz_,
                      double Lx_, double Ly_, double Lz_,
                      double a_t_, double t_)
        : u(u_), ni(ni_), nj(nj_), nk(nk_),
          i_start(i_start_), j_start(j_start_), k_start(k_start_),
          hx(hx_), hy(hy_), hz(hz_),
          Lx(Lx_), Ly(Ly_), Lz(Lz_),
          a_t(a_t_), t(t_) {}

    __host__ __device__
    double operator()(size_t idx) const {
        int k = idx % nk;
        int j = (idx / nk) % nj;
        int i = idx / (nj * nk);
        double val = u[IDX(i + 1, j + 1, k + 1, nj + 2, nk + 2)];
        double ex  = u_analytical((i_start + i) * hx, Lx, (j_start + j) * hy, Ly, (k_start + k) * hz, Lz, a_t, t);
        return fabs(val - ex);
    }
};


double compute_max_error_thrust(const double* d_u,
                                int ni, int nj, int nk,
                                int i_start, int j_start, int k_start,
                                double hx, double hy, double hz,
                                double Lx, double Ly, double Lz,
                                double a_t,
                                double t,
                                MPI_Comm cart_comm) {
    size_t total = (size_t)ni * nj * nk;
    thrust::counting_iterator<size_t> first(0);
    thrust::counting_iterator<size_t> last(total);
    max_error_functor f(d_u, ni, nj, nk, i_start, j_start, k_start,
                        hx, hy, hz, Lx, Ly, Lz, a_t, t);
    double local_error = thrust::transform_reduce(first, last, f, 0.0, thrust::maximum<double>());
    double global_error = 0.0;
    MPI_Reduce(&local_error, &global_error, 1, MPI_DOUBLE, MPI_MAX, 0, cart_comm);
    return global_error;
}


enum Route {
    I_LEFT  = 0,
    I_RIGHT = 1,
    J_DOWN  = 2,
    J_UP    = 3,
    K_BACK  = 4,
    K_FRONT = 5,
};
constexpr int NUM_ROUTES = 6;

struct HaloBuffers {
    double* h_send[NUM_ROUTES] = {nullptr};
    double* h_recv[NUM_ROUTES] = {nullptr};
    double* d_pack[NUM_ROUTES] = {nullptr};
    size_t slice_elems[NUM_ROUTES] = {0};
};


void alloc_halo(HaloBuffers& hb, int alloc_ni, int alloc_nj, int alloc_nk) {
    hb.slice_elems[I_LEFT]  = hb.slice_elems[I_RIGHT] = (size_t)alloc_nj * alloc_nk;
    hb.slice_elems[J_DOWN]  = hb.slice_elems[J_UP]    = (size_t)alloc_ni * alloc_nk;
    hb.slice_elems[K_BACK]  = hb.slice_elems[K_FRONT] = (size_t)alloc_ni * alloc_nj;
    for (int r = 0; r < NUM_ROUTES; ++r) {
        SAFE_CALL(cudaMalloc(&hb.d_pack[r], hb.slice_elems[r] * sizeof(double)));
    }
    for (int r = 0; r < NUM_ROUTES; ++r) {
        SAFE_CALL(cudaHostAlloc(&hb.h_send[r], hb.slice_elems[r] * sizeof(double), cudaHostAllocPortable));
        SAFE_CALL(cudaHostAlloc(&hb.h_recv[r], hb.slice_elems[r] * sizeof(double), cudaHostAllocPortable));
    }
}


void free_halo(HaloBuffers& hb) {
    for (int r = 0; r < NUM_ROUTES; ++r) {
        if (hb.d_pack[r]) {
            cudaFree(hb.d_pack[r]);
            hb.d_pack[r] = nullptr;
        }
    }
    for (int r = 0; r < NUM_ROUTES; ++r) {
        if (hb.h_send[r]) {
            cudaFreeHost(hb.h_send[r]);
            hb.h_send[r] = nullptr;
        }
        if (hb.h_recv[r]) {
            cudaFreeHost(hb.h_recv[r]);
            hb.h_recv[r] = nullptr;
        }
    }
}


inline int get_neighbor(int route, int nL, int nR, int nD, int nU, int nB, int nF) {
    switch (route) {
        case I_LEFT:  return nL;
        case I_RIGHT: return nR;
        case J_DOWN:  return nD;
        case J_UP:    return nU;
        case K_BACK:  return nB;
        case K_FRONT: return nF;
        default: return MPI_PROC_NULL;
    }
}

void pack_and_post(MPI_Comm cart_comm, HaloBuffers &hb, int rank,
                   double* d_u_curr,
                   int local_ni, int local_nj, int local_nk,
                   int alloc_ni, int alloc_nj, int alloc_nk,
                   int nL, int nR, int nD, int nU, int nB, int nF,
                   bool do_x, bool do_y, bool do_z,
                   MPI_Request *requests, int &req_count) {
    const int PACK_BLOCK = 256;
    if (do_x) {
        int total = alloc_nj * alloc_nk;
        int grid = (total + PACK_BLOCK - 1) / PACK_BLOCK;
        pack_i_layer<<<grid, PACK_BLOCK>>>(d_u_curr, hb.d_pack[I_LEFT], 1, alloc_ni, alloc_nj, alloc_nk);
        pack_i_layer<<<grid, PACK_BLOCK>>>(d_u_curr, hb.d_pack[I_RIGHT], local_ni, alloc_ni, alloc_nj, alloc_nk);
    }
    if (do_y) {
        int total = alloc_ni * alloc_nk;
        int grid = (total + PACK_BLOCK - 1) / PACK_BLOCK;
        pack_j_layer<<<grid, PACK_BLOCK>>>(d_u_curr, hb.d_pack[J_DOWN], 1, alloc_ni, alloc_nj, alloc_nk);
        pack_j_layer<<<grid, PACK_BLOCK>>>(d_u_curr, hb.d_pack[J_UP], local_nj, alloc_ni, alloc_nj, alloc_nk);
    }
    if (do_z) {
        int total = alloc_ni * alloc_nj;
        int grid = (total + PACK_BLOCK - 1) / PACK_BLOCK;
        pack_k_layer<<<grid, PACK_BLOCK>>>(d_u_curr, hb.d_pack[K_BACK], 1, alloc_ni, alloc_nj, alloc_nk);
        pack_k_layer<<<grid, PACK_BLOCK>>>(d_u_curr, hb.d_pack[K_FRONT], local_nk, alloc_ni, alloc_nj, alloc_nk);
    }
    SAFE_CALL(cudaDeviceSynchronize());

    if (do_x) {
        SAFE_CALL(cudaMemcpy(hb.h_send[I_LEFT], hb.d_pack[I_LEFT], hb.slice_elems[I_LEFT] * sizeof(double), cudaMemcpyDeviceToHost));
        SAFE_CALL(cudaMemcpy(hb.h_send[I_RIGHT], hb.d_pack[I_RIGHT], hb.slice_elems[I_RIGHT] * sizeof(double), cudaMemcpyDeviceToHost));
    }
    if (do_y) {
        SAFE_CALL(cudaMemcpy(hb.h_send[J_DOWN], hb.d_pack[J_DOWN], hb.slice_elems[J_DOWN] * sizeof(double), cudaMemcpyDeviceToHost));
        SAFE_CALL(cudaMemcpy(hb.h_send[J_UP], hb.d_pack[J_UP], hb.slice_elems[J_UP] * sizeof(double), cudaMemcpyDeviceToHost));
    }
    if (do_z) {
        SAFE_CALL(cudaMemcpy(hb.h_send[K_BACK], hb.d_pack[K_BACK], hb.slice_elems[K_BACK] * sizeof(double), cudaMemcpyDeviceToHost));
        SAFE_CALL(cudaMemcpy(hb.h_send[K_FRONT], hb.d_pack[K_FRONT], hb.slice_elems[K_FRONT] * sizeof(double), cudaMemcpyDeviceToHost));
    }

    req_count = 0;

    if (do_y && nU == rank) {
        memcpy(hb.h_recv[J_DOWN], hb.h_send[J_UP], hb.slice_elems[J_UP] * sizeof(double));
    }
    if (do_y && nD == rank) {
        memcpy(hb.h_recv[J_UP], hb.h_send[J_DOWN], hb.slice_elems[J_DOWN] * sizeof(double));
    }
    if (do_z && nF == rank) {
        memcpy(hb.h_recv[K_BACK], hb.h_send[K_FRONT], hb.slice_elems[K_FRONT] * sizeof(double));
    }
    if (do_z && nB == rank) {
        memcpy(hb.h_recv[K_FRONT], hb.h_send[K_BACK], hb.slice_elems[K_BACK] * sizeof(double));
    }
    
    if (do_x) {
        if (nL != MPI_PROC_NULL && nL != rank)
            MPI_Irecv(hb.h_recv[I_LEFT], (int)hb.slice_elems[I_LEFT], MPI_DOUBLE, nL, I_RIGHT, cart_comm, &requests[req_count++]);
        if (nR != MPI_PROC_NULL && nR != rank)
            MPI_Irecv(hb.h_recv[I_RIGHT], (int)hb.slice_elems[I_RIGHT], MPI_DOUBLE, nR, I_LEFT, cart_comm, &requests[req_count++]);
    }
    if (do_y) {
        if (nD != MPI_PROC_NULL && nD != rank)
            MPI_Irecv(hb.h_recv[J_DOWN], (int)hb.slice_elems[J_DOWN], MPI_DOUBLE, nD, J_UP, cart_comm, &requests[req_count++]);
        if (nU != MPI_PROC_NULL && nU != rank)
            MPI_Irecv(hb.h_recv[J_UP], (int)hb.slice_elems[J_UP], MPI_DOUBLE, nU, J_DOWN, cart_comm, &requests[req_count++]);
    }
    if (do_z) {
        if (nB != MPI_PROC_NULL && nB != rank)
            MPI_Irecv(hb.h_recv[K_BACK], (int)hb.slice_elems[K_BACK], MPI_DOUBLE, nB, K_FRONT, cart_comm, &requests[req_count++]);
        if (nF != MPI_PROC_NULL && nF != rank)
            MPI_Irecv(hb.h_recv[K_FRONT], (int)hb.slice_elems[K_FRONT], MPI_DOUBLE, nF, K_BACK, cart_comm, &requests[req_count++]);
    }

    if (do_x) {
        if (nR != MPI_PROC_NULL && nR != rank)
            MPI_Isend(hb.h_send[I_RIGHT], (int)hb.slice_elems[I_RIGHT], MPI_DOUBLE, nR, I_RIGHT, cart_comm, &requests[req_count++]);
        if (nL != MPI_PROC_NULL && nL != rank)
            MPI_Isend(hb.h_send[I_LEFT], (int)hb.slice_elems[I_LEFT], MPI_DOUBLE, nL, I_LEFT, cart_comm, &requests[req_count++]);
    }
    if (do_y) {
        if (nU != MPI_PROC_NULL && nU != rank)
            MPI_Isend(hb.h_send[J_UP], (int)hb.slice_elems[J_UP], MPI_DOUBLE, nU, J_UP, cart_comm, &requests[req_count++]);
        if (nD != MPI_PROC_NULL && nD != rank)
            MPI_Isend(hb.h_send[J_DOWN], (int)hb.slice_elems[J_DOWN], MPI_DOUBLE, nD, J_DOWN, cart_comm, &requests[req_count++]);
    }
    if (do_z) {
        if (nF != MPI_PROC_NULL && nF != rank)
            MPI_Isend(hb.h_send[K_FRONT], (int)hb.slice_elems[K_FRONT], MPI_DOUBLE, nF, K_FRONT, cart_comm, &requests[req_count++]);
        if (nB != MPI_PROC_NULL && nB != rank)
            MPI_Isend(hb.h_send[K_BACK], (int)hb.slice_elems[K_BACK], MPI_DOUBLE, nB, K_BACK, cart_comm, &requests[req_count++]);
    }
}


void recv_and_unpack(HaloBuffers &hb, double* d_u_curr,
                     int local_ni, int local_nj, int local_nk,
                     int alloc_ni, int alloc_nj, int alloc_nk,
                     bool do_x, bool do_y, bool do_z) {
    if (do_x) {
        SAFE_CALL(cudaMemcpy(hb.d_pack[I_LEFT], hb.h_recv[I_LEFT], hb.slice_elems[I_LEFT] * sizeof(double), cudaMemcpyHostToDevice));
        SAFE_CALL(cudaMemcpy(hb.d_pack[I_RIGHT], hb.h_recv[I_RIGHT], hb.slice_elems[I_RIGHT] * sizeof(double), cudaMemcpyHostToDevice));
    }
    if (do_y) {
        SAFE_CALL(cudaMemcpy(hb.d_pack[J_DOWN], hb.h_recv[J_DOWN], hb.slice_elems[J_DOWN] * sizeof(double), cudaMemcpyHostToDevice));
        SAFE_CALL(cudaMemcpy(hb.d_pack[J_UP], hb.h_recv[J_UP], hb.slice_elems[J_UP] * sizeof(double), cudaMemcpyHostToDevice));
    }
    if (do_z) {
        SAFE_CALL(cudaMemcpy(hb.d_pack[K_BACK], hb.h_recv[K_BACK], hb.slice_elems[K_BACK] * sizeof(double), cudaMemcpyHostToDevice));
        SAFE_CALL(cudaMemcpy(hb.d_pack[K_FRONT], hb.h_recv[K_FRONT], hb.slice_elems[K_FRONT] * sizeof(double), cudaMemcpyHostToDevice));
    }

    const int PACK_BLOCK = 256;
    if (do_x) {
        int total = alloc_nj * alloc_nk;
        int grid = (total + PACK_BLOCK - 1) / PACK_BLOCK;
        unpack_i_layer<<<grid, PACK_BLOCK>>>(d_u_curr, hb.d_pack[I_LEFT], 0, alloc_ni, alloc_nj, alloc_nk);
        unpack_i_layer<<<grid, PACK_BLOCK>>>(d_u_curr, hb.d_pack[I_RIGHT], local_ni+1, alloc_ni, alloc_nj, alloc_nk);
    }
    if (do_y) {
        int total = alloc_ni * alloc_nk;
        int grid = (total + PACK_BLOCK - 1) / PACK_BLOCK;
        unpack_j_layer<<<grid, PACK_BLOCK>>>(d_u_curr, hb.d_pack[J_DOWN], 0, alloc_ni, alloc_nj, alloc_nk);
        unpack_j_layer<<<grid, PACK_BLOCK>>>(d_u_curr, hb.d_pack[J_UP], local_nj+1, alloc_ni, alloc_nj, alloc_nk);
    }
    if (do_z) {
        int total = alloc_ni * alloc_nj;
        int grid = (total + PACK_BLOCK - 1) / PACK_BLOCK;
        unpack_k_layer<<<grid, PACK_BLOCK>>>(d_u_curr, hb.d_pack[K_BACK], 0, alloc_ni, alloc_nj, alloc_nk);
        unpack_k_layer<<<grid, PACK_BLOCK>>>(d_u_curr, hb.d_pack[K_FRONT], local_nk+1, alloc_ni, alloc_nj, alloc_nk);
    }
    SAFE_CALL(cudaDeviceSynchronize());
}


void do_time_step(MPI_Comm cart_comm,
                  std::vector<double> &h_prev, std::vector<double> &h_curr, std::vector<double> &h_next,
                  double* d_prev, double* d_curr, double* d_next,
                  HaloBuffers &hb,
                  int local_ni, int local_nj, int local_nk,
                  int alloc_ni, int alloc_nj, int alloc_nk,
                  int nL, int nR, int nD, int nU, int nB, int nF,
                  bool do_x, bool do_y, bool do_z,
                  int dims_y, int dims_z, int periods_y, int periods_z,
                  double hx, double hy, double hz, double a2, double tau,
                  int rank, int step, bool verbose) {
    MPI_Request reqs[12];
    int req_count = 0;
    double t_start, t_end;

    t_start = MPI_Wtime();
    pack_and_post(cart_comm, hb, rank, d_curr,
                  local_ni, local_nj, local_nk,
                  alloc_ni, alloc_nj, alloc_nk,
                  nL, nR, nD, nU, nB, nF,
                  do_x, do_y, do_z,
                  reqs, req_count);
    t_end = MPI_Wtime();
    if (verbose && rank == 0) {
        std::cout << "  Step " << step << " - pack & post: " << (t_end - t_start) * 1000.0 << " ms\n";
    }

    t_start = MPI_Wtime();
    dim3 core_block(8, 8, 8);
    dim3 core_grid(
        (local_ni + core_block.x - 1) / core_block.x,
        (local_nj + core_block.y - 1) / core_block.y,
        (local_nk + core_block.z - 1) / core_block.z
    );
    core_kernel<<<core_grid, core_block>>>(d_prev, d_curr, d_next,
                                           local_ni, local_nj, local_nk,
                                           alloc_nj, alloc_nk,
                                           hx, hy, hz, a2, tau);
    SAFE_CALL(cudaDeviceSynchronize());                                       
    t_end = MPI_Wtime();
    if (verbose && rank == 0) {
        std::cout << "  Step " << step << " - core kernel launch: " << (t_end - t_start) * 1000.0 << " ms\n";
    }

    t_start = MPI_Wtime();
    if (req_count > 0) MPI_Waitall(req_count, reqs, MPI_STATUSES_IGNORE);
    t_end = MPI_Wtime();
    if (verbose && rank == 0) {
        std::cout << "  Step " << step << " - wait MPI: " << (t_end - t_start) * 1000.0 << " ms\n";
    }

    t_start = MPI_Wtime();
    recv_and_unpack(hb, d_curr, local_ni, local_nj, local_nk, alloc_ni, alloc_nj, alloc_nk, do_x, do_y, do_z);
    
    if (dims_y == 1 && periods_y) {
        dim3 block(16, 16);
        dim3 grid((alloc_ni + block.x - 1) / block.x, (alloc_nk + block.y - 1) / block.y);
        apply_periodic_y<<<grid, block>>>(d_curr, local_ni, local_nj, local_nk, alloc_nj, alloc_nk);
    }
    if (dims_z == 1 && periods_z) {
        dim3 block(16, 16);
        dim3 grid((alloc_ni + block.x - 1) / block.x, (alloc_nj + block.y - 1) / block.y);
        apply_periodic_z<<<grid, block>>>(d_curr, local_ni, local_nj, local_nk, alloc_nj, alloc_nk);
    }
    SAFE_CALL(cudaDeviceSynchronize());
    t_end = MPI_Wtime();
    if (verbose && rank == 0) {
        std::cout << "  Step " << step << " - recv & unpack: " << (t_end - t_start) * 1000.0 << " ms\n";
    }

    t_start = MPI_Wtime();
    SAFE_CALL(cudaDeviceSynchronize());

    dim3 halo_block(8, 8, 8);
    dim3 halo_grid(
        (alloc_ni + halo_block.x - 1) / halo_block.x,
        (alloc_nj + halo_block.y - 1) / halo_block.y,
        (alloc_nk + halo_block.z - 1) / halo_block.z
    );
    halo_kernel<<<halo_grid, halo_block>>>(d_prev, d_curr, d_next,
                                           local_ni, local_nj, local_nk,
                                           alloc_ni, alloc_nj, alloc_nk,
                                           hx, hy, hz, a2, tau);
    SAFE_CALL(cudaDeviceSynchronize());
    t_end = MPI_Wtime();
    if (verbose && rank == 0) {
        std::cout << "  Step " << step << " - halo kernel: " << (t_end - t_start) * 1000.0 << " ms\n";
    }
}


int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Expected 2 arguments: N and L (1 or PI)\n";
        return 1;
    }
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int device_count;
    SAFE_CALL(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
        if (rank == 0) {
            std::cerr << "No CUDA devices available!\n";
        }
        MPI_Finalize();
        return 1;
    }
    int device_id = rank % device_count;
    SAFE_CALL(cudaSetDevice(device_id));
    

    int N = std::atoi(argv[1]);
    std::string L_str = argv[2];
    double L;

    if (L_str == "1") {
        L = 1.0;
    } else if (L_str == "PI") {
        L = M_PI;
    } else {
        std::cerr << "Error: L must be either 1 or PI.\n";
        MPI_Finalize(); 
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
    const double a_t = 2.0 * M_PI;
    const double a2 = 4.0 / (9.0 * Lx * Lx) + 4.0 / (Ly * Ly) + 4.0 / (Lz * Lz);
    int dims[3] = {0, 0, 0};
    MPI_Dims_create(size, 3, dims);
    if (rank == 0) {
        std::cout << "MPI decomposition: " << dims[0] << " x " << dims[1] << " x " << dims[2] << "\n";
    }
    int periods[3] = {0, 1, 1};
    MPI_Comm cart_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periods, 0, &cart_comm);
    int coords[3];
    MPI_Cart_coords(cart_comm, rank, 3, coords);
    int i_start, j_start, k_start;
    int local_ni = compute_local_size(N + 1, dims[0], coords[0], &i_start);
    int local_nj = compute_local_size(N + 1, dims[1], coords[1], &j_start);
    int local_nk = compute_local_size(N + 1, dims[2], coords[2], &k_start);
    int alloc_ni = local_ni + 2;
    int alloc_nj = local_nj + 2;
    int alloc_nk = local_nk + 2;
    size_t local_elems = (size_t)alloc_ni * alloc_nj * alloc_nk;
    size_t local_bytes = local_elems * sizeof(double);
    if (rank == 0) {
        std::cout << "Local indices: " << local_ni << " x " << local_nj << " x " << local_nk << "\n";
    }
    std::vector<double> h_u_prev(local_elems, 0.0);
    std::vector<double> h_u_curr(local_elems, 0.0);
    std::vector<double> h_u_next(local_elems, 0.0);
    init_host_u(h_u_prev, local_ni, local_nj, local_nk, i_start, j_start, k_start, hx, hy, hz, Lx, Ly, Lz, a_t);
    double *d_u_prev = nullptr, *d_u_curr = nullptr, *d_u_next = nullptr;
    SAFE_CALL(cudaMalloc((void**)&d_u_prev, local_bytes));
    SAFE_CALL(cudaMalloc((void**)&d_u_curr, local_bytes));
    SAFE_CALL(cudaMalloc((void**)&d_u_next, local_bytes));
    SAFE_CALL(cudaMemcpy(d_u_prev, h_u_prev.data(), local_bytes, cudaMemcpyHostToDevice));
    SAFE_CALL(cudaMemset(d_u_next, 0, local_bytes));
    int nL, nR, nD, nU, nB, nF;
    MPI_Cart_shift(cart_comm, 0, 1, &nL, &nR);
    MPI_Cart_shift(cart_comm, 1, 1, &nD, &nU);
    MPI_Cart_shift(cart_comm, 2, 1, &nB, &nF);

    int dims_cart[3], periods_cart[3], coords_cart[3];
    MPI_Cart_get(cart_comm, 3, dims_cart, periods_cart, coords_cart);

    bool do_x = (dims_cart[0] > 1);
    bool do_y = (dims_cart[1] > 1);
    bool do_z = (dims_cart[2] > 1);

    HaloBuffers hb;
    alloc_halo(hb, alloc_ni, alloc_nj, alloc_nk);

    {
        double t_init_ex = MPI_Wtime();
        MPI_Request tmp_reqs[12];
        int tmp_n = 0;
        pack_and_post(cart_comm, hb, rank, d_u_prev,
                      local_ni, local_nj, local_nk,
                      alloc_ni, alloc_nj, alloc_nk,
                      nL, nR, nD, nU, nB, nF, do_x, do_y, do_z,
                      tmp_reqs, tmp_n);
        if (tmp_n > 0) MPI_Waitall(tmp_n, tmp_reqs, MPI_STATUSES_IGNORE);
        recv_and_unpack(hb, d_u_prev, local_ni, local_nj, local_nk, alloc_ni, alloc_nj, alloc_nk, do_x, do_y, do_z);
        if (dims_cart[1] == 1 && periods_cart[1]) {
            dim3 block(16, 16);
            dim3 grid((alloc_ni + block.x - 1) / block.x, (alloc_nk + block.y - 1) / block.y);
            apply_periodic_y<<<grid, block>>>(d_u_prev, local_ni, local_nj, local_nk, alloc_nj, alloc_nk);
        }
        if (dims_cart[2] == 1 && periods_cart[2]) {
            dim3 block(16, 16);
            dim3 grid((alloc_ni + block.x - 1) / block.x, (alloc_nj + block.y - 1) / block.y);
            apply_periodic_z<<<grid, block>>>(d_u_prev, local_ni, local_nj, local_nk, alloc_nj, alloc_nk);
        }
        SAFE_CALL(cudaDeviceSynchronize());
        t_init_ex = MPI_Wtime() - t_init_ex;
        if (rank == 0) {
            std::cout << "Initial exchange: " << t_init_ex * 1000.0 << " ms\n";
        }
    }

    double t_init_step = MPI_Wtime();
    dim3 init_block(8, 8, 8);
    dim3 init_grid(
        (local_ni + init_block.x - 1) / init_block.x,
        (local_nj + init_block.y - 1) / init_block.y,
        (local_nk + init_block.z - 1) / init_block.z
    );
    initial_step_kernel<<<init_grid, init_block>>>(
        d_u_prev, d_u_curr,
        local_ni, local_nj, local_nk,
        alloc_nj, alloc_nk,
        hx, hy, hz, a2, tau
    );
    SAFE_CALL(cudaDeviceSynchronize());
    t_init_step = MPI_Wtime() - t_init_step;
    if (rank == 0) {
        std::cout << "Initial step computation: " << t_init_step * 1000.0 << " ms\n";
    }

    double t0 = MPI_Wtime();
    for (int step = 1; step <= TIME_STEPS; ++step) {
        bool verbose = (step <= 3 || step % 5 == 0 || step == TIME_STEPS);
        if (verbose && rank == 0) {
            std::cout << "\nStep " << step << "\n";
        }
        double t_step_start = MPI_Wtime();
        do_time_step(cart_comm, h_u_prev, h_u_curr, h_u_next,
                     d_u_prev, d_u_curr, d_u_next,
                     hb,
                     local_ni, local_nj, local_nk,
                     alloc_ni, alloc_nj, alloc_nk,
                     nL, nR, nD, nU, nB, nF,
                     do_x, do_y, do_z,
                     dims_cart[1], dims_cart[2], periods_cart[1], periods_cart[2],
                     hx, hy, hz, a2, tau,
                     rank, step, verbose);
        double t_step_end = MPI_Wtime();
        if (verbose && rank == 0) {
            std::cout << "  Step " << step << " - total time: " << (t_step_end - t_step_start) * 1000.0 << " ms\n";
        }

        h_u_prev.swap(h_u_curr);
        h_u_curr.swap(h_u_next);
        std::swap(d_u_prev, d_u_curr);
        std::swap(d_u_curr, d_u_next);

        double t_err = MPI_Wtime();
        double max_error = compute_max_error_thrust(d_u_curr,
            local_ni, local_nj, local_nk,
            i_start, j_start, k_start,
            hx, hy, hz,
            Lx, Ly, Lz,
            a_t,
            step * tau,
            cart_comm
        );
        t_err = MPI_Wtime() - t_err;
        if (rank == 0)
        {
            if (step < TIME_STEPS) {
                std::cout << "Max error at step " << step << ": " << max_error << std::endl
                        << "Error calc: " << t_err * 1000.0 << " ms\n";
            } else {
                std::cout << "Final error: " << max_error << std::endl;
            }
        }
    }
    double t1 = MPI_Wtime();
    if (rank == 0) {
        std::cout << "Total time: " << (t1 - t0) << " s\n";
    }

    free_halo(hb);
    cudaFree(d_u_prev);
    cudaFree(d_u_curr);
    cudaFree(d_u_next);
    MPI_Comm_free(&cart_comm);
    MPI_Finalize();
    return 0;
}
