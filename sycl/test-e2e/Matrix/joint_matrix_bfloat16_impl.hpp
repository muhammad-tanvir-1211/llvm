#define TM 8
#define TK 16
#define NUM_SG 8
#define FRAG_M 2
#define FRAG_N 4
#define WARMUP_ITERATIONS 10
#define TOTAL_ITERATIONS 30

#include <chrono>
template <size_t num_sub_groups, size_t frags_m, size_t frags_n, typename T1,
          typename T2, size_t M, size_t N, size_t K>
void matrix_multiply(big_matrix<T1, M, N> &C, big_matrix<T2, M, K> &A,
                     big_matrix<T2, K / 2, N * 2> &B) {
  size_t frags_sg_m = M / TM;
  size_t frags_sg_n = N / TN;
  size_t sg_m = frags_sg_m / frags_m;
  size_t sg_n = frags_sg_n / frags_n;
  size_t total_sg_reqd = sg_m * sg_n;
  constexpr size_t wg_size = SG_SZ * num_sub_groups;
  size_t total_wg_reqd = total_sg_reqd / num_sub_groups;
  assert(sg_m != 0);
  assert(sg_n != 0);
  assert(total_sg_reqd % num_sub_groups == 0);
  buffer<T2, 1> bufA(A.get_data(), range<1>(M * K));
  buffer<T2, 1> bufB(B.get_data(), range<1>(K * N));
  buffer<T1, 1> bufC(C.get_data(), range<1>(M * N));

  queue q;
  // check for correctness
  try {
    q.submit([&](handler &cgh) {
       auto accC = bufC.template get_access<access::mode::read_write>(cgh);
       auto accA = bufA.template get_access<access::mode::read_write>(cgh);
       auto accB = bufB.template get_access<access::mode::read_write>(cgh);

       cgh.parallel_for(
           nd_range<1>({total_wg_reqd * num_sub_groups * SG_SZ},
                       {num_sub_groups * SG_SZ}),
           [=](nd_item<1> spmd_item) [[intel::reqd_sub_group_size(SG_SZ)]]

           {
             // The submatrix API has to be accessed by all the workitems in a
             // subgroup these functions will be called once by the subgroup no
             // code divergence between the workitems
             sub_group sg = spmd_item.get_sub_group();
             const size_t sg_start_id =
                 spmd_item.get_global_linear_id() / SG_SZ;
             const auto sg_starty = sg_start_id % sg_n;
             const auto sg_startx = sg_start_id / sg_n;

             joint_matrix<sub_group, bfloat16, use::a, TM, TK,
                          layout::row_major>
                 sub_a[frags_m];
             // For B, we assume B has been already VNNIed.
             joint_matrix<sub_group, bfloat16, use::b, TK, TN,
                          layout::ext_intel_packed>
                 sub_b[frags_n];
             joint_matrix<sub_group, float, use::accumulator, TM, TN>
                 sub_c[frags_m][frags_n];

             auto ptrA = accA.template get_multi_ptr<access::decorated::yes>();
             auto ptrB = accB.template get_multi_ptr<access::decorated::yes>();
             auto ptrC = accC.template get_multi_ptr<access::decorated::yes>();

             ptrC +=
                 ((sg_startx * TM * frags_m) * N + sg_starty * TN * frags_n);
#pragma unroll
             for (size_t i = 0; i < frags_m; i++) {
               auto new_C = ptrC + i * TM * N;
#pragma unroll
               for (size_t j = 0; j < frags_n; j++) {
                 joint_matrix_load(sg, sub_c[i][j], new_C + j * TN, N,
                                   layout::row_major);
               }
             }

             ptrA += (sg_startx * TM * frags_m) * K;
             ptrB += (sg_starty * TN * 2 * frags_n);
#pragma unroll
             for (int k = 0; k < K / TK;
                  k += 1, ptrA += TK, ptrB += (TK / 2) * (N * 2)) { //
               auto newA = ptrA;
               auto newB = ptrB;
#pragma unroll
               for (int i = 0; i < frags_m; i++, newA += TM * K) {
                 joint_matrix_load(sg, sub_a[i], newA, K);
#pragma unroll
                 for (int j = 0; j < frags_n; j++, newB += TN * 2) {
                   if (i == 0) {
                     joint_matrix_load(sg, sub_b[j], newB, N * 2);
                   }
                   joint_matrix_mad(sg, sub_c[i][j], sub_a[i], sub_b[j],
                                    sub_c[i][j]);
                 }
               }
             }

#pragma unroll
             for (size_t i = 0; i < frags_m; i++) {
               auto new_C = ptrC + i * TM * N;
#pragma unroll
               for (size_t j = 0; j < frags_n; j++) {
                 joint_matrix_store(sg, sub_c[i][j], new_C + j * TN, N,
                                    layout::row_major);
               }
             }
           }); // parallel for
     }).wait();
  } catch (sycl::exception &e) {
    std::cerr << e.what() << std::endl;
  }

  std::vector<T1> dummy(M * N, T1{1});
  buffer<T1, 1> bufD(dummy.data(), range<1>(M * N));
  long long total_time = 0;
  // run warmup and performance measurement iterations
  for (int i = 0; i < TOTAL_ITERATIONS; i++) {
    auto st = std::chrono::high_resolution_clock::now();
    try {
      q.submit([&](handler &cgh) {
         auto accC = bufD.template get_access<access::mode::read_write>(cgh);
         auto accA = bufA.template get_access<access::mode::read_write>(cgh);
         auto accB = bufB.template get_access<access::mode::read_write>(cgh);

         cgh.parallel_for(
             nd_range<1>({total_wg_reqd * num_sub_groups * SG_SZ},
                         {num_sub_groups * SG_SZ}),
             [=](nd_item<1> spmd_item) [[intel::reqd_sub_group_size(SG_SZ)]]

             {
               // The submatrix API has to be accessed by all the workitems in a
               // subgroup these functions will be called once by the subgroup
               // no code divergence between the workitems
               sub_group sg = spmd_item.get_sub_group();
               const size_t sg_start_id =
                   spmd_item.get_global_linear_id() / SG_SZ;
               const auto sg_starty = sg_start_id % sg_n;
               const auto sg_startx = sg_start_id / sg_n;

               joint_matrix<sub_group, bfloat16, use::a, TM, TK,
                            layout::row_major>
                   sub_a[frags_m];
               // For B, we assume B has been already VNNIed.
               joint_matrix<sub_group, bfloat16, use::b, TK, TN,
                            layout::ext_intel_packed>
                   sub_b[frags_n];
               joint_matrix<sub_group, float, use::accumulator, TM, TN>
                   sub_c[frags_m][frags_n];

               auto ptrA =
                   accA.template get_multi_ptr<access::decorated::yes>();
               auto ptrB =
                   accB.template get_multi_ptr<access::decorated::yes>();
               auto ptrC =
                   accC.template get_multi_ptr<access::decorated::yes>();

               ptrC +=
                   ((sg_startx * TM * frags_m) * N + sg_starty * TN * frags_n);
#pragma unroll
               for (size_t i = 0; i < frags_m; i++) {
                 auto new_C = ptrC + i * TM * N;
#pragma unroll
                 for (size_t j = 0; j < frags_n; j++) {
                   joint_matrix_load(sg, sub_c[i][j], new_C + j * TN, N,
                                     layout::row_major);
                 }
               }

               ptrA += (sg_startx * TM * frags_m) * K;
               ptrB += (sg_starty * TN * 2 * frags_n);
#pragma unroll
               for (int k = 0; k < K / TK;
                    k += 1, ptrA += TK, ptrB += (TK / 2) * (N * 2)) { //
                 auto newA = ptrA;
                 auto newB = ptrB;
#pragma unroll
                 for (int i = 0; i < frags_m; i++, newA += TM * K) {
                   joint_matrix_load(sg, sub_a[i], newA, K);
#pragma unroll
                   for (int j = 0; j < frags_n; j++, newB += TN * 2) {
                     if (i == 0) {
                       joint_matrix_load(sg, sub_b[j], newB, N * 2);
                     }
                     joint_matrix_mad(sg, sub_c[i][j], sub_a[i], sub_b[j],
                                      sub_c[i][j]);
                   }
                 }
               }

#pragma unroll
               for (size_t i = 0; i < frags_m; i++) {
                 auto new_C = ptrC + i * TM * N;
#pragma unroll
                 for (size_t j = 0; j < frags_n; j++) {
                   joint_matrix_store(sg, sub_c[i][j], new_C + j * TN, N,
                                      layout::row_major);
                 }
               }
             }); // parallel for
       }).wait();
    } catch (sycl::exception &e) {
      std::cerr << e.what() << std::endl;
    }
    auto end = std::chrono::high_resolution_clock::now();
    size_t curr_time = (end - st).count();
    if (i >= WARMUP_ITERATIONS)
      total_time += curr_time;
  }
  long long average_time = total_time / (TOTAL_ITERATIONS - WARMUP_ITERATIONS);
  std::cout << "Average Time: " << average_time / 1e3 << " us\n";
}

template <size_t MATRIX_M, size_t MATRIX_N, size_t MATRIX_K> int run_gemm() {
  //   constexpr size_t MATRIX_M = TM * 64;
  //   constexpr size_t MATRIX_N = TN * 64;
  //   constexpr size_t MATRIX_K = TK * 64;
  std::vector<bfloat16> A(MATRIX_M * MATRIX_K);
  std::vector<bfloat16> B((MATRIX_K / 2) * (MATRIX_N * 2));
  std::vector<float> C(MATRIX_M * MATRIX_N);
  std::vector<float> D(MATRIX_M * MATRIX_N);

  matrix_rand(MATRIX_M, MATRIX_K, A.data(), bfloat16{2});
  matrix_rand(MATRIX_K / 2, MATRIX_N * 2, B.data(), bfloat16{2});
  matrix_fill(MATRIX_M, MATRIX_N, C.data(), float{1});
  matrix_fill(MATRIX_M, MATRIX_N, D.data(), float{1});

  big_matrix<float, MATRIX_M, MATRIX_N> MC(C.data());
  big_matrix<float, MATRIX_M, MATRIX_N> MD(D.data());
  big_matrix<bfloat16, MATRIX_M, MATRIX_K> MA(A.data());
  big_matrix<bfloat16, MATRIX_K / 2, MATRIX_N * 2> MB(B.data());
  printf("Running MatMul for M:%lu | N:%lu | K:%lu | NUM_SG:%d | FRAGS_M:%d | "
         "FRAGS_N:%d\n",
         MATRIX_M, MATRIX_N, MATRIX_K, NUM_SG, FRAG_M, FRAG_N);
  matrix_multiply<NUM_SG, FRAG_M, FRAG_N>(MC, MA, MB);
  matrix_multiply_ref<bfloat16, bfloat16, float, 2>(
      A.data(), B.data(), D.data(), MATRIX_M, MATRIX_N, MATRIX_K / 2);

  bool res = matrix_compare(MATRIX_M, MATRIX_N, C.data(), D.data());
  std::cout << (res ? "passed" : "failed") << std::endl;
  return !res;
}

int main() {
  int result = -1;
  result = run_gemm<TM * 8, TN * 4, TK * 4>();
  result = run_gemm<TM * 16, TN * 8, TK * 8>();
  result = run_gemm<TM * 32, TN * 16, TK * 16>();
  result = run_gemm<TM * 64, TN * 32, TK * 32>();
  result = run_gemm<TM * 128, TN * 64, TK * 64>();
  result = run_gemm<TM * 256, TN * 128, TK * 128>();
  result = run_gemm<TM * 512, TN * 256, TK * 256>();
  return result;
}
