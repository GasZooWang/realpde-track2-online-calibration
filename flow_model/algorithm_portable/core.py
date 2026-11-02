from ctypes import c_void_p, c_long
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty, empty_strided
from torch._inductor.select_algorithm import extern_kernels
aten = torch.ops.aten
inductor_ops = torch.ops.inductor
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
alloc_from_pool = torch.ops.inductor._alloc_from_pool
reinterpret_tensor = torch.ops.inductor._reinterpret_tensor
triton_per_fused_mean_var_0 = kernels['triton_per_fused_mean_var_0']
import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream
triton_per_fused__to_copy_eq_mean_var_1 = kernels['triton_per_fused__to_copy_eq_mean_var_1']
triton_poi_fused_add_mul_rsub_where_2 = kernels['triton_poi_fused_add_mul_rsub_where_2']
triton_poi_fused_add_clamp_min_div_log_mean_mul_sqrt_sub_var_3 = kernels['triton_poi_fused_add_clamp_min_div_log_mean_mul_sqrt_sub_var_3']
triton_poi_fused_cat_4 = kernels['triton_poi_fused_cat_4']
triton_poi_fused_add_clamp_min_div_mul_sqrt_var_5 = kernels['triton_poi_fused_add_clamp_min_div_mul_sqrt_var_5']
triton_poi_fused_cat_6 = kernels['triton_poi_fused_cat_6']
triton_poi_fused_cat_7 = kernels['triton_poi_fused_cat_7']
triton_poi_fused_cat_8 = kernels['triton_poi_fused_cat_8']
triton_poi_fused__to_copy_eq_mean_9 = kernels['triton_poi_fused__to_copy_eq_mean_9']
triton_poi_fused_div_mean_10 = kernels['triton_poi_fused_div_mean_10']
triton_poi_fused_clamp_11 = kernels['triton_poi_fused_clamp_11']
triton_poi_fused__to_copy_12 = kernels['triton_poi_fused__to_copy_12']
triton_poi_fused__to_copy_convolution_silu_13 = kernels['triton_poi_fused__to_copy_convolution_silu_13']
triton_poi_fused_add_convolution_silu_14 = kernels['triton_poi_fused_add_convolution_silu_14']
triton_poi_fused_convolution_silu_15 = kernels['triton_poi_fused_convolution_silu_15']
triton_poi_fused_add_convolution_silu_16 = kernels['triton_poi_fused_add_convolution_silu_16']
triton_poi_fused_convolution_silu_17 = kernels['triton_poi_fused_convolution_silu_17']
triton_poi_fused_add_convolution_silu_18 = kernels['triton_poi_fused_add_convolution_silu_18']
triton_poi_fused__to_copy__unsafe_index_add_arange_clamp_convolution_mul_rsub_silu_sub_19 = kernels['triton_poi_fused__to_copy__unsafe_index_add_arange_clamp_convolution_mul_rsub_silu_sub_19']
triton_poi_fused__to_copy_add_clone_20 = kernels['triton_poi_fused__to_copy_add_clone_20']
triton_poi_fused__to_copy__unsafe_index_add_arange_clamp_convolution_mul_rsub_silu_sub_21 = kernels['triton_poi_fused__to_copy__unsafe_index_add_arange_clamp_convolution_mul_rsub_silu_sub_21']
triton_poi_fused__to_copy_add_clone_22 = kernels['triton_poi_fused__to_copy_add_clone_22']
triton_poi_fused__log_softmax_add_23 = kernels['triton_poi_fused__log_softmax_add_23']
triton_poi_fused__log_softmax_add_exp_24 = kernels['triton_poi_fused__log_softmax_add_exp_24']
triton_poi_fused_add_mul_sum_tanh_25 = kernels['triton_poi_fused_add_mul_sum_tanh_25']
triton_poi_fused__to_copy_convolution_silu_26 = kernels['triton_poi_fused__to_copy_convolution_silu_26']
triton_poi_fused_add_convolution_silu_27 = kernels['triton_poi_fused_add_convolution_silu_27']
triton_poi_fused_convolution_silu_28 = kernels['triton_poi_fused_convolution_silu_28']
triton_poi_fused_add_convolution_silu_29 = kernels['triton_poi_fused_add_convolution_silu_29']
triton_poi_fused_convolution_silu_30 = kernels['triton_poi_fused_convolution_silu_30']
triton_poi_fused_add_convolution_silu_31 = kernels['triton_poi_fused_add_convolution_silu_31']
triton_poi_fused__to_copy__unsafe_index_add_arange_clamp_convolution_mul_rsub_silu_sub_32 = kernels['triton_poi_fused__to_copy__unsafe_index_add_arange_clamp_convolution_mul_rsub_silu_sub_32']
triton_poi_fused__to_copy_add_clone_33 = kernels['triton_poi_fused__to_copy_add_clone_33']
triton_poi_fused__to_copy__unsafe_index_add_arange_clamp_convolution_mul_rsub_silu_sub_34 = kernels['triton_poi_fused__to_copy__unsafe_index_add_arange_clamp_convolution_mul_rsub_silu_sub_34']
triton_poi_fused__to_copy_add_clone_35 = kernels['triton_poi_fused__to_copy_add_clone_35']
triton_poi_fused_add_convolution_silu_36 = kernels['triton_poi_fused_add_convolution_silu_36']
triton_poi_fused_add_mul_sub_37 = kernels['triton_poi_fused_add_mul_sub_37']
triton_per_fused_mean_sub_var_38 = kernels['triton_per_fused_mean_sub_var_38']
triton_poi_fused_bitwise_and_clamp_min_div_exp_gt_mean_mul_sqrt_sub_var_where_39 = kernels['triton_poi_fused_bitwise_and_clamp_min_div_exp_gt_mean_mul_sqrt_sub_var_where_39']
triton_per_fused_add_mean_mul_pow_sub_40 = kernels['triton_per_fused_add_mean_mul_pow_sub_40']
triton_poi_fused_add_cat_clamp_min_div_expm1_gt_mean_mul_pow_sub_sum_where_zeros_like_41 = kernels['triton_poi_fused_add_cat_clamp_min_div_expm1_gt_mean_mul_pow_sub_sum_where_zeros_like_41']
triton_poi_fused_add_cat_clamp_min_expm1_log1p_mul_sum_42 = kernels['triton_poi_fused_add_cat_clamp_min_expm1_log1p_mul_sum_42']
triton_poi_fused_add_div_erf_exp_mul_sub_tanh_43 = kernels['triton_poi_fused_add_div_erf_exp_mul_sub_tanh_43']
triton_poi_fused_add_div_exp_mean_mul_sub_sum_44 = kernels['triton_poi_fused_add_div_exp_mean_mul_sub_sum_44']
triton_per_fused_mean_45 = kernels['triton_per_fused_mean_45']
triton_poi_fused_cat_46 = kernels['triton_poi_fused_cat_46']
triton_poi_fused_add_clone_full_like_gt_maximum_mul_sub_where_zeros_like_47 = kernels['triton_poi_fused_add_clone_full_like_gt_maximum_mul_sub_where_zeros_like_47']
triton_poi_fused_add_full_like_gt_maximum_mul_sub_where_zeros_like_48 = kernels['triton_poi_fused_add_full_like_gt_maximum_mul_sub_where_zeros_like_48']
triton_poi_fused_add_clamp_min_div_mul_sub_49 = kernels['triton_poi_fused_add_clamp_min_div_mul_sub_49']
triton_poi_fused_add_div_exp_mul_sub_tanh_50 = kernels['triton_poi_fused_add_div_exp_mul_sub_tanh_50']
triton_poi_fused_div_erf_exp_mul_neg_pow_sub_sum_tanh_51 = kernels['triton_poi_fused_div_erf_exp_mul_neg_pow_sub_sum_tanh_51']
triton_poi_fused_clamp_min_div_exp_mul_neg_sub_52 = kernels['triton_poi_fused_clamp_min_div_exp_mul_neg_sub_52']
triton_poi_fused_abs_add_bitwise_and_clamp_clamp_min_div_gt_lt_mul_neg_pow_sub_where_53 = kernels['triton_poi_fused_abs_add_bitwise_and_clamp_clamp_min_div_gt_lt_mul_neg_pow_sub_where_53']
triton_poi_fused_abs_add_bitwise_and_clamp_eq_ge_mul_ne_sub_54 = kernels['triton_poi_fused_abs_add_bitwise_and_clamp_eq_ge_mul_ne_sub_54']
triton_poi_fused_add_div_erf_exp_mul_sub_tanh_55 = kernels['triton_poi_fused_add_div_erf_exp_mul_sub_tanh_55']
triton_poi_fused_add_clamp_clamp_min_div_exp_mul_neg_sub_sum_56 = kernels['triton_poi_fused_add_clamp_clamp_min_div_exp_mul_neg_sub_sum_56']
triton_poi_fused_add_bitwise_and_clamp_clamp_min_div_gt_mul_sub_where_57 = kernels['triton_poi_fused_add_bitwise_and_clamp_clamp_min_div_gt_mul_sub_where_57']
triton_poi_fused_cat_58 = kernels['triton_poi_fused_cat_58']

def call(args):
    (arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1) = args
    args.clear()
    assert_size_stride(arg0_1, (32, 196, 3, 3), (1764, 1, 588, 196))
    assert_size_stride(arg1_1, (32,), (1,))
    assert_size_stride(arg2_1, (32, 32, 3, 3), (288, 1, 96, 32))
    assert_size_stride(arg3_1, (32,), (1,))
    assert_size_stride(arg4_1, (32, 32, 3, 3), (288, 1, 96, 32))
    assert_size_stride(arg5_1, (32,), (1,))
    assert_size_stride(arg6_1, (64, 32, 3, 3), (288, 1, 96, 32))
    assert_size_stride(arg7_1, (64,), (1,))
    assert_size_stride(arg8_1, (64, 64, 3, 3), (576, 1, 192, 64))
    assert_size_stride(arg9_1, (64,), (1,))
    assert_size_stride(arg10_1, (64, 64, 3, 3), (576, 1, 192, 64))
    assert_size_stride(arg11_1, (64,), (1,))
    assert_size_stride(arg12_1, (96, 64, 3, 3), (576, 1, 192, 64))
    assert_size_stride(arg13_1, (96,), (1,))
    assert_size_stride(arg14_1, (96, 96, 3, 3), (864, 1, 288, 96))
    assert_size_stride(arg15_1, (96,), (1,))
    assert_size_stride(arg16_1, (96, 96, 3, 3), (864, 1, 288, 96))
    assert_size_stride(arg17_1, (96,), (1,))
    assert_size_stride(arg18_1, (96, 96, 3, 3), (864, 1, 288, 96))
    assert_size_stride(arg19_1, (96,), (1,))
    assert_size_stride(arg20_1, (96, 96, 3, 3), (864, 1, 288, 96))
    assert_size_stride(arg21_1, (96,), (1,))
    assert_size_stride(arg22_1, (64, 96, 1, 1), (96, 1, 96, 96))
    assert_size_stride(arg23_1, (64,), (1,))
    assert_size_stride(arg24_1, (64, 64, 3, 3), (576, 1, 192, 64))
    assert_size_stride(arg25_1, (64,), (1,))
    assert_size_stride(arg26_1, (64, 64, 3, 3), (576, 1, 192, 64))
    assert_size_stride(arg27_1, (64,), (1,))
    assert_size_stride(arg28_1, (32, 64, 1, 1), (64, 1, 64, 64))
    assert_size_stride(arg29_1, (32,), (1,))
    assert_size_stride(arg30_1, (32, 32, 3, 3), (288, 1, 96, 32))
    assert_size_stride(arg31_1, (32,), (1,))
    assert_size_stride(arg32_1, (32, 32, 3, 3), (288, 1, 96, 32))
    assert_size_stride(arg33_1, (32,), (1,))
    assert_size_stride(arg34_1, (124, 32, 3, 3), (288, 1, 96, 32))
    assert_size_stride(arg35_1, (124,), (1,))
    assert_size_stride(arg36_1, (24, 196, 3, 3), (1764, 1, 588, 196))
    assert_size_stride(arg37_1, (24,), (1,))
    assert_size_stride(arg38_1, (24, 24, 3, 3), (216, 1, 72, 24))
    assert_size_stride(arg39_1, (24,), (1,))
    assert_size_stride(arg40_1, (24, 24, 3, 3), (216, 1, 72, 24))
    assert_size_stride(arg41_1, (24,), (1,))
    assert_size_stride(arg42_1, (48, 24, 3, 3), (216, 1, 72, 24))
    assert_size_stride(arg43_1, (48,), (1,))
    assert_size_stride(arg44_1, (48, 48, 3, 3), (432, 1, 144, 48))
    assert_size_stride(arg45_1, (48,), (1,))
    assert_size_stride(arg46_1, (48, 48, 3, 3), (432, 1, 144, 48))
    assert_size_stride(arg47_1, (48,), (1,))
    assert_size_stride(arg48_1, (72, 48, 3, 3), (432, 1, 144, 48))
    assert_size_stride(arg49_1, (72,), (1,))
    assert_size_stride(arg50_1, (72, 72, 3, 3), (648, 1, 216, 72))
    assert_size_stride(arg51_1, (72,), (1,))
    assert_size_stride(arg52_1, (72, 72, 3, 3), (648, 1, 216, 72))
    assert_size_stride(arg53_1, (72,), (1,))
    assert_size_stride(arg54_1, (72, 72, 3, 3), (648, 1, 216, 72))
    assert_size_stride(arg55_1, (72,), (1,))
    assert_size_stride(arg56_1, (72, 72, 3, 3), (648, 1, 216, 72))
    assert_size_stride(arg57_1, (72,), (1,))
    assert_size_stride(arg58_1, (48, 72, 1, 1), (72, 1, 72, 72))
    assert_size_stride(arg59_1, (48,), (1,))
    assert_size_stride(arg60_1, (48, 48, 3, 3), (432, 1, 144, 48))
    assert_size_stride(arg61_1, (48,), (1,))
    assert_size_stride(arg62_1, (48, 48, 3, 3), (432, 1, 144, 48))
    assert_size_stride(arg63_1, (48,), (1,))
    assert_size_stride(arg64_1, (24, 48, 1, 1), (48, 1, 48, 48))
    assert_size_stride(arg65_1, (24,), (1,))
    assert_size_stride(arg66_1, (24, 24, 3, 3), (216, 1, 72, 24))
    assert_size_stride(arg67_1, (24,), (1,))
    assert_size_stride(arg68_1, (24, 24, 3, 3), (216, 1, 72, 24))
    assert_size_stride(arg69_1, (24,), (1,))
    assert_size_stride(arg70_1, (360, 24, 3, 3), (216, 1, 72, 24))
    assert_size_stride(arg71_1, (360,), (1,))
    assert_size_stride(arg72_1, (1, 1, 1, 1, 3), (3, 3, 3, 3, 1))
    assert_size_stride(arg73_1, (1, 1, 1, 1, 3), (3, 3, 3, 3, 1))
    assert_size_stride(arg74_1, (1, 1, 1, 1, 3), (3, 3, 3, 3, 1))
    assert_size_stride(arg75_1, (1, 1, 1, 1, 3), (3, 3, 3, 3, 1))
    assert_size_stride(arg76_1, (1, 1, 1, 1, 3), (3, 3, 3, 3, 1))
    assert_size_stride(arg77_1, (1, 1, 1, 1, 3), (3, 3, 3, 3, 1))
    assert_size_stride(arg78_1, (3, 1, 1, 1, 1), (1, 1, 1, 1, 1))
    assert_size_stride(arg79_1, (1, 2, 1, 1), (2, 1, 1, 1))
    assert_size_stride(arg80_1, (3,), (1,))
    assert_size_stride(arg81_1, (3,), (1,))
    assert_size_stride(arg82_1, (1, 1, 2, 1, 1), (2, 2, 1, 1, 1))
    assert_size_stride(arg83_1, (1, 20, 32, 64, 3), (122880, 6144, 192, 3, 1))
    assert_size_stride(arg84_1, (1, 20, 32, 64, 3), (122880, 6144, 192, 3, 1))
    assert_size_stride(arg85_1, (1, 20, 32, 64, 3), (122880, 6144, 192, 3, 1))
    assert_size_stride(arg86_1, (1, 20, 32, 64, 3), (122880, 6144, 192, 3, 1))
    assert_size_stride(arg87_1, (1, 20, 32, 64, 3), (122880, 6144, 192, 3, 1))
    assert_size_stride(arg88_1, (), ())
    assert_size_stride(arg89_1, (), ())
    assert_size_stride(arg90_1, (1, 2, 32, 64), (4096, 2048, 64, 1))
    assert_size_stride(arg91_1, (1, 2, 32, 64), (4096, 2048, 64, 1))
    assert_size_stride(arg92_1, (3, 1, 2, 32, 64), (4096, 4096, 2048, 64, 1))
    assert_size_stride(arg93_1, (3, 1, 2, 32, 64), (4096, 4096, 2048, 64, 1))
    assert_size_stride(arg94_1, (3, 1, 2, 32, 64), (4096, 4096, 2048, 64, 1))
    assert_size_stride(arg95_1, (3, 1, 2, 32, 64), (4096, 4096, 2048, 64, 1))
    with torch.cuda._DeviceGuard(DEVICE_INDEX):
        torch.cuda.set_device(DEVICE_INDEX)
        buf0 = empty_strided((1, 2, 32, 64), (4096, 1, 128, 2), device='cuda', dtype=torch.float32)
        buf6 = empty_strided((1, 2, 32, 64), (4096, 1, 128, 2), device='cuda', dtype=torch.float32)
        buf12 = empty_strided((1, 2, 32, 64), (4096, 1, 128, 2), device='cuda', dtype=torch.float32)
        buf15 = empty_strided((1, 2, 32, 64), (4096, 1, 128, 2), device='cuda', dtype=torch.float32)
        stream0 = get_cuda_stream(DEVICE_INDEX)
        triton_per_fused_mean_var_0.run(arg84_1, arg76_1, arg77_1, arg87_1, arg75_1, buf0, buf6, buf12, buf15, 4096, 20, grid=grid(4096), stream=stream0)
        del arg75_1
        del arg87_1
        buf3 = empty_strided((1, 2, 32, 64), (4096, 1, 128, 2), device='cuda', dtype=torch.float32)
        buf9 = empty_strided((1, 2, 32, 64), (4096, 1, 128, 2), device='cuda', dtype=torch.float32)
        buf20 = empty_strided((1, 2, 32, 64), (4096, 1, 128, 2), device='cuda', dtype=torch.float32)
        triton_per_fused__to_copy_eq_mean_var_1.run(arg83_1, arg72_1, arg73_1, arg74_1, buf3, buf9, buf20, 4096, 20, grid=grid(4096), stream=stream0)
        buf10 = empty((3, 1, 2, 32, 64), device='cuda', dtype=torch.float32)
        buf11 = empty((3, 1, 2, 32, 64), device='cuda', dtype=torch.float32)
        buf13 = empty((3, 1, 2, 32, 64), device='cuda', dtype=torch.float32)
        buf17 = empty((3, 1, 2, 32, 64), device='cuda', dtype=torch.float32)
        triton_poi_fused_add_mul_rsub_where_2.run(arg88_1, buf9, arg92_1, arg78_1, buf3, arg93_1, arg89_1, arg94_1, buf12, arg90_1, arg95_1, buf15, arg91_1, buf10, buf11, buf13, buf17, 12288, grid=grid(12288), stream=stream0)
        del arg78_1
        del arg88_1
        del arg89_1
        del arg90_1
        del arg91_1
        del arg92_1
        del arg93_1
        del arg94_1
        del arg95_1
        del buf12
        buf1 = buf0
        del buf0
        buf8 = buf6
        del buf6
        buf43 = empty((1, 196, 32, 64), device='cuda', dtype=torch.float32)
        buf25 = reinterpret_tensor(buf43, (1, 2, 32, 64), (401408, 2048, 64, 1), 327680)
        buf29 = reinterpret_tensor(buf43, (1, 2, 32, 64), (401408, 2048, 64, 1), 344064)
        buf33 = reinterpret_tensor(buf43, (1, 2, 32, 64), (401408, 2048, 64, 1), 360448)
        buf37 = reinterpret_tensor(buf43, (1, 2, 32, 64), (401408, 2048, 64, 1), 376832)
        buf39 = reinterpret_tensor(buf43, (1, 2, 32, 64), (401408, 2048, 64, 1), 385024)
        buf42 = reinterpret_tensor(buf43, (1, 2, 32, 64), (401408, 2048, 64, 1), 397312)
        triton_poi_fused_add_clamp_min_div_log_mean_mul_sqrt_sub_var_3.run(buf1, buf8, arg83_1, arg72_1, arg73_1, arg74_1, buf10, buf3, arg79_1, buf25, buf29, buf33, buf37, buf39, buf42, 2048, 2, grid=grid(2048, 2), stream=stream0)
        buf21 = reinterpret_tensor(buf43, (1, 40, 32, 64), (401408, 2048, 64, 1), 0)
        buf22 = reinterpret_tensor(buf43, (1, 40, 32, 64), (401408, 2048, 64, 1), 81920)
        buf23 = reinterpret_tensor(buf43, (1, 40, 32, 64), (401408, 2048, 64, 1), 163840)
        buf24 = reinterpret_tensor(buf43, (1, 40, 32, 64), (401408, 2048, 64, 1), 245760)
        triton_poi_fused_cat_4.run(arg83_1, arg72_1, arg73_1, arg74_1, buf1, buf3, buf8, arg84_1, arg76_1, arg77_1, arg85_1, arg86_1, buf21, buf22, buf23, buf24, 2048, 40, grid=grid(2048, 40), stream=stream0)
        del arg72_1
        del arg73_1
        del arg74_1
        del arg83_1
        buf26 = reinterpret_tensor(buf43, (1, 2, 32, 64), (401408, 2048, 64, 1), 331776)
        buf27 = reinterpret_tensor(buf43, (1, 2, 32, 64), (401408, 2048, 64, 1), 335872)
        buf30 = reinterpret_tensor(buf43, (1, 2, 32, 64), (401408, 2048, 64, 1), 348160)
        buf31 = reinterpret_tensor(buf43, (1, 2, 32, 64), (401408, 2048, 64, 1), 352256)
        buf34 = reinterpret_tensor(buf43, (1, 2, 32, 64), (401408, 2048, 64, 1), 364544)
        buf35 = reinterpret_tensor(buf43, (1, 2, 32, 64), (401408, 2048, 64, 1), 368640)
        triton_poi_fused_add_clamp_min_div_mul_sqrt_var_5.run(buf11, buf3, buf8, buf13, buf26, buf27, buf30, buf31, buf34, buf35, 2, 2048, grid=grid(2, 2048), stream=stream0)
        buf28 = reinterpret_tensor(buf43, (1, 2, 32, 64), (401408, 2048, 64, 1), 339968)
        triton_poi_fused_cat_6.run(buf17, buf28, 4096, grid=grid(4096), stream=stream0)
        buf32 = reinterpret_tensor(buf43, (1, 2, 32, 64), (401408, 2048, 64, 1), 356352)
        triton_poi_fused_cat_7.run(buf17, buf32, 4096, grid=grid(4096), stream=stream0)
        buf36 = reinterpret_tensor(buf43, (1, 2, 32, 64), (401408, 2048, 64, 1), 372736)
        triton_poi_fused_cat_8.run(buf17, buf36, 4096, grid=grid(4096), stream=stream0)
        buf38 = reinterpret_tensor(buf43, (1, 2, 32, 64), (401408, 2048, 64, 1), 380928)
        triton_poi_fused__to_copy_eq_mean_9.run(buf20, buf38, 2, 2048, grid=grid(2, 2048), stream=stream0)
        buf40 = reinterpret_tensor(buf43, (1, 2, 32, 64), (401408, 2048, 64, 1), 389120)
        buf41 = reinterpret_tensor(buf43, (1, 2, 32, 64), (401408, 2048, 64, 1), 393216)
        triton_poi_fused_div_mean_10.run(buf1, arg79_1, buf9, buf40, buf41, 2, 2048, grid=grid(2, 2048), stream=stream0)
        del arg79_1
        buf44 = empty_strided((1, 196, 32, 64), (401408, 1, 12544, 196), device='cuda', dtype=torch.float32)
        triton_poi_fused_clamp_11.run(buf43, buf44, 196, 2048, grid=grid(196, 2048), stream=stream0)
        del buf21
        del buf22
        del buf23
        del buf24
        del buf25
        del buf26
        del buf27
        del buf28
        del buf29
        del buf30
        del buf31
        del buf32
        del buf33
        del buf34
        del buf35
        del buf36
        del buf37
        del buf38
        del buf39
        del buf40
        del buf41
        del buf42
        del buf43
        buf45 = empty_strided((1, 196, 32, 64), (401408, 1, 12544, 196), device='cuda', dtype=torch.float16)
        buf95 = empty_strided((1, 196, 32, 64), (401408, 1, 12544, 196), device='cuda', dtype=torch.float16)
        triton_poi_fused__to_copy_12.run(buf44, buf45, buf95, 401408, grid=grid(401408), stream=stream0)
        buf46 = extern_kernels.convolution(buf45, arg36_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf46, (1, 24, 32, 64), (49152, 1, 1536, 24))
        del arg36_1
        del buf45
        buf47 = buf46
        del buf46
        triton_poi_fused__to_copy_convolution_silu_13.run(buf47, arg37_1, 49152, grid=grid(49152), stream=stream0)
        del arg37_1
        buf48 = extern_kernels.convolution(buf47, arg38_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf48, (1, 24, 32, 64), (49152, 1, 1536, 24))
        del arg38_1
        buf49 = buf48
        del buf48
        triton_poi_fused__to_copy_convolution_silu_13.run(buf49, arg39_1, 49152, grid=grid(49152), stream=stream0)
        del arg39_1
        buf50 = extern_kernels.convolution(buf49, arg40_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf50, (1, 24, 32, 64), (49152, 1, 1536, 24))
        del arg40_1
        del buf49
        buf51 = buf47
        del buf47
        triton_poi_fused_add_convolution_silu_14.run(buf51, buf50, arg41_1, 49152, grid=grid(49152), stream=stream0)
        del arg41_1
        del buf50
        buf52 = extern_kernels.convolution(buf51, arg42_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf52, (1, 48, 16, 32), (24576, 1, 1536, 48))
        del arg42_1
        buf53 = buf52
        del buf52
        triton_poi_fused_convolution_silu_15.run(buf53, arg43_1, 24576, grid=grid(24576), stream=stream0)
        del arg43_1
        buf54 = extern_kernels.convolution(buf53, arg44_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf54, (1, 48, 16, 32), (24576, 1, 1536, 48))
        del arg44_1
        buf55 = buf54
        del buf54
        triton_poi_fused_convolution_silu_15.run(buf55, arg45_1, 24576, grid=grid(24576), stream=stream0)
        del arg45_1
        buf56 = extern_kernels.convolution(buf55, arg46_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf56, (1, 48, 16, 32), (24576, 1, 1536, 48))
        del arg46_1
        del buf55
        buf57 = buf53
        del buf53
        triton_poi_fused_add_convolution_silu_16.run(buf57, buf56, arg47_1, 24576, grid=grid(24576), stream=stream0)
        del arg47_1
        del buf56
        buf58 = extern_kernels.convolution(buf57, arg48_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf58, (1, 72, 8, 16), (9216, 1, 1152, 72))
        del arg48_1
        buf59 = buf58
        del buf58
        triton_poi_fused_convolution_silu_17.run(buf59, arg49_1, 9216, grid=grid(9216), stream=stream0)
        del arg49_1
        buf60 = extern_kernels.convolution(buf59, arg50_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf60, (1, 72, 8, 16), (9216, 1, 1152, 72))
        del arg50_1
        buf61 = buf60
        del buf60
        triton_poi_fused_convolution_silu_17.run(buf61, arg51_1, 9216, grid=grid(9216), stream=stream0)
        del arg51_1
        buf62 = extern_kernels.convolution(buf61, arg52_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf62, (1, 72, 8, 16), (9216, 1, 1152, 72))
        del arg52_1
        del buf61
        buf63 = buf59
        del buf59
        triton_poi_fused_add_convolution_silu_18.run(buf63, buf62, arg53_1, 9216, grid=grid(9216), stream=stream0)
        del arg53_1
        del buf62
        buf64 = extern_kernels.convolution(buf63, arg54_1, stride=(1, 1), padding=(2, 2), dilation=(2, 2), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf64, (1, 72, 8, 16), (9216, 1, 1152, 72))
        del arg54_1
        buf65 = buf64
        del buf64
        triton_poi_fused_convolution_silu_17.run(buf65, arg55_1, 9216, grid=grid(9216), stream=stream0)
        del arg55_1
        buf66 = extern_kernels.convolution(buf65, arg56_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf66, (1, 72, 8, 16), (9216, 1, 1152, 72))
        del arg56_1
        del buf65
        buf67 = buf63
        del buf63
        triton_poi_fused_add_convolution_silu_18.run(buf67, buf66, arg57_1, 9216, grid=grid(9216), stream=stream0)
        del arg57_1
        del buf66
        buf68 = extern_kernels.convolution(buf67, arg58_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf68, (1, 48, 8, 16), (6144, 1, 768, 48))
        del arg58_1
        del buf67
        buf69 = empty((1, 48, 16, 32), device='cuda', dtype=torch.float32)
        buf73 = buf69
        del buf69
        triton_poi_fused__to_copy__unsafe_index_add_arange_clamp_convolution_mul_rsub_silu_sub_19.run(buf73, buf68, arg59_1, 24576, grid=grid(24576), stream=stream0)
        del arg59_1
        del buf68
        buf74 = buf57
        del buf57
        triton_poi_fused__to_copy_add_clone_20.run(buf74, buf73, 512, 48, grid=grid(512, 48), stream=stream0)
        del buf73
        buf75 = extern_kernels.convolution(buf74, arg60_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf75, (1, 48, 16, 32), (24576, 1, 1536, 48))
        del arg60_1
        buf76 = buf75
        del buf75
        triton_poi_fused_convolution_silu_15.run(buf76, arg61_1, 24576, grid=grid(24576), stream=stream0)
        del arg61_1
        buf77 = extern_kernels.convolution(buf76, arg62_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf77, (1, 48, 16, 32), (24576, 1, 1536, 48))
        del arg62_1
        del buf76
        buf78 = buf74
        del buf74
        triton_poi_fused_add_convolution_silu_16.run(buf78, buf77, arg63_1, 24576, grid=grid(24576), stream=stream0)
        del arg63_1
        del buf77
        buf79 = extern_kernels.convolution(buf78, arg64_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf79, (1, 24, 16, 32), (12288, 1, 768, 24))
        del arg64_1
        del buf78
        buf80 = empty((1, 24, 32, 64), device='cuda', dtype=torch.float32)
        buf84 = buf80
        del buf80
        triton_poi_fused__to_copy__unsafe_index_add_arange_clamp_convolution_mul_rsub_silu_sub_21.run(buf84, buf79, arg65_1, 49152, grid=grid(49152), stream=stream0)
        del arg65_1
        del buf79
        buf85 = buf51
        del buf51
        triton_poi_fused__to_copy_add_clone_22.run(buf85, buf84, 2048, 24, grid=grid(2048, 24), stream=stream0)
        del buf84
        buf86 = extern_kernels.convolution(buf85, arg66_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf86, (1, 24, 32, 64), (49152, 1, 1536, 24))
        del arg66_1
        buf87 = buf86
        del buf86
        triton_poi_fused__to_copy_convolution_silu_13.run(buf87, arg67_1, 49152, grid=grid(49152), stream=stream0)
        del arg67_1
        buf88 = extern_kernels.convolution(buf87, arg68_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf88, (1, 24, 32, 64), (49152, 1, 1536, 24))
        del arg68_1
        del buf87
        buf89 = buf85
        del buf85
        triton_poi_fused_add_convolution_silu_14.run(buf89, buf88, arg69_1, 49152, grid=grid(49152), stream=stream0)
        del arg69_1
        del buf88
        buf90 = extern_kernels.convolution(buf89, arg70_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf90, (1, 360, 32, 64), (737280, 1, 23040, 360))
        del arg70_1
        del buf89
        buf91 = empty_strided((1, 1, 20, 2, 32, 64), (81920, 81920, 2, 1, 2560, 40), device='cuda', dtype=torch.float32)
        buf92 = empty_strided((1, 1, 20, 2, 32, 64), (81920, 81920, 2, 1, 2560, 40), device='cuda', dtype=torch.float32)
        triton_poi_fused__log_softmax_add_23.run(buf90, arg71_1, arg81_1, buf91, buf92, 81920, grid=grid(81920), stream=stream0)
        buf93 = empty_strided((1, 3, 20, 2, 32, 64), (245760, 40, 2, 1, 7680, 120), device='cuda', dtype=torch.float32)
        triton_poi_fused__log_softmax_add_exp_24.run(buf90, arg71_1, arg81_1, buf91, buf92, buf93, 245760, grid=grid(245760), stream=stream0)
        del arg81_1
        buf94 = reinterpret_tensor(buf92, (1, 20, 2, 32, 64), (81920, 2, 1, 2560, 40), 0)
        del buf92
        buf155 = reinterpret_tensor(buf91, (1, 20, 2, 32, 64), (81920, 2, 1, 2560, 40), 0)
        del buf91
        triton_poi_fused_add_mul_sum_tanh_25.run(arg80_1, buf90, arg71_1, buf93, buf94, buf155, 81920, grid=grid(81920), stream=stream0)
        buf96 = extern_kernels.convolution(buf95, arg0_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf96, (1, 32, 32, 64), (65536, 1, 2048, 32))
        del arg0_1
        del buf95
        buf97 = buf96
        del buf96
        triton_poi_fused__to_copy_convolution_silu_26.run(buf97, arg1_1, 65536, grid=grid(65536), stream=stream0)
        del arg1_1
        buf98 = extern_kernels.convolution(buf97, arg2_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf98, (1, 32, 32, 64), (65536, 1, 2048, 32))
        del arg2_1
        buf99 = buf98
        del buf98
        triton_poi_fused__to_copy_convolution_silu_26.run(buf99, arg3_1, 65536, grid=grid(65536), stream=stream0)
        del arg3_1
        buf100 = extern_kernels.convolution(buf99, arg4_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf100, (1, 32, 32, 64), (65536, 1, 2048, 32))
        del arg4_1
        del buf99
        buf101 = buf100
        del buf100
        triton_poi_fused_add_convolution_silu_27.run(buf101, buf97, arg5_1, 65536, grid=grid(65536), stream=stream0)
        del arg5_1
        del buf97
        buf102 = extern_kernels.convolution(buf101, arg6_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf102, (1, 64, 16, 32), (32768, 1, 2048, 64))
        del arg6_1
        buf103 = buf102
        del buf102
        triton_poi_fused_convolution_silu_28.run(buf103, arg7_1, 32768, grid=grid(32768), stream=stream0)
        del arg7_1
        buf104 = extern_kernels.convolution(buf103, arg8_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf104, (1, 64, 16, 32), (32768, 1, 2048, 64))
        del arg8_1
        buf105 = buf104
        del buf104
        triton_poi_fused_convolution_silu_28.run(buf105, arg9_1, 32768, grid=grid(32768), stream=stream0)
        del arg9_1
        buf106 = extern_kernels.convolution(buf105, arg10_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf106, (1, 64, 16, 32), (32768, 1, 2048, 64))
        del arg10_1
        del buf105
        buf107 = buf103
        del buf103
        triton_poi_fused_add_convolution_silu_29.run(buf107, buf106, arg11_1, 32768, grid=grid(32768), stream=stream0)
        del arg11_1
        del buf106
        buf108 = extern_kernels.convolution(buf107, arg12_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf108, (1, 96, 8, 16), (12288, 1, 1536, 96))
        del arg12_1
        buf109 = buf108
        del buf108
        triton_poi_fused_convolution_silu_30.run(buf109, arg13_1, 12288, grid=grid(12288), stream=stream0)
        del arg13_1
        buf110 = extern_kernels.convolution(buf109, arg14_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf110, (1, 96, 8, 16), (12288, 1, 1536, 96))
        del arg14_1
        buf111 = buf110
        del buf110
        triton_poi_fused_convolution_silu_30.run(buf111, arg15_1, 12288, grid=grid(12288), stream=stream0)
        del arg15_1
        buf112 = extern_kernels.convolution(buf111, arg16_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf112, (1, 96, 8, 16), (12288, 1, 1536, 96))
        del arg16_1
        del buf111
        buf113 = buf109
        del buf109
        triton_poi_fused_add_convolution_silu_31.run(buf113, buf112, arg17_1, 12288, grid=grid(12288), stream=stream0)
        del arg17_1
        del buf112
        buf114 = extern_kernels.convolution(buf113, arg18_1, stride=(1, 1), padding=(2, 2), dilation=(2, 2), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf114, (1, 96, 8, 16), (12288, 1, 1536, 96))
        del arg18_1
        buf115 = buf114
        del buf114
        triton_poi_fused_convolution_silu_30.run(buf115, arg19_1, 12288, grid=grid(12288), stream=stream0)
        del arg19_1
        buf116 = extern_kernels.convolution(buf115, arg20_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf116, (1, 96, 8, 16), (12288, 1, 1536, 96))
        del arg20_1
        del buf115
        buf117 = buf113
        del buf113
        triton_poi_fused_add_convolution_silu_31.run(buf117, buf116, arg21_1, 12288, grid=grid(12288), stream=stream0)
        del arg21_1
        del buf116
        buf118 = extern_kernels.convolution(buf117, arg22_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf118, (1, 64, 8, 16), (8192, 1, 1024, 64))
        del arg22_1
        del buf117
        buf119 = empty((1, 64, 16, 32), device='cuda', dtype=torch.float32)
        buf123 = buf119
        del buf119
        triton_poi_fused__to_copy__unsafe_index_add_arange_clamp_convolution_mul_rsub_silu_sub_32.run(buf123, buf118, arg23_1, 32768, grid=grid(32768), stream=stream0)
        del arg23_1
        del buf118
        buf124 = buf107
        del buf107
        triton_poi_fused__to_copy_add_clone_33.run(buf124, buf123, 512, 64, grid=grid(512, 64), stream=stream0)
        del buf123
        buf125 = extern_kernels.convolution(buf124, arg24_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf125, (1, 64, 16, 32), (32768, 1, 2048, 64))
        del arg24_1
        buf126 = buf125
        del buf125
        triton_poi_fused_convolution_silu_28.run(buf126, arg25_1, 32768, grid=grid(32768), stream=stream0)
        del arg25_1
        buf127 = extern_kernels.convolution(buf126, arg26_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf127, (1, 64, 16, 32), (32768, 1, 2048, 64))
        del arg26_1
        del buf126
        buf128 = buf124
        del buf124
        triton_poi_fused_add_convolution_silu_29.run(buf128, buf127, arg27_1, 32768, grid=grid(32768), stream=stream0)
        del arg27_1
        del buf127
        buf129 = extern_kernels.convolution(buf128, arg28_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf129, (1, 32, 16, 32), (16384, 1, 1024, 32))
        del arg28_1
        del buf128
        buf130 = empty((1, 32, 32, 64), device='cuda', dtype=torch.float32)
        buf134 = buf130
        del buf130
        triton_poi_fused__to_copy__unsafe_index_add_arange_clamp_convolution_mul_rsub_silu_sub_34.run(buf134, buf129, arg29_1, 65536, grid=grid(65536), stream=stream0)
        del arg29_1
        del buf129
        buf135 = buf101
        del buf101
        triton_poi_fused__to_copy_add_clone_35.run(buf135, buf134, 2048, 32, grid=grid(2048, 32), stream=stream0)
        del buf134
        buf136 = extern_kernels.convolution(buf135, arg30_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf136, (1, 32, 32, 64), (65536, 1, 2048, 32))
        del arg30_1
        buf137 = buf136
        del buf136
        triton_poi_fused__to_copy_convolution_silu_26.run(buf137, arg31_1, 65536, grid=grid(65536), stream=stream0)
        del arg31_1
        buf138 = extern_kernels.convolution(buf137, arg32_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf138, (1, 32, 32, 64), (65536, 1, 2048, 32))
        del arg32_1
        del buf137
        buf139 = buf135
        del buf135
        triton_poi_fused_add_convolution_silu_36.run(buf139, buf138, arg33_1, 65536, grid=grid(65536), stream=stream0)
        del arg33_1
        del buf138
        buf140 = extern_kernels.convolution(buf139, arg34_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf140, (1, 124, 32, 64), (253952, 1, 7936, 124))
        del arg34_1
        del buf139
        buf141 = empty_strided((1, 20, 2, 32, 64), (81920, 4096, 1, 128, 2), device='cuda', dtype=torch.float32)
        triton_poi_fused_add_mul_sub_37.run(arg84_1, arg76_1, arg77_1, buf1, buf3, buf8, buf140, arg35_1, buf141, 81920, grid=grid(81920), stream=stream0)
        buf142 = reinterpret_tensor(buf9, (1, 1, 2, 32, 64), (4096, 4096, 1, 128, 2), 0)
        del buf9
        buf144 = reinterpret_tensor(buf20, (1, 1, 2, 32, 64), (4096, 4096, 1, 128, 2), 0)
        del buf20
        triton_per_fused_mean_sub_var_38.run(buf141, buf142, buf144, 4096, 20, grid=grid(4096), stream=stream0)
        buf146 = empty_strided((1, 20, 2, 32, 64), (81920, 4096, 1, 128, 2), device='cuda', dtype=torch.float32)
        triton_poi_fused_bitwise_and_clamp_min_div_exp_gt_mean_mul_sqrt_sub_var_where_39.run(buf144, buf8, buf141, buf142, arg84_1, arg76_1, arg77_1, buf1, buf140, arg35_1, buf146, 81920, grid=grid(81920), stream=stream0)
        buf147 = reinterpret_tensor(buf3, (1, 1, 2, 32, 64), (4096, 4096, 1, 128, 2), 0)
        del buf3
        buf148 = buf144
        del buf144
        buf149 = reinterpret_tensor(buf142, (1, 2, 32, 64), (4096, 1, 128, 2), 0)
        del buf142
        triton_per_fused_add_mean_mul_pow_sub_40.run(buf147, buf1, buf8, buf140, arg35_1, buf146, buf148, buf149, 4096, 20, grid=grid(4096), stream=stream0)
        buf150 = buf15
        del buf15
        triton_poi_fused_add_cat_clamp_min_div_expm1_gt_mean_mul_pow_sub_sum_where_zeros_like_41.run(buf149, buf140, arg35_1, buf150, 4096, grid=grid(4096), stream=stream0)
        buf151 = buf149
        del buf149
        triton_poi_fused_add_cat_clamp_min_expm1_log1p_mul_sum_42.run(buf140, arg35_1, buf150, buf151, 4096, grid=grid(4096), stream=stream0)
        buf156 = empty_strided((1, 3, 20, 2, 32, 64), (245760, 40, 2, 1, 7680, 120), device='cuda', dtype=torch.float32)
        buf193 = empty_strided((1, 3, 20, 2, 32, 64), (245760, 40, 2, 1, 7680, 120), device='cuda', dtype=torch.float32)
        buf196 = empty_strided((1, 3, 20, 2, 32, 64), (245760, 40, 2, 1, 7680, 120), device='cuda', dtype=torch.float32)
        buf198 = empty_strided((1, 3, 20, 2, 32, 64), (245760, 40, 2, 1, 7680, 120), device='cuda', dtype=torch.float32)
        buf158 = empty_strided((1, 3, 20, 2, 32, 64), (245760, 40, 2, 1, 7680, 120), device='cuda', dtype=torch.float32)
        buf160 = empty_strided((1, 3, 20, 2, 32, 64), (245760, 40, 2, 1, 7680, 120), device='cuda', dtype=torch.float32)
        buf162 = empty_strided((1, 3, 20, 2, 32, 64), (245760, 40, 2, 1, 7680, 120), device='cuda', dtype=torch.float32)
        buf164 = empty_strided((1, 3, 20, 2, 32, 64), (245760, 40, 2, 1, 7680, 120), device='cuda', dtype=torch.float32)
        buf166 = empty_strided((1, 3, 20, 2, 32, 64), (245760, 40, 2, 1, 7680, 120), device='cuda', dtype=torch.float32)
        buf169 = empty_strided((1, 3, 20, 2, 32, 64), (245760, 40, 2, 1, 7680, 120), device='cuda', dtype=torch.float32)
        buf171 = empty_strided((1, 3, 20, 2, 32, 64), (245760, 40, 2, 1, 7680, 120), device='cuda', dtype=torch.float32)
        buf173 = empty_strided((1, 3, 20, 2, 32, 64), (245760, 40, 2, 1, 7680, 120), device='cuda', dtype=torch.float32)
        buf175 = empty_strided((1, 3, 20, 2, 32, 64), (245760, 40, 2, 1, 7680, 120), device='cuda', dtype=torch.float32)
        buf178 = empty_strided((1, 3, 20, 2, 32, 64), (245760, 40, 2, 1, 7680, 120), device='cuda', dtype=torch.float32)
        buf180 = empty_strided((1, 3, 20, 2, 32, 64), (245760, 40, 2, 1, 7680, 120), device='cuda', dtype=torch.float32)
        buf182 = empty_strided((1, 3, 20, 2, 32, 64), (245760, 40, 2, 1, 7680, 120), device='cuda', dtype=torch.float32)
        buf184 = empty_strided((1, 3, 20, 2, 32, 64), (245760, 40, 2, 1, 7680, 120), device='cuda', dtype=torch.float32)
        buf187 = empty_strided((1, 3, 20, 2, 32, 64), (245760, 40, 2, 1, 7680, 120), device='cuda', dtype=torch.float32)
        buf189 = empty_strided((1, 3, 20, 2, 32, 64), (245760, 40, 2, 1, 7680, 120), device='cuda', dtype=torch.float32)
        buf191 = empty_strided((1, 3, 20, 2, 32, 64), (245760, 40, 2, 1, 7680, 120), device='cuda', dtype=torch.float32)
        triton_poi_fused_add_div_erf_exp_mul_sub_tanh_43.run(buf155, arg80_1, buf90, arg71_1, buf156, buf193, buf196, buf198, buf158, buf160, buf162, buf164, buf166, buf169, buf171, buf173, buf175, buf178, buf180, buf182, buf184, buf187, buf189, buf191, 245760, grid=grid(245760), stream=stream0)
        buf152 = buf141
        del buf141
        buf157 = empty_strided((1, 20, 2, 32, 64), (81920, 2, 1, 2560, 40), device='cuda', dtype=torch.float32)
        buf159 = empty_strided((1, 20, 2, 32, 64), (81920, 2, 1, 2560, 40), device='cuda', dtype=torch.float32)
        buf161 = empty_strided((1, 20, 2, 32, 64), (81920, 2, 1, 2560, 40), device='cuda', dtype=torch.float32)
        buf163 = empty_strided((1, 20, 2, 32, 64), (81920, 2, 1, 2560, 40), device='cuda', dtype=torch.float32)
        buf165 = empty_strided((1, 20, 2, 32, 64), (81920, 2, 1, 2560, 40), device='cuda', dtype=torch.float32)
        buf167 = empty_strided((1, 20, 2, 32, 64), (81920, 2, 1, 2560, 40), device='cuda', dtype=torch.float32)
        buf170 = empty_strided((1, 20, 2, 32, 64), (81920, 2, 1, 2560, 40), device='cuda', dtype=torch.float32)
        buf172 = empty_strided((1, 20, 2, 32, 64), (81920, 2, 1, 2560, 40), device='cuda', dtype=torch.float32)
        buf174 = empty_strided((1, 20, 2, 32, 64), (81920, 2, 1, 2560, 40), device='cuda', dtype=torch.float32)
        buf176 = empty_strided((1, 20, 2, 32, 64), (81920, 2, 1, 2560, 40), device='cuda', dtype=torch.float32)
        buf179 = empty_strided((1, 20, 2, 32, 64), (81920, 2, 1, 2560, 40), device='cuda', dtype=torch.float32)
        buf181 = empty_strided((1, 20, 2, 32, 64), (81920, 2, 1, 2560, 40), device='cuda', dtype=torch.float32)
        buf183 = empty_strided((1, 20, 2, 32, 64), (81920, 2, 1, 2560, 40), device='cuda', dtype=torch.float32)
        buf185 = empty_strided((1, 20, 2, 32, 64), (81920, 2, 1, 2560, 40), device='cuda', dtype=torch.float32)
        buf188 = empty_strided((1, 20, 2, 32, 64), (81920, 2, 1, 2560, 40), device='cuda', dtype=torch.float32)
        buf190 = empty_strided((1, 20, 2, 32, 64), (81920, 2, 1, 2560, 40), device='cuda', dtype=torch.float32)
        buf192 = empty_strided((1, 20, 2, 32, 64), (81920, 2, 1, 2560, 40), device='cuda', dtype=torch.float32)
        buf194 = empty_strided((1, 20, 2, 32, 64), (81920, 2, 1, 2560, 40), device='cuda', dtype=torch.float32)
        buf197 = empty_strided((1, 20, 2, 32, 64), (81920, 2, 1, 2560, 40), device='cuda', dtype=torch.float32)
        buf199 = empty_strided((1, 20, 2, 32, 64), (81920, 2, 1, 2560, 40), device='cuda', dtype=torch.float32)
        triton_poi_fused_add_div_exp_mean_mul_sub_sum_44.run(arg85_1, arg76_1, arg77_1, arg86_1, buf94, buf147, buf146, buf148, buf151, buf156, buf93, buf158, buf160, buf162, buf164, buf166, buf169, buf171, buf173, buf175, buf178, buf180, buf182, buf184, buf187, buf189, buf191, buf193, buf196, buf198, buf152, buf157, buf159, buf161, buf163, buf165, buf167, buf170, buf172, buf174, buf176, buf179, buf181, buf183, buf185, buf188, buf190, buf192, buf194, buf197, buf199, 81920, grid=grid(81920), stream=stream0)
        del buf156
        del buf158
        del buf160
        del buf162
        del buf164
        del buf166
        del buf169
        del buf171
        del buf173
        del buf175
        del buf178
        del buf180
        del buf182
        del buf184
        del buf187
        del buf189
        del buf191
        buf153 = reinterpret_tensor(buf150, (1, 1, 2, 32, 64), (4096, 4096, 1, 128, 2), 0)
        del buf150
        triton_per_fused_mean_45.run(buf152, buf153, 4096, 20, grid=grid(4096), stream=stream0)
        buf271 = empty((3, 20, 32, 64, 3), device='cuda', dtype=torch.float32)
        buf154 = reinterpret_tensor(buf271, (1, 20, 32, 64, 3), (122880, 6144, 192, 3, 1), 0)
        triton_poi_fused_cat_46.run(buf147, buf146, buf148, buf151, arg82_1, buf153, arg77_1, arg76_1, arg84_1, buf154, 122880, grid=grid(122880), stream=stream0)
        del arg82_1
        del arg84_1
        del buf147
        del buf148
        del buf151
        del buf153
        buf204 = buf146
        del buf146
        buf224 = buf152
        del buf152
        triton_poi_fused_add_clone_full_like_gt_maximum_mul_sub_where_zeros_like_47.run(buf161, buf159, arg85_1, arg76_1, arg77_1, arg86_1, buf204, buf224, 81920, grid=grid(81920), stream=stream0)
        buf205 = buf94
        del buf94
        buf225 = empty_strided((1, 20, 2, 32, 64), (81920, 2, 1, 2560, 40), device='cuda', dtype=torch.float32)
        buf226 = buf225
        del buf225
        buf206 = buf205
        del buf205
        buf227 = buf226
        del buf226
        buf207 = buf206
        del buf206
        buf208 = buf207
        del buf207
        buf209 = buf208
        del buf208
        buf228 = buf227
        del buf227
        buf229 = buf228
        del buf228
        buf230 = buf229
        del buf229
        buf210 = buf209
        del buf209
        buf231 = buf230
        del buf230
        buf211 = buf210
        del buf210
        buf212 = buf211
        del buf211
        buf213 = buf212
        del buf212
        buf232 = buf231
        del buf231
        buf233 = buf232
        del buf232
        buf234 = buf233
        del buf233
        buf214 = buf213
        del buf213
        buf235 = buf234
        del buf234
        buf215 = buf214
        del buf214
        buf216 = buf215
        del buf215
        buf217 = buf216
        del buf216
        buf236 = buf235
        del buf235
        buf237 = buf236
        del buf236
        buf238 = buf237
        del buf237
        buf218 = buf217
        del buf217
        buf219 = buf218
        del buf218
        buf220 = buf219
        del buf219
        buf239 = buf238
        del buf238
        buf240 = buf239
        del buf239
        triton_poi_fused_add_full_like_gt_maximum_mul_sub_where_zeros_like_48.run(buf240, buf220, buf159, buf161, buf163, buf165, buf167, arg85_1, arg76_1, arg77_1, arg86_1, buf204, buf224, arg80_1, buf90, arg71_1, buf170, buf172, buf174, buf176, buf179, buf181, buf183, buf185, buf188, buf190, buf192, buf194, buf197, buf155, buf199, buf157, 81920, grid=grid(81920), stream=stream0)
        del buf155
        del buf157
        del buf159
        del buf161
        del buf163
        del buf165
        del buf167
        del buf170
        del buf172
        del buf174
        del buf176
        del buf179
        del buf181
        buf221 = reinterpret_tensor(buf224, (1, 20, 2, 32, 64), (81920, 4096, 2048, 64, 1), 0)
        del buf224
        buf241 = reinterpret_tensor(buf204, (1, 20, 2, 32, 64), (81920, 4096, 2048, 64, 1), 0)
        del buf204
        triton_poi_fused_add_clamp_min_div_mul_sub_49.run(buf220, arg85_1, arg76_1, arg77_1, arg86_1, buf240, buf221, buf241, 40960, 2, grid=grid(40960, 2), stream=stream0)
        buf222 = buf198
        del buf198
        buf242 = buf196
        del buf196
        triton_poi_fused_add_div_exp_mul_sub_tanh_50.run(buf221, arg80_1, buf90, arg71_1, buf241, buf222, buf242, 245760, grid=grid(245760), stream=stream0)
        buf243 = buf240
        del buf240
        buf223 = buf220
        del buf220
        buf244 = buf199
        del buf199
        buf245 = buf197
        del buf197
        buf246 = buf194
        del buf194
        triton_poi_fused_div_erf_exp_mul_neg_pow_sub_sum_tanh_51.run(buf242, buf222, buf93, buf90, arg71_1, buf243, buf223, buf244, buf245, buf246, 81920, grid=grid(81920), stream=stream0)
        buf249 = reinterpret_tensor(buf192, (1, 20, 2, 32, 64), (81920, 4096, 1, 128, 2), 0)
        del buf192
        buf250 = reinterpret_tensor(buf190, (1, 20, 2, 32, 64), (81920, 4096, 1, 128, 2), 0)
        del buf190
        buf262 = reinterpret_tensor(buf188, (1, 20, 2, 32, 64), (81920, 4096, 1, 128, 2), 0)
        del buf188
        triton_poi_fused_clamp_min_div_exp_mul_neg_sub_52.run(arg86_1, arg76_1, arg77_1, arg85_1, buf244, buf243, buf246, buf241, buf221, buf249, buf250, buf262, 40960, 2, grid=grid(40960, 2), stream=stream0)
        buf251 = buf185
        del buf185
        buf252 = buf183
        del buf183
        triton_poi_fused_abs_add_bitwise_and_clamp_clamp_min_div_gt_lt_mul_neg_pow_sub_where_53.run(buf223, buf243, buf244, buf245, buf246, buf249, buf250, buf251, buf252, 81920, grid=grid(81920), stream=stream0)
        buf253 = empty((1, 20, 2, 32, 64), device='cuda', dtype=torch.bool)
        buf256 = empty((1, 20, 2, 32, 64), device='cuda', dtype=torch.bool)
        buf259 = empty((1, 20, 2, 32, 64), device='cuda', dtype=torch.bool)
        triton_poi_fused_abs_add_bitwise_and_clamp_eq_ge_mul_ne_sub_54.run(buf241, buf251, buf221, buf252, buf253, buf256, buf259, 40, 2048, grid=grid(40, 2048), stream=stream0)
        buf254 = buf242
        del buf242
        buf257 = buf222
        del buf222
        buf260 = buf193
        del buf193
        triton_poi_fused_add_div_erf_exp_mul_sub_tanh_55.run(buf241, buf251, arg80_1, buf90, arg71_1, buf221, buf252, buf254, buf257, buf260, 245760, grid=grid(245760), stream=stream0)
        del arg71_1
        del arg80_1
        del buf90
        buf255 = reinterpret_tensor(buf250, (1, 20, 2, 32, 64), (81920, 2, 1, 2560, 40), 0)
        del buf250
        buf258 = reinterpret_tensor(buf249, (1, 20, 2, 32, 64), (81920, 2, 1, 2560, 40), 0)
        del buf249
        buf261 = buf246
        del buf246
        triton_poi_fused_add_clamp_clamp_min_div_exp_mul_neg_sub_sum_56.run(buf254, buf93, arg86_1, arg76_1, arg77_1, arg85_1, buf241, buf251, buf221, buf252, buf257, buf260, buf255, buf258, buf261, 2048, 40, grid=grid(2048, 40), stream=stream0)
        del buf254
        del buf257
        del buf260
        del buf93
        buf265 = reinterpret_tensor(buf245, (1, 20, 2, 32, 64), (81920, 4096, 1, 128, 2), 0)
        del buf245
        buf267 = reinterpret_tensor(buf244, (1, 20, 2, 32, 64), (81920, 4096, 1, 128, 2), 0)
        del buf244
        buf266 = reinterpret_tensor(buf243, (1, 20, 2, 32, 64), (81920, 4096, 1, 128, 2), 0)
        del buf243
        buf268 = reinterpret_tensor(buf223, (1, 20, 2, 32, 64), (81920, 4096, 1, 128, 2), 0)
        del buf223
        triton_poi_fused_add_bitwise_and_clamp_clamp_min_div_gt_mul_sub_where_57.run(buf256, buf258, buf259, buf261, buf262, buf253, buf255, arg86_1, arg76_1, arg77_1, arg85_1, buf221, buf252, buf241, buf251, buf140, arg35_1, buf265, buf267, buf266, buf268, 40960, 2, grid=grid(40960, 2), stream=stream0)
        del arg35_1
        del buf140
        del buf221
        del buf241
        del buf251
        del buf252
        del buf253
        del buf255
        del buf256
        del buf258
        del buf259
        del buf261
        del buf262
        buf269 = reinterpret_tensor(buf271, (1, 20, 32, 64, 3), (122880, 6144, 192, 3, 1), 122880)
        buf270 = reinterpret_tensor(buf271, (1, 20, 32, 64, 3), (122880, 6144, 192, 3, 1), 245760)
        triton_poi_fused_cat_58.run(buf266, arg85_1, arg76_1, arg77_1, arg86_1, buf265, buf268, buf267, buf269, buf270, 122880, grid=grid(122880), stream=stream0)
        del arg76_1
        del arg77_1
        del arg85_1
        del arg86_1
        del buf265
        del buf266
        del buf267
        del buf268
        return (reinterpret_tensor(buf271, (3, 1, 20, 32, 64, 3), (122880, 122880, 6144, 192, 3, 1), 0), buf1, buf8, buf10, buf11, buf13, buf17, buf44)
