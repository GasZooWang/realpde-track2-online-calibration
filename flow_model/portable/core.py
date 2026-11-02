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
triton_poi_fused_clone_0 = kernels['triton_poi_fused_clone_0']
import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream
triton_poi_fused_cat_1 = kernels['triton_poi_fused_cat_1']
triton_poi_fused_add_clone_mul_2 = kernels['triton_poi_fused_add_clone_mul_2']
triton_red_fused_mean_3 = kernels['triton_red_fused_mean_3']
triton_per_fused__to_copy_mean_4 = kernels['triton_per_fused__to_copy_mean_4']
triton_poi_fused__to_copy_silu_5 = kernels['triton_poi_fused__to_copy_silu_5']
triton_poi_fused_sub_6 = kernels['triton_poi_fused_sub_6']
triton_poi_fused_add_mul_7 = kernels['triton_poi_fused_add_mul_7']
triton_poi_fused_cat_8 = kernels['triton_poi_fused_cat_8']
triton_per_fused_add_clamp_detach_div_log_mean_mul_pow_sub_var_9 = kernels['triton_per_fused_add_clamp_detach_div_log_mean_mul_pow_sub_var_9']
triton_red_fused_mean_mul_var_10 = kernels['triton_red_fused_mean_mul_var_10']
triton_per_fused_mean_mul_var_11 = kernels['triton_per_fused_mean_mul_var_11']
triton_per_fused_mul_var_12 = kernels['triton_per_fused_mul_var_12']
triton_per_fused_detach_mean_mul_var_13 = kernels['triton_per_fused_detach_mean_mul_var_13']
triton_poi_fused_add_clamp_exp_log_mean_mul_ne_sub_where_14 = kernels['triton_poi_fused_add_clamp_exp_log_mean_mul_ne_sub_where_14']
triton_per_fused__to_copy_div_exp_le_mean_mul_15 = kernels['triton_per_fused__to_copy_div_exp_le_mean_mul_15']
triton_poi_fused__to_copy_div_exp_le_mean_mul_16 = kernels['triton_poi_fused__to_copy_div_exp_le_mean_mul_16']
triton_poi_fused__to_copy_add_clamp_div_exp_le_mean_mul_sub_17 = kernels['triton_poi_fused__to_copy_add_clamp_div_exp_le_mean_mul_sub_17']
triton_poi_fused__softmax_18 = kernels['triton_poi_fused__softmax_18']
triton_poi_fused_clone_19 = kernels['triton_poi_fused_clone_19']
triton_per_fused_mean_mul_sub_20 = kernels['triton_per_fused_mean_mul_sub_20']
triton_per_fused_add_constant_pad_nd_div_mean_mul_sqrt_sub_21 = kernels['triton_per_fused_add_constant_pad_nd_div_mean_mul_sqrt_sub_21']
triton_red_fused_mean_pow_22 = kernels['triton_red_fused_mean_pow_22']
triton_per_fused_mean_pow_23 = kernels['triton_per_fused_mean_pow_23']
triton_per_fused_mean_pow_sqrt_24 = kernels['triton_per_fused_mean_pow_sqrt_24']
triton_poi_fused_add_clamp_min_div_gt_mean_mul_ones_like_pow_sqrt_where_25 = kernels['triton_poi_fused_add_clamp_min_div_gt_mean_mul_ones_like_pow_sqrt_where_25']
triton_poi_fused_clone_26 = kernels['triton_poi_fused_clone_26']
triton_poi_fused_minimum_mul_27 = kernels['triton_poi_fused_minimum_mul_27']
triton_poi_fused_mul_sum_28 = kernels['triton_poi_fused_mul_sum_28']
triton_poi_fused_add_mul_29 = kernels['triton_poi_fused_add_mul_29']
triton_poi_fused_add_clamp_constant_pad_nd_copy_div_mean_mul_slice_scatter_sub_zeros_30 = kernels['triton_poi_fused_add_clamp_constant_pad_nd_copy_div_mean_mul_slice_scatter_sub_zeros_30']
triton_per_fused_mean_31 = kernels['triton_per_fused_mean_31']
triton_per_fused__to_copy_eq_mean_mul_pow_std_sub_32 = kernels['triton_per_fused__to_copy_eq_mean_mul_pow_std_sub_32']
triton_per_fused_detach_mean_mul_pow_sub_33 = kernels['triton_per_fused_detach_mean_mul_pow_sub_33']
triton_poi_fused_mean_mul_sub_34 = kernels['triton_poi_fused_mean_mul_sub_34']
triton_per_fused_add_clamp_detach_div_log_mean_mul_pow_35 = kernels['triton_per_fused_add_clamp_detach_div_log_mean_mul_pow_35']
triton_poi_fused_add_clamp_copy_exp_mean_mul_slice_scatter_36 = kernels['triton_poi_fused_add_clamp_copy_exp_mean_mul_slice_scatter_36']
triton_poi_fused_add_mul_37 = kernels['triton_poi_fused_add_mul_37']
triton_poi_fused_replication_pad2d_38 = kernels['triton_poi_fused_replication_pad2d_38']
triton_poi_fused_avg_pool2d_39 = kernels['triton_poi_fused_avg_pool2d_39']
triton_poi_fused_abs_add_mul_ne_stack_sub_where_40 = kernels['triton_poi_fused_abs_add_mul_ne_stack_sub_where_40']
triton_per_fused_abs_add_any_bitwise_or_clamp_min_div_mul_ne_sum_where_41 = kernels['triton_per_fused_abs_add_any_bitwise_or_clamp_min_div_mul_ne_sum_where_41']
triton_per_fused_std_42 = kernels['triton_per_fused_std_42']
triton_poi_fused_add_clamp_min_div_mul_stack_sub_43 = kernels['triton_poi_fused_add_clamp_min_div_mul_stack_sub_43']
triton_poi_fused_stack_44 = kernels['triton_poi_fused_stack_44']
triton_poi_fused_stack_45 = kernels['triton_poi_fused_stack_45']
triton_poi_fused_stack_46 = kernels['triton_poi_fused_stack_46']
triton_poi_fused_stack_47 = kernels['triton_poi_fused_stack_47']
triton_poi_fused_stack_48 = kernels['triton_poi_fused_stack_48']
triton_poi_fused_stack_49 = kernels['triton_poi_fused_stack_49']
triton_poi_fused_replication_pad2d_50 = kernels['triton_poi_fused_replication_pad2d_50']
triton_poi_fused_replication_pad2d_51 = kernels['triton_poi_fused_replication_pad2d_51']
triton_poi_fused_avg_pool2d_52 = kernels['triton_poi_fused_avg_pool2d_52']
triton_poi_fused_add_clamp_min_div_mul_stack_sub_53 = kernels['triton_poi_fused_add_clamp_min_div_mul_stack_sub_53']
triton_per_fused__to_copy_eq_mean_std_54 = kernels['triton_per_fused__to_copy_eq_mean_std_54']
triton_poi_fused_stack_55 = kernels['triton_poi_fused_stack_55']
triton_poi_fused__to_copy_cat_56 = kernels['triton_poi_fused__to_copy_cat_56']
triton_poi_fused__to_copy_silu_57 = kernels['triton_poi_fused__to_copy_silu_57']
triton_poi_fused_add_exp_mul_ne_sub_tanh_where_58 = kernels['triton_poi_fused_add_exp_mul_ne_sub_tanh_where_58']
triton_per_fused_abs_add_any_bitwise_or_clamp_min_div_mul_ne_sum_where_59 = kernels['triton_per_fused_abs_add_any_bitwise_or_clamp_min_div_mul_ne_sum_where_59']
triton_poi_fused_cat_60 = kernels['triton_poi_fused_cat_60']
triton_poi_fused_cat_61 = kernels['triton_poi_fused_cat_61']
triton_poi_fused_cat_62 = kernels['triton_poi_fused_cat_62']
triton_poi_fused_cat_63 = kernels['triton_poi_fused_cat_63']
triton_poi_fused_cat_64 = kernels['triton_poi_fused_cat_64']
triton_poi_fused_cat_65 = kernels['triton_poi_fused_cat_65']
triton_poi_fused_cat_66 = kernels['triton_poi_fused_cat_66']
triton_poi_fused_full_like_67 = kernels['triton_poi_fused_full_like_67']
triton_per_fused_mean_std_68 = kernels['triton_per_fused_mean_std_68']
triton_poi_fused_cat_69 = kernels['triton_poi_fused_cat_69']
triton_poi_fused__to_copy_cat_70 = kernels['triton_poi_fused__to_copy_cat_70']
triton_poi_fused__to_copy_add_cat_convolution_mul_tanh_71 = kernels['triton_poi_fused__to_copy_add_cat_convolution_mul_tanh_71']
triton_poi_fused_replication_pad2d_72 = kernels['triton_poi_fused_replication_pad2d_72']
triton_poi_fused_replication_pad2d_73 = kernels['triton_poi_fused_replication_pad2d_73']
triton_poi_fused__to_copy_convolution_replication_pad2d_silu_74 = kernels['triton_poi_fused__to_copy_convolution_replication_pad2d_silu_74']
triton_poi_fused_replication_pad2d_75 = kernels['triton_poi_fused_replication_pad2d_75']
triton_poi_fused_replication_pad2d_76 = kernels['triton_poi_fused_replication_pad2d_76']
triton_poi_fused_replication_pad2d_77 = kernels['triton_poi_fused_replication_pad2d_77']
triton_poi_fused__to_copy_add_convolution_replication_pad2d_silu_78 = kernels['triton_poi_fused__to_copy_add_convolution_replication_pad2d_silu_78']
triton_poi_fused__to_copy_avg_pool2d_silu_79 = kernels['triton_poi_fused__to_copy_avg_pool2d_silu_79']
triton_poi_fused_replication_pad2d_80 = kernels['triton_poi_fused_replication_pad2d_80']
triton_poi_fused_replication_pad2d_81 = kernels['triton_poi_fused_replication_pad2d_81']
triton_poi_fused_replication_pad2d_82 = kernels['triton_poi_fused_replication_pad2d_82']
triton_poi_fused__to_copy_convolution_replication_pad2d_silu_83 = kernels['triton_poi_fused__to_copy_convolution_replication_pad2d_silu_83']
triton_poi_fused__to_copy_add_avg_pool2d_convolution_replication_pad2d_silu_84 = kernels['triton_poi_fused__to_copy_add_avg_pool2d_convolution_replication_pad2d_silu_84']
triton_poi_fused__to_copy_avg_pool2d_85 = kernels['triton_poi_fused__to_copy_avg_pool2d_85']
triton_poi_fused_replication_pad2d_86 = kernels['triton_poi_fused_replication_pad2d_86']
triton_poi_fused_replication_pad2d_87 = kernels['triton_poi_fused_replication_pad2d_87']
triton_poi_fused_replication_pad2d_88 = kernels['triton_poi_fused_replication_pad2d_88']
triton_poi_fused__to_copy_convolution_replication_pad2d_silu_89 = kernels['triton_poi_fused__to_copy_convolution_replication_pad2d_silu_89']
triton_poi_fused__to_copy_add_avg_pool2d_convolution_replication_pad2d_silu_90 = kernels['triton_poi_fused__to_copy_add_avg_pool2d_convolution_replication_pad2d_silu_90']
triton_poi_fused__to_copy_avg_pool2d_91 = kernels['triton_poi_fused__to_copy_avg_pool2d_91']
triton_poi_fused_replication_pad2d_92 = kernels['triton_poi_fused_replication_pad2d_92']
triton_poi_fused_replication_pad2d_93 = kernels['triton_poi_fused_replication_pad2d_93']
triton_poi_fused_replication_pad2d_94 = kernels['triton_poi_fused_replication_pad2d_94']
triton_poi_fused__to_copy_convolution_replication_pad2d_silu_95 = kernels['triton_poi_fused__to_copy_convolution_replication_pad2d_silu_95']
triton_poi_fused__to_copy_add_avg_pool2d_convolution_replication_pad2d_silu_96 = kernels['triton_poi_fused__to_copy_add_avg_pool2d_convolution_replication_pad2d_silu_96']
triton_poi_fused__to_copy__unsafe_index_add_arange_avg_pool2d_clamp_convolution_mul_replication_pad2d_rsub_silu_sub_97 = kernels['triton_poi_fused__to_copy__unsafe_index_add_arange_avg_pool2d_clamp_convolution_mul_replication_pad2d_rsub_silu_sub_97']
triton_poi_fused_replication_pad2d_98 = kernels['triton_poi_fused_replication_pad2d_98']
triton_poi_fused__to_copy_add_convolution_replication_pad2d_silu_99 = kernels['triton_poi_fused__to_copy_add_convolution_replication_pad2d_silu_99']
triton_poi_fused__to_copy__unsafe_index_add_arange_clamp_convolution_mul_replication_pad2d_rsub_silu_sub_100 = kernels['triton_poi_fused__to_copy__unsafe_index_add_arange_clamp_convolution_mul_replication_pad2d_rsub_silu_sub_100']
triton_poi_fused_replication_pad2d_101 = kernels['triton_poi_fused_replication_pad2d_101']
triton_poi_fused__to_copy_add_convolution_replication_pad2d_silu_102 = kernels['triton_poi_fused__to_copy_add_convolution_replication_pad2d_silu_102']
triton_poi_fused__to_copy__unsafe_index_add_arange_clamp_convolution_mul_replication_pad2d_rsub_silu_sub_103 = kernels['triton_poi_fused__to_copy__unsafe_index_add_arange_clamp_convolution_mul_replication_pad2d_rsub_silu_sub_103']
triton_per_fused_mean_var_104 = kernels['triton_per_fused_mean_var_104']
triton_per_fused__to_copy_eq_mean_stack_var_105 = kernels['triton_per_fused__to_copy_eq_mean_stack_var_105']
triton_per_fused_var_106 = kernels['triton_per_fused_var_106']
triton_per_fused_mean_pow_sub_107 = kernels['triton_per_fused_mean_pow_sub_107']
triton_poi_fused_replication_pad2d_108 = kernels['triton_poi_fused_replication_pad2d_108']
triton_poi_fused_avg_pool2d_109 = kernels['triton_poi_fused_avg_pool2d_109']
triton_per_fused_add_clamp_div_log_mean_mul_stack_sub_var_110 = kernels['triton_per_fused_add_clamp_div_log_mean_mul_stack_sub_var_110']
triton_poi_fused_stack_111 = kernels['triton_poi_fused_stack_111']
triton_poi_fused_stack_112 = kernels['triton_poi_fused_stack_112']
triton_poi_fused_stack_113 = kernels['triton_poi_fused_stack_113']
triton_poi_fused_stack_114 = kernels['triton_poi_fused_stack_114']
triton_poi_fused_stack_115 = kernels['triton_poi_fused_stack_115']
triton_poi_fused_stack_116 = kernels['triton_poi_fused_stack_116']
triton_poi_fused__to_copy_clamp_117 = kernels['triton_poi_fused__to_copy_clamp_117']
triton_poi_fused__to_copy_silu_118 = kernels['triton_poi_fused__to_copy_silu_118']
triton_poi_fused_add_mul_sub_119 = kernels['triton_poi_fused_add_mul_sub_119']
triton_poi_fused_copy_div_select_scatter_slice_scatter_sub_120 = kernels['triton_poi_fused_copy_div_select_scatter_slice_scatter_sub_120']
triton_poi_fused_add_div_exp_mean_mul_sub_tanh_121 = kernels['triton_poi_fused_add_div_exp_mean_mul_sub_tanh_121']
triton_per_fused_add_mean_mul_var_122 = kernels['triton_per_fused_add_mean_mul_var_122']
triton_per_fused_mean_123 = kernels['triton_per_fused_mean_123']
triton_poi_fused_add_div_mean_mul_sub_124 = kernels['triton_poi_fused_add_div_mean_mul_sub_124']
triton_per_fused_mean_std_125 = kernels['triton_per_fused_mean_std_125']
triton_per_fused_mean_mul_std_var_126 = kernels['triton_per_fused_mean_mul_std_var_126']
triton_poi_fused__to_copy_cat_127 = kernels['triton_poi_fused__to_copy_cat_127']
triton_poi_fused__to_copy_add_convolution_mul_tanh_128 = kernels['triton_poi_fused__to_copy_add_convolution_mul_tanh_128']
triton_poi_fused_replication_pad2d_129 = kernels['triton_poi_fused_replication_pad2d_129']
triton_poi_fused_replication_pad2d_130 = kernels['triton_poi_fused_replication_pad2d_130']
triton_poi_fused__to_copy_convolution_replication_pad2d_silu_131 = kernels['triton_poi_fused__to_copy_convolution_replication_pad2d_silu_131']
triton_poi_fused_replication_pad2d_132 = kernels['triton_poi_fused_replication_pad2d_132']
triton_poi_fused_replication_pad2d_133 = kernels['triton_poi_fused_replication_pad2d_133']
triton_poi_fused_replication_pad2d_134 = kernels['triton_poi_fused_replication_pad2d_134']
triton_poi_fused_avg_pool2d_silu_135 = kernels['triton_poi_fused_avg_pool2d_silu_135']
triton_poi_fused_replication_pad2d_136 = kernels['triton_poi_fused_replication_pad2d_136']
triton_poi_fused_replication_pad2d_137 = kernels['triton_poi_fused_replication_pad2d_137']
triton_poi_fused__to_copy_convolution_replication_pad2d_silu_138 = kernels['triton_poi_fused__to_copy_convolution_replication_pad2d_silu_138']
triton_poi_fused_replication_pad2d_139 = kernels['triton_poi_fused_replication_pad2d_139']
triton_poi_fused_replication_pad2d_140 = kernels['triton_poi_fused_replication_pad2d_140']
triton_poi_fused_replication_pad2d_141 = kernels['triton_poi_fused_replication_pad2d_141']
triton_poi_fused__to_copy_add_convolution_replication_pad2d_silu_142 = kernels['triton_poi_fused__to_copy_add_convolution_replication_pad2d_silu_142']
triton_poi_fused__to_copy__unsafe_index_add_arange_clamp_mul_rsub_sub_143 = kernels['triton_poi_fused__to_copy__unsafe_index_add_arange_clamp_mul_rsub_sub_143']
triton_poi_fused__to_copy_add_convolution_replication_pad2d_silu_144 = kernels['triton_poi_fused__to_copy_add_convolution_replication_pad2d_silu_144']
triton_poi_fused__to_copy_silu_145 = kernels['triton_poi_fused__to_copy_silu_145']
triton_poi_fused_stack_146 = kernels['triton_poi_fused_stack_146']
triton_poi_fused_add_147 = kernels['triton_poi_fused_add_147']
triton_poi_fused_add_div_mul_sub_148 = kernels['triton_poi_fused_add_div_mul_sub_148']
triton_per_fused_mean_sub_var_149 = kernels['triton_per_fused_mean_sub_var_149']
triton_poi_fused_add_bitwise_and_clamp_min_div_gt_mean_mul_sqrt_sub_var_where_150 = kernels['triton_poi_fused_add_bitwise_and_clamp_min_div_gt_mean_mul_sqrt_sub_var_where_150']
triton_per_fused_add_mean_mul_151 = kernels['triton_per_fused_add_mean_mul_151']
triton_poi_fused_add_exp_mean_mul_sub_152 = kernels['triton_poi_fused_add_exp_mean_mul_sub_152']
triton_poi_fused_copy_div_select_scatter_slice_scatter_sub_153 = kernels['triton_poi_fused_copy_div_select_scatter_slice_scatter_sub_153']
triton_per_fused_mean_var_154 = kernels['triton_per_fused_mean_var_154']
triton_poi_fused_clone_155 = kernels['triton_poi_fused_clone_155']
triton_poi_fused_add_cat_mul_stack_sub_156 = kernels['triton_poi_fused_add_cat_mul_stack_sub_156']
triton_poi_fused_cat_157 = kernels['triton_poi_fused_cat_157']
triton_per_fused_mean_std_158 = kernels['triton_per_fused_mean_std_158']
triton_poi_fused_cat_159 = kernels['triton_poi_fused_cat_159']
triton_poi_fused_clamp_min_div_mean_sqrt_var_160 = kernels['triton_poi_fused_clamp_min_div_mean_sqrt_var_160']
triton_poi_fused__to_copy_cat_161 = kernels['triton_poi_fused__to_copy_cat_161']
triton_poi_fused__to_copy_162 = kernels['triton_poi_fused__to_copy_162']
triton_poi_fused__to_copy_cat_163 = kernels['triton_poi_fused__to_copy_cat_163']
triton_poi_fused__to_copy_add_cat_convolution_mul_silu_tanh_164 = kernels['triton_poi_fused__to_copy_add_cat_convolution_mul_silu_tanh_164']
triton_poi_fused__to_copy_convolution_silu_165 = kernels['triton_poi_fused__to_copy_convolution_silu_165']
triton_poi_fused__to_copy_convolution_silu_166 = kernels['triton_poi_fused__to_copy_convolution_silu_166']
triton_poi_fused__to_copy_add_convolution_mul_silu_167 = kernels['triton_poi_fused__to_copy_add_convolution_mul_silu_167']
triton_poi_fused__to_copy_avg_pool2d_168 = kernels['triton_poi_fused__to_copy_avg_pool2d_168']
triton_poi_fused__to_copy_avg_pool2d_convolution_silu_169 = kernels['triton_poi_fused__to_copy_avg_pool2d_convolution_silu_169']
triton_poi_fused__to_copy_avg_pool2d_convolution_silu_170 = kernels['triton_poi_fused__to_copy_avg_pool2d_convolution_silu_170']
triton_poi_fused__to_copy_add_avg_pool2d_convolution_mul_silu_171 = kernels['triton_poi_fused__to_copy_add_avg_pool2d_convolution_mul_silu_171']
triton_poi_fused__to_copy_avg_pool2d_convolution_silu_172 = kernels['triton_poi_fused__to_copy_avg_pool2d_convolution_silu_172']
triton_poi_fused__to_copy_avg_pool2d_convolution_silu_173 = kernels['triton_poi_fused__to_copy_avg_pool2d_convolution_silu_173']
triton_poi_fused__to_copy_add_avg_pool2d_convolution_mul_silu_174 = kernels['triton_poi_fused__to_copy_add_avg_pool2d_convolution_mul_silu_174']
triton_poi_fused__to_copy_add_avg_pool2d_convolution_mul_silu_175 = kernels['triton_poi_fused__to_copy_add_avg_pool2d_convolution_mul_silu_175']
triton_poi_fused__to_copy_add_176 = kernels['triton_poi_fused__to_copy_add_176']
triton_poi_fused__to_copy_add_convolution_mul_silu_177 = kernels['triton_poi_fused__to_copy_add_convolution_mul_silu_177']
triton_poi_fused__to_copy__unsafe_index_add_arange_clamp_convolution_mul_rsub_silu_sub_178 = kernels['triton_poi_fused__to_copy__unsafe_index_add_arange_clamp_convolution_mul_rsub_silu_sub_178']
triton_poi_fused__to_copy_add_179 = kernels['triton_poi_fused__to_copy_add_179']
triton_poi_fused__to_copy_add_convolution_mul_silu_180 = kernels['triton_poi_fused__to_copy_add_convolution_mul_silu_180']
triton_poi_fused_add_mul_sub_tanh_181 = kernels['triton_poi_fused_add_mul_sub_tanh_181']
triton_poi_fused_mean_mul_sub_where_182 = kernels['triton_poi_fused_mean_mul_sub_where_182']
triton_poi_fused_where_183 = kernels['triton_poi_fused_where_183']
triton_per_fused_clamp_min_div_mean_sqrt_var_184 = kernels['triton_per_fused_clamp_min_div_mean_sqrt_var_184']
triton_per_fused_mean_var_185 = kernels['triton_per_fused_mean_var_185']
triton_poi_fused_mean_sub_186 = kernels['triton_poi_fused_mean_sub_186']
triton_poi_fused_mul_pow_sum_var_187 = kernels['triton_poi_fused_mul_pow_sum_var_187']
triton_poi_fused_sum_var_188 = kernels['triton_poi_fused_sum_var_188']
triton_poi_fused_avg_pool2d_clamp_min_sqrt_var_189 = kernels['triton_poi_fused_avg_pool2d_clamp_min_sqrt_var_189']
triton_poi_fused_add_avg_pool2d_div_log_190 = kernels['triton_poi_fused_add_avg_pool2d_div_log_190']
triton_red_fused_cat_mean_191 = kernels['triton_red_fused_cat_mean_191']
triton_poi_fused_cat_192 = kernels['triton_poi_fused_cat_192']
triton_poi_fused_clamp_min_sqrt_193 = kernels['triton_poi_fused_clamp_min_sqrt_193']
triton_poi_fused_cat_194 = kernels['triton_poi_fused_cat_194']
triton_poi_fused__to_copy_clamp_195 = kernels['triton_poi_fused__to_copy_clamp_195']
triton_poi_fused__to_copy_clamp_convolution_silu_196 = kernels['triton_poi_fused__to_copy_clamp_convolution_silu_196']
triton_poi_fused_replication_pad2d_197 = kernels['triton_poi_fused_replication_pad2d_197']
triton_poi_fused_replication_pad2d_198 = kernels['triton_poi_fused_replication_pad2d_198']
triton_poi_fused_replication_pad2d_199 = kernels['triton_poi_fused_replication_pad2d_199']
triton_poi_fused_replication_pad2d_200 = kernels['triton_poi_fused_replication_pad2d_200']
triton_poi_fused_replication_pad2d_201 = kernels['triton_poi_fused_replication_pad2d_201']
triton_poi_fused_avg_pool2d_202 = kernels['triton_poi_fused_avg_pool2d_202']
triton_poi_fused_replication_pad2d_203 = kernels['triton_poi_fused_replication_pad2d_203']
triton_poi_fused_replication_pad2d_204 = kernels['triton_poi_fused_replication_pad2d_204']
triton_poi_fused__to_copy_convolution_replication_pad2d_silu_205 = kernels['triton_poi_fused__to_copy_convolution_replication_pad2d_silu_205']
triton_poi_fused_replication_pad2d_206 = kernels['triton_poi_fused_replication_pad2d_206']
triton_poi_fused_replication_pad2d_207 = kernels['triton_poi_fused_replication_pad2d_207']
triton_poi_fused_replication_pad2d_208 = kernels['triton_poi_fused_replication_pad2d_208']
triton_poi_fused__to_copy_add_convolution_replication_pad2d_silu_209 = kernels['triton_poi_fused__to_copy_add_convolution_replication_pad2d_silu_209']
triton_poi_fused__to_copy__unsafe_index_add_arange_clamp_mul_rsub_sub_210 = kernels['triton_poi_fused__to_copy__unsafe_index_add_arange_clamp_mul_rsub_sub_210']
triton_poi_fused__to_copy_add_convolution_replication_pad2d_silu_211 = kernels['triton_poi_fused__to_copy_add_convolution_replication_pad2d_silu_211']
triton_poi_fused__to_copy_add_avg_pool2d_clamp_min_convolution_div_mul_replication_pad2d_silu_sum_tanh_var_212 = kernels['triton_poi_fused__to_copy_add_avg_pool2d_clamp_min_convolution_div_mul_replication_pad2d_silu_sum_tanh_var_212']
triton_poi_fused_copy_div_select_scatter_slice_scatter_sub_213 = kernels['triton_poi_fused_copy_div_select_scatter_slice_scatter_sub_213']
triton_poi_fused_add_mul_sub_214 = kernels['triton_poi_fused_add_mul_sub_214']
triton_poi_fused_add_select_scatter_slice_scatter_215 = kernels['triton_poi_fused_add_select_scatter_slice_scatter_215']
triton_poi_fused_add_copy_mean_slice_scatter_216 = kernels['triton_poi_fused_add_copy_mean_slice_scatter_216']
triton_poi_fused_bitwise_or_ne_217 = kernels['triton_poi_fused_bitwise_or_ne_217']

def call(args):
    (arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1) = args
    args.clear()
    assert_size_stride(arg0_1, (32, 28), (28, 1))
    assert_size_stride(arg1_1, (32,), (1,))
    assert_size_stride(arg2_1, (32, 32), (32, 1))
    assert_size_stride(arg3_1, (32,), (1,))
    assert_size_stride(arg4_1, (2, 32), (32, 1))
    assert_size_stride(arg5_1, (2,), (1,))
    assert_size_stride(arg6_1, (64, 22), (22, 1))
    assert_size_stride(arg7_1, (64,), (1,))
    assert_size_stride(arg8_1, (64, 64), (64, 1))
    assert_size_stride(arg9_1, (64,), (1,))
    assert_size_stride(arg10_1, (2, 64), (64, 1))
    assert_size_stride(arg11_1, (2,), (1,))
    assert_size_stride(arg12_1, (96, 327, 1, 1), (327, 1, 1, 1))
    assert_size_stride(arg13_1, (96,), (1,))
    assert_size_stride(arg14_1, (32, 256), (256, 1))
    assert_size_stride(arg15_1, (32,), (1,))
    assert_size_stride(arg16_1, (192, 32), (32, 1))
    assert_size_stride(arg17_1, (192,), (1,))
    assert_size_stride(arg18_1, (96, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg19_1, (96,), (1,))
    assert_size_stride(arg20_1, (96, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(arg21_1, (96,), (1,))
    assert_size_stride(arg22_1, (96, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg23_1, (96,), (1,))
    assert_size_stride(arg24_1, (96, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(arg25_1, (96,), (1,))
    assert_size_stride(arg26_1, (80, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(arg27_1, (80,), (1,))
    assert_size_stride(arg28_1, (64, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(arg29_1, (64,), (1,))
    assert_size_stride(arg30_1, (64, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg31_1, (64,), (1,))
    assert_size_stride(arg32_1, (64, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg33_1, (64,), (1,))
    assert_size_stride(arg34_1, (96, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg35_1, (96,), (1,))
    assert_size_stride(arg36_1, (96, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg37_1, (96,), (1,))
    assert_size_stride(arg38_1, (96, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(arg39_1, (96,), (1,))
    assert_size_stride(arg40_1, (128, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(arg41_1, (128,), (1,))
    assert_size_stride(arg42_1, (128, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg43_1, (128,), (1,))
    assert_size_stride(arg44_1, (128, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg45_1, (128,), (1,))
    assert_size_stride(arg46_1, (96, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg47_1, (96,), (1,))
    assert_size_stride(arg48_1, (96, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg49_1, (96,), (1,))
    assert_size_stride(arg50_1, (96, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(arg51_1, (96,), (1,))
    assert_size_stride(arg52_1, (64, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(arg53_1, (64,), (1,))
    assert_size_stride(arg54_1, (64, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg55_1, (64,), (1,))
    assert_size_stride(arg56_1, (64, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg57_1, (64,), (1,))
    assert_size_stride(arg58_1, (80, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg59_1, (80,), (1,))
    assert_size_stride(arg60_1, (64, 333, 1, 1), (333, 1, 1, 1))
    assert_size_stride(arg61_1, (64,), (1,))
    assert_size_stride(arg62_1, (32, 256), (256, 1))
    assert_size_stride(arg63_1, (32,), (1,))
    assert_size_stride(arg64_1, (128, 32), (32, 1))
    assert_size_stride(arg65_1, (128,), (1,))
    assert_size_stride(arg66_1, (64, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg67_1, (64,), (1,))
    assert_size_stride(arg68_1, (64, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg69_1, (64,), (1,))
    assert_size_stride(arg70_1, (64, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg71_1, (64,), (1,))
    assert_size_stride(arg72_1, (64, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg73_1, (64,), (1,))
    assert_size_stride(arg74_1, (64, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg75_1, (64,), (1,))
    assert_size_stride(arg76_1, (64, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg77_1, (64,), (1,))
    assert_size_stride(arg78_1, (64, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg79_1, (64,), (1,))
    assert_size_stride(arg80_1, (64, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg81_1, (64,), (1,))
    assert_size_stride(arg82_1, (4, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg83_1, (4,), (1,))
    assert_size_stride(arg84_1, (32, 375, 1, 1), (375, 1, 1, 1))
    assert_size_stride(arg85_1, (32,), (1,))
    assert_size_stride(arg86_1, (32, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(arg87_1, (32,), (1,))
    assert_size_stride(arg88_1, (32, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(arg89_1, (32,), (1,))
    assert_size_stride(arg90_1, (64, 256), (256, 1))
    assert_size_stride(arg91_1, (64,), (1,))
    assert_size_stride(arg92_1, (64, 64), (64, 1))
    assert_size_stride(arg93_1, (64,), (1,))
    assert_size_stride(arg94_1, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(arg95_1, (32,), (1,))
    assert_size_stride(arg96_1, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(arg97_1, (32,), (1,))
    assert_size_stride(arg98_1, (64, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(arg99_1, (64,), (1,))
    assert_size_stride(arg100_1, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg101_1, (64,), (1,))
    assert_size_stride(arg102_1, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg103_1, (64,), (1,))
    assert_size_stride(arg104_1, (96, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg105_1, (96,), (1,))
    assert_size_stride(arg106_1, (96, 96, 3, 3), (864, 9, 3, 1))
    assert_size_stride(arg107_1, (96,), (1,))
    assert_size_stride(arg108_1, (96, 96, 3, 3), (864, 9, 3, 1))
    assert_size_stride(arg109_1, (96,), (1,))
    assert_size_stride(arg110_1, (96, 96, 3, 3), (864, 9, 3, 1))
    assert_size_stride(arg111_1, (96,), (1,))
    assert_size_stride(arg112_1, (96, 96, 3, 3), (864, 9, 3, 1))
    assert_size_stride(arg113_1, (96,), (1,))
    assert_size_stride(arg114_1, (64, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(arg115_1, (64,), (1,))
    assert_size_stride(arg116_1, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg117_1, (64,), (1,))
    assert_size_stride(arg118_1, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg119_1, (64,), (1,))
    assert_size_stride(arg120_1, (32, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg121_1, (32,), (1,))
    assert_size_stride(arg122_1, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(arg123_1, (32,), (1,))
    assert_size_stride(arg124_1, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(arg125_1, (32,), (1,))
    assert_size_stride(arg126_1, (3, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(arg127_1, (3,), (1,))
    assert_size_stride(arg128_1, (40, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(arg129_1, (40,), (1,))
    assert_size_stride(arg130_1, (80, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(arg131_1, (80,), (1,))
    assert_size_stride(arg132_1, (32, 78, 1, 1), (78, 1, 1, 1))
    assert_size_stride(arg133_1, (32,), (1,))
    assert_size_stride(arg134_1, (32, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg135_1, (32,), (1,))
    assert_size_stride(arg136_1, (32, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(arg137_1, (32,), (1,))
    assert_size_stride(arg138_1, (32, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg139_1, (32,), (1,))
    assert_size_stride(arg140_1, (32, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(arg141_1, (32,), (1,))
    assert_size_stride(arg142_1, (32, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg143_1, (32,), (1,))
    assert_size_stride(arg144_1, (32, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(arg145_1, (32,), (1,))
    assert_size_stride(arg146_1, (32, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg147_1, (32,), (1,))
    assert_size_stride(arg148_1, (32, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(arg149_1, (32,), (1,))
    assert_size_stride(arg150_1, (1, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(arg151_1, (1,), (1,))
    assert_size_stride(arg152_1, (1, 1, 1, 1, 3), (3, 3, 3, 3, 1))
    assert_size_stride(arg153_1, (1, 1, 1, 1, 3), (3, 3, 3, 3, 1))
    assert_size_stride(arg154_1, (1, 1, 1, 1, 3), (3, 3, 3, 3, 1))
    assert_size_stride(arg155_1, (2,), (1,))
    assert_size_stride(arg156_1, (1, 1, 1, 1, 2), (2, 2, 2, 2, 1))
    assert_size_stride(arg157_1, (1, 1, 1, 1, 3), (3, 3, 3, 3, 1))
    assert_size_stride(arg158_1, (1, 1, 1, 1, 3), (3, 3, 3, 3, 1))
    assert_size_stride(arg159_1, (1, 1, 1, 1, 2), (2, 2, 2, 2, 1))
    assert_size_stride(arg160_1, (1, 1, 1, 1, 2), (2, 2, 2, 2, 1))
    assert_size_stride(arg161_1, (1, 1, 1, 1, 2), (2, 2, 2, 2, 1))
    assert_size_stride(arg162_1, (1, 1, 1, 1, 2), (2, 2, 2, 2, 1))
    assert_size_stride(arg163_1, (1, 1, 1, 1, 2), (2, 2, 2, 2, 1))
    assert_size_stride(arg164_1, (1, 1, 1, 1, 2), (2, 2, 2, 2, 1))
    assert_size_stride(arg165_1, (1, 1, 1, 1, 2), (2, 2, 2, 2, 1))
    assert_size_stride(arg166_1, (1, 1, 1, 1, 2), (2, 2, 2, 2, 1))
    assert_size_stride(arg167_1, (1, 20, 32, 64, 2, 22), (1802240, 90112, 2816, 44, 22, 1))
    assert_size_stride(arg168_1, (1, 20, 32, 64, 2), (81920, 4096, 128, 2, 1))
    assert_size_stride(arg169_1, (1, 1, 32, 64, 2, 6), (24576, 24576, 768, 12, 6, 1))
    assert_size_stride(arg170_1, (1, 1, 1, 1, 3), (3, 3, 3, 3, 1))
    assert_size_stride(arg171_1, (1, 1, 1, 1, 3), (3, 3, 3, 3, 1))
    assert_size_stride(arg172_1, (2,), (1,))
    assert_size_stride(arg173_1, (1, 20, 32, 64, 3), (122880, 6144, 192, 3, 1))
    assert_size_stride(arg174_1, (1, 20, 32, 64, 3), (122880, 6144, 192, 3, 1))
    assert_size_stride(arg175_1, (1, 20, 32, 64, 3), (122880, 6144, 192, 3, 1))
    assert_size_stride(arg176_1, (1, 20, 32, 64, 3), (122880, 6144, 192, 3, 1))
    assert_size_stride(arg177_1, (1, 20, 32, 64, 3), (122880, 6144, 192, 3, 1))
    assert_size_stride(arg178_1, (1, 2), (2, 1))
    assert_size_stride(arg179_1, (1, 20, 32, 64, 3), (122880, 6144, 192, 3, 1))
    assert_size_stride(arg180_1, (1, 1, 32, 64, 3), (6144, 6144, 192, 3, 1))
    assert_size_stride(arg181_1, (1, 20, 1, 1, 2), (40, 2, 2, 2, 1))
    assert_size_stride(arg182_1, (1, 20, 32, 64, 2), (81920, 4096, 128, 2, 1))
    assert_size_stride(arg183_1, (1, 20, 32, 64, 2), (81920, 4096, 128, 2, 1))
    assert_size_stride(arg184_1, (1, 1, 32, 64, 2), (4096, 4096, 128, 2, 1))
    assert_size_stride(arg185_1, (7, 1, 20, 32, 64, 2), (81920, 81920, 4096, 128, 2, 1))
    assert_size_stride(arg186_1, (32, 64, 2), (128, 2, 1))
    assert_size_stride(arg187_1, (32, 64, 2), (128, 2, 1))
    assert_size_stride(arg188_1, (32, 64, 2), (128, 2, 1))
    assert_size_stride(arg189_1, (32, 64, 2), (128, 2, 1))
    assert_size_stride(arg190_1, (41, 41), (41, 1))
    assert_size_stride(arg191_1, (41, 40), (40, 1))
    assert_size_stride(arg192_1, (41, 40), (1, 41))
    assert_size_stride(arg193_1, (2048, 1), (1, 1))
    assert_size_stride(arg194_1, (41, 41), (41, 1))
    assert_size_stride(arg195_1, (7, 1, 32, 64, 2), (4096, 4096, 128, 2, 1))
    assert_size_stride(arg196_1, (7, 1, 1, 1, 1, 1), (1, 1, 1, 1, 1, 1))
    assert_size_stride(arg197_1, (1, 1, 1, 1, 1, 2), (2, 2, 2, 2, 2, 1))
    assert_size_stride(arg198_1, (20, 32, 64, 2), (4096, 128, 2, 1))
    assert_size_stride(arg199_1, (32, 64, 2), (128, 2, 1))
    assert_size_stride(arg200_1, (32, 64, 2), (128, 2, 1))
    assert_size_stride(arg201_1, (32, 64, 2), (128, 2, 1))
    assert_size_stride(arg202_1, (32, 64, 2), (128, 2, 1))
    assert_size_stride(arg203_1, (32, 64, 2), (128, 2, 1))
    assert_size_stride(arg204_1, (20, 32, 64, 2), (4096, 128, 2, 1))
    assert_size_stride(arg205_1, (20, 32, 64, 2), (4096, 128, 2, 1))
    assert_size_stride(arg206_1, (20, 32, 64, 2), (4096, 128, 2, 1))
    assert_size_stride(arg207_1, (20, 32, 64, 2), (4096, 128, 2, 1))
    assert_size_stride(arg208_1, (32, 64, 2), (128, 2, 1))
    assert_size_stride(arg209_1, (32, 64, 2), (128, 2, 1))
    assert_size_stride(arg210_1, (32, 64, 2), (128, 2, 1))
    assert_size_stride(arg211_1, (32, 64, 2), (128, 2, 1))
    assert_size_stride(arg212_1, (20, 32, 64, 2), (4096, 128, 2, 1))
    assert_size_stride(arg213_1, (20, 32, 64, 2), (4096, 128, 2, 1))
    assert_size_stride(arg214_1, (32, 64, 2), (128, 2, 1))
    assert_size_stride(arg215_1, (32, 64, 2), (128, 2, 1))
    assert_size_stride(arg216_1, (2, 32, 64), (2048, 64, 1))
    assert_size_stride(arg217_1, (2, 32, 64), (2048, 64, 1))
    assert_size_stride(arg218_1, (2, 2, 32, 64), (4096, 2048, 64, 1))
    assert_size_stride(arg219_1, (2, 2, 32, 64), (4096, 2048, 64, 1))
    assert_size_stride(arg220_1, (2, 2, 32, 64), (4096, 2048, 64, 1))
    assert_size_stride(arg221_1, (2, 2, 32, 64), (4096, 2048, 64, 1))
    assert_size_stride(arg222_1, (2, 2, 32, 64), (4096, 2048, 64, 1))
    assert_size_stride(arg223_1, (2, 2, 32, 64), (4096, 2048, 64, 1))
    assert_size_stride(arg224_1, (2, 2, 32, 64), (4096, 2048, 64, 1))
    assert_size_stride(arg225_1, (2, 2, 32, 64), (4096, 2048, 64, 1))
    assert_size_stride(arg226_1, (2, 2, 32, 64), (4096, 2048, 64, 1))
    assert_size_stride(arg227_1, (2, 2, 32, 64), (4096, 2048, 64, 1))
    assert_size_stride(arg228_1, (), ())
    with torch.cuda._DeviceGuard(DEVICE_INDEX):
        torch.cuda.set_device(DEVICE_INDEX)
        buf0 = empty((2048, 1), device='cuda', dtype=torch.float32)
        stream0 = get_cuda_stream(DEVICE_INDEX)
        triton_poi_fused_clone_0.run(arg193_1, buf0, 2048, grid=grid(2048), stream=stream0)
        del arg193_1
        buf1 = empty((2048, 41), device='cuda', dtype=torch.float32)
        triton_poi_fused_cat_1.run(buf0, arg176_1, buf1, 83968, grid=grid(83968), stream=stream0)
        buf2 = empty((41, 41), device='cuda', dtype=torch.float32)
        extern_kernels.mm(reinterpret_tensor(buf1, (41, 2048), (1, 41), 0), buf1, out=buf2)
        buf3 = buf2
        del buf2
        buf4 = empty((41, 41), device='cuda', dtype=torch.float32)
        buf5 = empty((41, 41), device='cuda', dtype=torch.float32)
        triton_poi_fused_add_clone_mul_2.run(buf3, arg190_1, arg194_1, buf4, buf5, 1681, grid=grid(1681), stream=stream0)
        del arg190_1
        del arg194_1
        buf6 = aten.linalg_cholesky_ex(buf5)
        del buf5
        buf7 = buf6[0]
        del buf6
        buf9 = empty_strided((4, 4, 8, 2, 3), (192, 16, 2, 1, 64), device='cuda', dtype=torch.float32)
        triton_red_fused_mean_3.run(arg173_1, arg170_1, arg157_1, arg171_1, arg155_1, buf9, 768, 107, grid=grid(768), stream=stream0)
        buf11 = empty((1, 256), device='cuda', dtype=torch.float16)
        buf267 = empty((1, 256), device='cuda', dtype=torch.float16)
        buf385 = empty((1, 256), device='cuda', dtype=torch.float16)
        triton_per_fused__to_copy_mean_4.run(buf9, buf11, buf267, buf385, 256, 3, grid=grid(256), stream=stream0)
        del buf9
        buf12 = empty((1, 32), device='cuda', dtype=torch.float16)
        extern_kernels.mm(buf11, reinterpret_tensor(arg14_1, (256, 32), (1, 256), 0), out=buf12)
        del arg14_1
        del buf11
        buf13 = buf12
        del buf12
        triton_poi_fused__to_copy_silu_5.run(buf13, arg15_1, 32, grid=grid(32), stream=stream0)
        del arg15_1
        buf14 = empty((1, 192), device='cuda', dtype=torch.float16)
        extern_kernels.mm(buf13, reinterpret_tensor(arg16_1, (32, 192), (1, 32), 0), out=buf14)
        del arg16_1
        buf15 = empty((2048, 40), device='cuda', dtype=torch.float32)
        triton_poi_fused_sub_6.run(arg174_1, arg176_1, buf15, 81920, grid=grid(81920), stream=stream0)
        del arg176_1
        buf16 = empty((41, 40), device='cuda', dtype=torch.float32)
        extern_kernels.mm(reinterpret_tensor(buf1, (41, 2048), (1, 41), 0), buf15, out=buf16)
        buf17 = buf16
        del buf16
        triton_poi_fused_add_mul_7.run(buf17, arg191_1, 1640, grid=grid(1640), stream=stream0)
        del arg191_1
        buf18 = aten.cholesky_solve(buf17, buf7)
        del buf7
        buf19 = buf18
        del buf18
        buf20 = buf1
        del buf1
        triton_poi_fused_cat_8.run(arg175_1, buf20, 83968, grid=grid(83968), stream=stream0)
        buf21 = buf15
        del buf15
        extern_kernels.mm(buf20, buf19, out=buf21)
        del buf20
        buf23 = empty((1, 32, 64, 2), device='cuda', dtype=torch.float32)
        buf64 = empty((1, 1, 32, 64, 2), device='cuda', dtype=torch.float32)
        buf70 = empty((1, 1, 32, 64, 2), device='cuda', dtype=torch.float32)
        buf71 = reinterpret_tensor(buf70, (1, 32, 64, 2), (4096, 128, 2, 1), 0)
        del buf70
        buf33 = empty((1, 32, 64, 2), device='cuda', dtype=torch.float32)
        buf37 = empty((1, 1, 32, 64, 2), device='cuda', dtype=torch.float32)
        buf65 = buf64
        del buf64
        buf72 = empty((32, 64, 2), device='cuda', dtype=torch.float32)
        triton_per_fused_add_clamp_detach_div_log_mean_mul_pow_sub_var_9.run(buf71, buf65, arg174_1, arg153_1, arg179_1, arg175_1, buf21, arg184_1, arg187_1, arg186_1, buf23, buf33, buf37, buf72, 4096, 20, grid=grid(4096), stream=stream0)
        del arg184_1
        del arg186_1
        del arg187_1
        buf25 = empty_strided((1, 2, 16), (32, 1, 2), device='cuda', dtype=torch.float32)
        triton_red_fused_mean_mul_var_10.run(buf23, buf25, 32, 128, grid=grid(32), stream=stream0)
        buf26 = empty((1, 2), device='cuda', dtype=torch.float32)
        triton_per_fused_mean_mul_var_11.run(buf25, buf26, 2, 16, grid=grid(2), stream=stream0)
        buf28 = buf23
        del buf23
        triton_per_fused_mul_var_12.run(arg173_1, arg157_1, buf28, 4096, 20, grid=grid(4096), stream=stream0)
        buf30 = buf25
        del buf25
        triton_red_fused_mean_mul_var_10.run(buf28, buf30, 32, 128, grid=grid(32), stream=stream0)
        buf31 = empty((1, 2), device='cuda', dtype=torch.float32)
        triton_per_fused_mean_mul_var_11.run(buf30, buf31, 2, 16, grid=grid(2), stream=stream0)
        buf35 = buf30
        del buf30
        triton_red_fused_mean_mul_var_10.run(buf33, buf35, 32, 128, grid=grid(32), stream=stream0)
        buf36 = empty((1, 2), device='cuda', dtype=torch.float32)
        buf640 = empty((1, 2), device='cuda', dtype=torch.float32)
        triton_per_fused_detach_mean_mul_var_13.run(buf35, buf36, buf640, 2, 16, grid=grid(2), stream=stream0)
        del buf35
        buf55 = empty((1, 20, 32, 64, 2), device='cuda', dtype=torch.float32)
        buf38 = empty((1, 20, 32, 64, 2), device='cuda', dtype=torch.float32)
        buf56 = empty((1, 20, 32, 64, 2), device='cuda', dtype=torch.float32)
        buf57 = empty((1, 20, 32, 64, 2), device='cuda', dtype=torch.float32)
        triton_poi_fused_add_clamp_exp_log_mean_mul_ne_sub_where_14.run(arg174_1, arg156_1, arg183_1, arg182_1, arg179_1, arg153_1, buf26, arg178_1, buf31, buf36, arg175_1, buf21, buf37, arg159_1, arg160_1, arg161_1, buf55, buf38, buf56, buf57, 81920, grid=grid(81920), stream=stream0)
        del arg159_1
        del arg160_1
        del arg161_1
        del arg178_1
        del arg182_1
        del buf26
        del buf31
        buf39 = empty_strided((7, 1, 32, 64, 2), (4096, 28672, 128, 2, 1), device='cuda', dtype=torch.float32)
        triton_per_fused__to_copy_div_exp_le_mean_mul_15.run(arg174_1, arg179_1, arg153_1, arg185_1, buf39, 28672, 20, grid=grid(28672), stream=stream0)
        del arg185_1
        buf40 = reinterpret_tensor(buf33, (1, 1, 32, 64, 2), (4096, 4096, 128, 2, 1), 0)
        del buf33
        triton_poi_fused__to_copy_div_exp_le_mean_mul_16.run(buf39, buf40, 4096, grid=grid(4096), stream=stream0)
        buf41 = reinterpret_tensor(buf39, (7, 1, 32, 64, 2), (4096, 4096, 128, 2, 1), 0)
        del buf39
        triton_poi_fused__to_copy_add_clamp_div_exp_le_mean_mul_sub_17.run(buf41, arg195_1, buf40, 28672, grid=grid(28672), stream=stream0)
        del arg195_1
        buf42 = buf40
        del buf40
        buf43 = reinterpret_tensor(buf28, (1, 1, 32, 64, 2), (4096, 4096, 128, 2, 1), 0)
        del buf28
        triton_poi_fused__softmax_18.run(buf41, buf42, buf43, 4096, grid=grid(4096), stream=stream0)
        buf44 = empty((7, 1, 1, 1, 1, 1), device='cuda', dtype=torch.float32)
        triton_poi_fused_clone_19.run(arg196_1, buf44, 7, grid=grid(7), stream=stream0)
        del arg196_1
        buf66 = empty((1, 1, 32, 64, 2), device='cuda', dtype=torch.float32)
        triton_per_fused_mean_mul_sub_20.run(buf37, buf38, arg175_1, buf21, arg153_1, buf66, 4096, 20, grid=grid(4096), stream=stream0)
        buf45 = empty((1, 1, 32, 64, 3), device='cuda', dtype=torch.float32)
        buf46 = buf45
        del buf45
        buf67 = empty((1, 1, 32, 64, 3), device='cuda', dtype=torch.float32)
        triton_per_fused_add_constant_pad_nd_div_mean_mul_sqrt_sub_21.run(buf46, arg174_1, arg179_1, arg153_1, arg180_1, arg163_1, buf65, arg164_1, buf66, buf67, 6144, 20, grid=grid(6144), stream=stream0)
        del arg163_1
        del arg164_1
        del arg180_1
        buf47 = empty_strided((1, 20, 1, 1, 2, 16), (640, 32, 640, 640, 1, 2), device='cuda', dtype=torch.float32)
        triton_red_fused_mean_pow_22.run(arg174_1, arg179_1, arg153_1, buf47, 640, 128, grid=grid(640), stream=stream0)
        del arg179_1
        buf48 = empty_strided((1, 20, 1, 1, 2), (40, 2, 40, 40, 1), device='cuda', dtype=torch.float32)
        triton_per_fused_mean_pow_23.run(buf47, buf48, 40, 16, grid=grid(40), stream=stream0)
        del buf47
        buf49 = reinterpret_tensor(buf36, (1, 1, 1, 1, 2), (2, 2, 2, 2, 1), 0)
        del buf36
        triton_per_fused_mean_pow_sqrt_24.run(buf48, buf49, 2, 20, grid=grid(2), stream=stream0)
        buf50 = reinterpret_tensor(buf48, (1, 20, 1, 1, 2), (40, 2, 2, 2, 1), 0)
        del buf48
        triton_poi_fused_add_clamp_min_div_gt_mean_mul_ones_like_pow_sqrt_where_25.run(buf50, arg181_1, buf49, 40, grid=grid(40), stream=stream0)
        del arg181_1
        buf51 = reinterpret_tensor(buf49, (1, 1, 1, 1, 1, 2), (2, 2, 2, 2, 2, 1), 0)
        del buf49
        triton_poi_fused_clone_26.run(arg197_1, buf51, 2, grid=grid(2), stream=stream0)
        del arg197_1
        buf52 = empty((7, 1, 20, 32, 64, 2), device='cuda', dtype=torch.float32)
        triton_poi_fused_minimum_mul_27.run(buf44, arg158_1, buf46, buf50, buf51, buf52, 573440, grid=grid(573440), stream=stream0)
        del arg158_1
        buf53 = empty((1, 20, 32, 64, 2), device='cuda', dtype=torch.float32)
        triton_poi_fused_mul_sum_28.run(buf41, buf42, buf43, buf52, buf53, 81920, grid=grid(81920), stream=stream0)
        buf58 = empty((1, 20, 32, 64, 2), device='cuda', dtype=torch.float32)
        triton_poi_fused_add_mul_29.run(arg173_1, arg165_1, arg166_1, buf58, 81920, grid=grid(81920), stream=stream0)
        del arg165_1
        del arg166_1
        buf59 = empty((4096, 20, 1), device='cuda', dtype=torch.float32)
        extern_kernels.bmm(reinterpret_tensor(arg167_1, (4096, 20, 20), (22, 90112, 1), 2), reinterpret_tensor(buf58, (4096, 20, 1), (1, 4096, 0), 0), out=buf59)
        buf54 = empty((1, 20, 32, 64, 3), device='cuda', dtype=torch.float32)
        buf61 = empty((1, 20, 32, 64, 3), device='cuda', dtype=torch.float32)
        buf68 = empty((1, 20, 32, 64, 3), device='cuda', dtype=torch.float32)
        buf60 = buf54
        del buf54
        buf62 = buf61
        del buf61
        buf69 = empty((1, 20, 32, 64, 3), device='cuda', dtype=torch.float32)
        triton_poi_fused_add_clamp_constant_pad_nd_copy_div_mean_mul_slice_scatter_sub_zeros_30.run(buf60, buf62, buf37, buf38, arg175_1, buf21, buf53, arg162_1, arg153_1, buf67, buf57, arg167_1, buf59, arg168_1, buf56, buf68, buf69, 20, 6144, grid=grid(20, 6144), stream=stream0)
        del arg162_1
        del arg167_1
        del arg168_1
        del buf67
        buf77 = buf43
        del buf43
        triton_per_fused_mean_31.run(buf69, buf77, 4096, 20, grid=grid(4096), stream=stream0)
        buf73 = buf42
        del buf42
        buf74 = reinterpret_tensor(buf66, (1, 32, 64, 2), (4096, 128, 2, 1), 0)
        del buf66
        buf85 = empty((1, 32, 64, 2), device='cuda', dtype=torch.float32)
        buf89 = empty((1, 32, 64, 2), device='cuda', dtype=torch.float32)
        buf105 = empty((1, 32, 64, 2), device='cuda', dtype=torch.float32)
        triton_per_fused__to_copy_eq_mean_mul_pow_std_sub_32.run(arg173_1, arg157_1, arg170_1, arg171_1, buf73, buf74, buf85, buf89, buf105, 4096, 20, grid=grid(4096), stream=stream0)
        buf75 = empty((1, 1, 32, 64, 2), device='cuda', dtype=torch.float32)
        buf76 = reinterpret_tensor(buf75, (1, 32, 64, 2), (4096, 128, 2, 1), 0)
        del buf75
        buf643 = empty((32, 64, 2), device='cuda', dtype=torch.float32)
        triton_per_fused_detach_mean_mul_pow_sub_33.run(buf76, buf68, arg153_1, buf643, 4096, 20, grid=grid(4096), stream=stream0)
        buf78 = reinterpret_tensor(buf59, (1, 20, 32, 64, 2), (81920, 4096, 128, 2, 1), 0)
        del buf59
        triton_poi_fused_mean_mul_sub_34.run(buf72, buf74, buf76, buf69, buf77, buf78, 81920, grid=grid(81920), stream=stream0)
        buf80 = reinterpret_tensor(buf73, (1, 32, 64, 2), (4096, 128, 2, 1), 0)
        del buf73
        buf91 = reinterpret_tensor(buf76, (32, 64, 2), (128, 2, 1), 0)
        del buf76
        buf92 = empty((32, 64, 2), device='cuda', dtype=torch.float32)
        buf81 = reinterpret_tensor(buf71, (32, 64, 2), (128, 2, 1), 0)
        del buf71
        buf82 = reinterpret_tensor(buf74, (1, 1, 32, 64, 2), (4096, 4096, 128, 2, 1), 0)
        del buf74
        buf644 = empty((32, 64, 2), device='cuda', dtype=torch.float32)
        triton_per_fused_add_clamp_detach_div_log_mean_mul_pow_35.run(buf80, buf81, buf82, buf78, arg153_1, arg173_1, arg157_1, arg170_1, arg171_1, arg188_1, arg189_1, arg169_1, buf91, buf92, buf644, 4096, 20, grid=grid(4096), stream=stream0)
        del arg188_1
        del arg189_1
        buf83 = buf69
        del buf69
        triton_poi_fused_add_clamp_copy_exp_mean_mul_slice_scatter_36.run(buf83, buf77, buf82, arg169_1, buf78, 122880, grid=grid(122880), stream=stream0)
        del arg169_1
        buf84 = reinterpret_tensor(buf78, (20, 32, 64, 2), (4096, 1, 64, 32), 0)
        del buf78
        triton_poi_fused_add_mul_37.run(buf83, arg153_1, arg154_1, buf84, 640, 128, grid=grid(640, 128), stream=stream0)
        buf136 = empty_strided((20, 2, 34, 66), (4488, 34, 1, 68), device='cuda', dtype=torch.float32)
        triton_poi_fused_replication_pad2d_38.run(buf84, buf136, 89760, grid=grid(89760), stream=stream0)
        buf137 = reinterpret_tensor(buf57, (20, 2, 32, 64), (4096, 2048, 64, 1), 0)
        del buf57
        triton_poi_fused_avg_pool2d_39.run(buf136, buf137, 1280, 64, grid=grid(1280, 64), stream=stream0)
        buf94 = reinterpret_tensor(buf56, (20, 32, 64, 2), (4096, 128, 2, 1), 0)
        del buf56
        buf128 = reinterpret_tensor(buf53, (20, 32, 64, 2), (4096, 1, 64, 32), 0)
        del buf53
        buf129 = reinterpret_tensor(buf58, (20, 32, 64, 2), (4096, 1, 64, 32), 0)
        del buf58
        buf154 = empty((20, 32, 64, 2, 8), device='cuda', dtype=torch.float32)
        buf146 = reinterpret_tensor(buf154, (20, 32, 64, 2, 1), (32768, 1024, 16, 8, 1), 0)
        buf147 = reinterpret_tensor(buf154, (20, 32, 64, 2, 1), (32768, 1024, 16, 8, 1), 1)
        buf148 = reinterpret_tensor(buf154, (20, 32, 64, 2, 1), (32768, 1024, 16, 8, 1), 2)
        buf150 = reinterpret_tensor(buf154, (20, 32, 64, 2, 1), (32768, 1024, 16, 8, 1), 4)
        triton_poi_fused_abs_add_mul_ne_stack_sub_where_40.run(arg174_1, arg152_1, arg153_1, arg154_1, arg198_1, arg204_1, arg206_1, arg205_1, buf62, buf60, buf137, buf84, buf94, buf128, buf129, buf146, buf147, buf148, buf150, 640, 128, grid=grid(640, 128), stream=stream0)
        del arg198_1
        del arg204_1
        del arg205_1
        buf97 = reinterpret_tensor(buf82, (32, 64, 2), (128, 2, 1), 0)
        del buf82
        buf100 = reinterpret_tensor(buf77, (32, 64, 2), (128, 2, 1), 0)
        del buf77
        buf99 = reinterpret_tensor(buf80, (32, 64, 2), (128, 2, 1), 0)
        del buf80
        buf101 = empty((32, 64, 2), device='cuda', dtype=torch.float32)
        buf645 = empty((32, 64, 2), device='cuda', dtype=torch.bool)
        triton_per_fused_abs_add_any_bitwise_or_clamp_min_div_mul_ne_sum_where_41.run(arg174_1, arg152_1, arg153_1, arg154_1, buf94, arg201_1, arg203_1, arg202_1, buf97, buf100, buf99, buf101, buf645, 4096, 20, grid=grid(4096), stream=stream0)
        del arg201_1
        del arg202_1
        del arg203_1
        buf103 = empty_strided((1, 32, 64, 2), (4096, 1, 64, 32), device='cuda', dtype=torch.float32)
        triton_per_fused_std_42.run(buf84, buf103, 4096, 20, grid=grid(4096), stream=stream0)
        buf126 = empty((20, 32, 64, 2, 20), device='cuda', dtype=torch.float32)
        buf107 = reinterpret_tensor(buf126, (20, 32, 64, 2, 1), (81920, 2560, 40, 20, 1), 1)
        buf63 = reinterpret_tensor(buf94, (20, 32, 64, 2), (4096, 1, 64, 32), 0)
        del buf94
        buf108 = reinterpret_tensor(buf126, (20, 32, 64, 2, 1), (81920, 2560, 40, 20, 1), 2)
        buf109 = reinterpret_tensor(buf126, (20, 32, 64, 2, 1), (81920, 2560, 40, 20, 1), 3)
        buf112 = reinterpret_tensor(buf126, (20, 32, 64, 2, 1), (81920, 2560, 40, 20, 1), 6)
        buf113 = reinterpret_tensor(buf126, (20, 32, 64, 2, 1), (81920, 2560, 40, 20, 1), 7)
        buf114 = reinterpret_tensor(buf126, (20, 32, 64, 2, 1), (81920, 2560, 40, 20, 1), 8)
        buf115 = reinterpret_tensor(buf126, (20, 32, 64, 2, 1), (81920, 2560, 40, 20, 1), 9)
        buf116 = reinterpret_tensor(buf126, (20, 32, 64, 2, 1), (81920, 2560, 40, 20, 1), 10)
        buf117 = reinterpret_tensor(buf126, (20, 32, 64, 2, 1), (81920, 2560, 40, 20, 1), 11)
        buf118 = reinterpret_tensor(buf126, (20, 32, 64, 2, 1), (81920, 2560, 40, 20, 1), 12)
        buf125 = reinterpret_tensor(buf126, (20, 32, 64, 2, 1), (81920, 2560, 40, 20, 1), 19)
        buf106 = reinterpret_tensor(buf126, (20, 32, 64, 2, 1), (81920, 2560, 40, 20, 1), 0)
        buf110 = reinterpret_tensor(buf126, (20, 32, 64, 2, 1), (81920, 2560, 40, 20, 1), 4)
        buf111 = reinterpret_tensor(buf126, (20, 32, 64, 2, 1), (81920, 2560, 40, 20, 1), 5)
        triton_poi_fused_add_clamp_min_div_mul_stack_sub_43.run(buf62, arg153_1, arg154_1, buf60, arg155_1, arg173_1, arg170_1, arg157_1, arg171_1, buf84, buf85, buf89, buf91, buf92, buf97, buf99, buf100, buf101, buf103, buf107, buf63, buf108, buf109, buf112, buf113, buf114, buf115, buf116, buf117, buf118, buf125, buf106, buf110, buf111, 640, 128, grid=grid(640, 128), stream=stream0)
        buf119 = reinterpret_tensor(buf126, (20, 32, 64, 2, 1), (81920, 2560, 40, 20, 1), 13)
        triton_poi_fused_stack_44.run(buf119, 81920, grid=grid(81920), stream=stream0)
        buf120 = reinterpret_tensor(buf126, (20, 32, 64, 2, 1), (81920, 2560, 40, 20, 1), 14)
        triton_poi_fused_stack_45.run(buf120, 81920, grid=grid(81920), stream=stream0)
        buf121 = reinterpret_tensor(buf126, (20, 32, 64, 2, 1), (81920, 2560, 40, 20, 1), 15)
        triton_poi_fused_stack_46.run(buf121, 81920, grid=grid(81920), stream=stream0)
        buf122 = reinterpret_tensor(buf126, (20, 32, 64, 2, 1), (81920, 2560, 40, 20, 1), 16)
        triton_poi_fused_stack_47.run(buf122, 81920, grid=grid(81920), stream=stream0)
        buf123 = reinterpret_tensor(buf126, (20, 32, 64, 2, 1), (81920, 2560, 40, 20, 1), 17)
        triton_poi_fused_stack_48.run(buf123, 81920, grid=grid(81920), stream=stream0)
        buf124 = reinterpret_tensor(buf126, (20, 32, 64, 2, 1), (81920, 2560, 40, 20, 1), 18)
        triton_poi_fused_stack_49.run(buf105, buf124, 81920, grid=grid(81920), stream=stream0)
        buf133 = reinterpret_tensor(buf136, (20, 2, 34, 66), (4488, 2244, 66, 1), 0)
        del buf136
        buf130 = reinterpret_tensor(buf133, (20, 2, 1, 66), (4488, 2244, 66, 1), 0)
        buf141 = empty((20, 2, 34, 66), device='cuda', dtype=torch.float32)
        buf138 = reinterpret_tensor(buf141, (20, 2, 1, 66), (4488, 2244, 66, 1), 0)
        buf132 = reinterpret_tensor(buf133, (20, 2, 1, 66), (4488, 2244, 66, 1), 2178)
        buf140 = reinterpret_tensor(buf141, (20, 2, 1, 66), (4488, 2244, 66, 1), 2178)
        triton_poi_fused_replication_pad2d_50.run(arg173_1, arg170_1, arg157_1, arg171_1, buf130, buf138, buf132, buf140, 40, 66, grid=grid(40, 66), stream=stream0)
        del buf106
        del buf107
        del buf108
        del buf109
        del buf110
        del buf111
        del buf112
        del buf113
        del buf114
        del buf115
        del buf116
        del buf117
        del buf118
        del buf119
        del buf120
        del buf121
        del buf122
        del buf123
        del buf124
        del buf125
        buf131 = reinterpret_tensor(buf133, (20, 2, 32, 66), (4488, 2244, 66, 1), 66)
        buf139 = reinterpret_tensor(buf141, (20, 2, 32, 66), (4488, 2244, 66, 1), 66)
        triton_poi_fused_replication_pad2d_51.run(arg173_1, arg170_1, arg157_1, arg171_1, buf131, buf139, 40, 2112, grid=grid(40, 2112), stream=stream0)
        buf134 = buf137
        del buf137
        triton_poi_fused_avg_pool2d_52.run(buf133, buf134, 81920, grid=grid(81920), stream=stream0)
        del buf130
        del buf131
        del buf132
        del buf133
        buf142 = empty((20, 2, 32, 64), device='cuda', dtype=torch.float32)
        triton_poi_fused_avg_pool2d_52.run(buf141, buf142, 81920, grid=grid(81920), stream=stream0)
        del buf138
        del buf139
        del buf140
        del buf141
        buf151 = reinterpret_tensor(buf154, (20, 32, 64, 2, 1), (32768, 1024, 16, 8, 1), 5)
        buf149 = reinterpret_tensor(buf154, (20, 32, 64, 2, 1), (32768, 1024, 16, 8, 1), 3)
        triton_poi_fused_add_clamp_min_div_mul_stack_sub_53.run(buf134, arg173_1, arg170_1, arg157_1, arg171_1, buf62, arg153_1, arg154_1, buf60, buf142, buf151, buf149, 40960, 2, grid=grid(40960, 2), stream=stream0)
        buf144 = buf105
        del buf105
        buf170 = buf92
        del buf92
        triton_per_fused__to_copy_eq_mean_std_54.run(arg173_1, arg170_1, arg157_1, arg171_1, buf144, buf170, 4096, 20, grid=grid(4096), stream=stream0)
        buf152 = reinterpret_tensor(buf154, (20, 32, 64, 2, 1), (32768, 1024, 16, 8, 1), 6)
        buf153 = reinterpret_tensor(buf154, (20, 32, 64, 2, 1), (32768, 1024, 16, 8, 1), 7)
        triton_poi_fused_stack_55.run(buf144, arg155_1, buf97, buf152, buf153, 81920, grid=grid(81920), stream=stream0)
        buf155 = empty((20, 32, 64, 2, 28), device='cuda', dtype=torch.float16)
        triton_poi_fused__to_copy_cat_56.run(buf126, buf154, buf155, 2293760, grid=grid(2293760), stream=stream0)
        del buf126
        del buf146
        del buf147
        del buf148
        del buf149
        del buf150
        del buf151
        del buf152
        del buf153
        del buf154
        buf156 = empty((81920, 32), device='cuda', dtype=torch.float16)
        extern_kernels.mm(reinterpret_tensor(buf155, (81920, 28), (28, 1), 0), reinterpret_tensor(arg0_1, (28, 32), (1, 28), 0), out=buf156)
        del arg0_1
        del buf155
        buf157 = reinterpret_tensor(buf156, (20, 32, 64, 2, 32), (131072, 4096, 64, 32, 1), 0)
        del buf156
        triton_poi_fused__to_copy_silu_57.run(buf157, arg1_1, 2621440, grid=grid(2621440), stream=stream0)
        del arg1_1
        buf158 = empty((81920, 32), device='cuda', dtype=torch.float16)
        extern_kernels.mm(reinterpret_tensor(buf157, (81920, 32), (32, 1), 0), reinterpret_tensor(arg2_1, (32, 32), (1, 32), 0), out=buf158)
        del arg2_1
        del buf157
        buf159 = reinterpret_tensor(buf158, (20, 32, 64, 2, 32), (131072, 4096, 64, 32, 1), 0)
        del buf158
        triton_poi_fused__to_copy_silu_57.run(buf159, arg3_1, 2621440, grid=grid(2621440), stream=stream0)
        del arg3_1
        buf160 = empty((81920, 2), device='cuda', dtype=torch.float16)
        extern_kernels.mm(reinterpret_tensor(buf159, (81920, 32), (32, 1), 0), reinterpret_tensor(arg4_1, (32, 2), (1, 32), 0), out=buf160)
        del arg4_1
        del buf159
        buf161 = reinterpret_tensor(buf142, (20, 32, 64, 2), (4096, 128, 2, 1), 0)
        del buf142
        buf162 = reinterpret_tensor(buf134, (20, 32, 64, 2), (4096, 128, 2, 1), 0)
        del buf134
        buf163 = empty_strided((20, 32, 64, 2), (4096, 1, 64, 32), device='cuda', dtype=torch.float32)
        buf164 = empty((20, 32, 64, 2), device='cuda', dtype=torch.float32)
        buf165 = empty_strided((20, 32, 64, 2), (4096, 1, 64, 32), device='cuda', dtype=torch.float32)
        triton_poi_fused_add_exp_mul_ne_sub_tanh_where_58.run(buf63, buf62, arg153_1, arg154_1, buf60, buf160, arg5_1, arg174_1, arg152_1, arg212_1, arg213_1, buf161, buf162, buf163, buf164, buf165, 640, 128, grid=grid(640, 128), stream=stream0)
        del arg212_1
        del arg213_1
        del arg5_1
        del buf160
        buf167 = reinterpret_tensor(buf144, (32, 64, 2), (128, 2, 1), 0)
        del buf144
        buf169 = buf167
        del buf167
        buf647 = empty((32, 64, 2), device='cuda', dtype=torch.bool)
        triton_per_fused_abs_add_any_bitwise_or_clamp_min_div_mul_ne_sum_where_59.run(buf169, arg174_1, arg152_1, arg153_1, arg154_1, buf164, arg215_1, arg214_1, buf647, 4096, 20, grid=grid(4096), stream=stream0)
        del arg214_1
        del arg215_1
        buf180 = empty((167, 32, 64), device='cuda', dtype=torch.float32)
        buf171 = reinterpret_tensor(buf180, (40, 32, 64), (2048, 64, 1), 0)
        triton_poi_fused_cat_60.run(arg173_1, arg170_1, arg157_1, arg171_1, arg155_1, buf171, 40, 2048, grid=grid(40, 2048), stream=stream0)
        buf172 = reinterpret_tensor(buf180, (40, 32, 64), (2048, 64, 1), 81920)
        buf174 = reinterpret_tensor(buf180, (40, 32, 64), (2048, 64, 1), 245760)
        triton_poi_fused_cat_61.run(buf163, arg155_1, buf165, buf172, buf174, 1280, 64, grid=grid(1280, 64), stream=stream0)
        buf173 = reinterpret_tensor(buf180, (40, 32, 64), (2048, 64, 1), 163840)
        triton_poi_fused_cat_62.run(buf162, arg154_1, arg153_1, buf62, buf161, buf60, arg155_1, buf173, 2048, 40, grid=grid(2048, 40), stream=stream0)
        buf175 = reinterpret_tensor(buf180, (2, 32, 64), (2048, 64, 1), 327680)
        triton_poi_fused_cat_63.run(buf169, arg155_1, buf175, 2, 2048, grid=grid(2, 2048), stream=stream0)
        buf176 = reinterpret_tensor(buf180, (2, 32, 64), (2048, 64, 1), 331776)
        triton_poi_fused_cat_64.run(buf170, buf176, 2, 2048, grid=grid(2, 2048), stream=stream0)
        buf177 = reinterpret_tensor(buf180, (1, 32, 64), (2048, 64, 1), 335872)
        triton_poi_fused_cat_65.run(buf177, 2048, grid=grid(2048), stream=stream0)
        buf178 = reinterpret_tensor(buf180, (1, 32, 64), (2048, 64, 1), 337920)
        triton_poi_fused_cat_66.run(buf178, 2048, grid=grid(2048), stream=stream0)
        buf179 = reinterpret_tensor(buf180, (1, 32, 64), (2048, 64, 1), 339968)
        triton_poi_fused_full_like_67.run(buf179, 2048, grid=grid(2048), stream=stream0)
        buf181 = reinterpret_tensor(buf170, (1, 1, 2, 32, 64), (4096, 4096, 2048, 64, 1), 0)
        del buf170
        buf183 = reinterpret_tensor(buf91, (1, 1, 2, 32, 64), (4096, 4096, 2048, 64, 1), 0)
        del buf91
        triton_per_fused_mean_std_68.run(buf180, buf181, buf183, 4096, 20, grid=grid(4096), stream=stream0)
        del buf171
        del buf172
        del buf173
        del buf174
        del buf175
        del buf176
        del buf177
        del buf178
        del buf179
        buf185 = empty((1, 80, 2, 32, 64), device='cuda', dtype=torch.float32)
        triton_poi_fused_cat_69.run(buf180, buf181, buf183, buf185, 327680, grid=grid(327680), stream=stream0)
        buf186 = empty_strided((1, 327, 32, 64), (669696, 1, 20928, 327), device='cuda', dtype=torch.float16)
        triton_poi_fused__to_copy_cat_70.run(buf180, buf185, buf186, 327, 2048, grid=grid(327, 2048), stream=stream0)
        buf187 = extern_kernels.convolution(buf186, arg12_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf187, (1, 96, 32, 64), (196608, 1, 6144, 96))
        del arg12_1
        del buf186
        buf188 = empty_strided((1, 96, 32, 64), (196608, 1, 6144, 96), device='cuda', dtype=torch.float32)
        triton_poi_fused__to_copy_add_cat_convolution_mul_tanh_71.run(buf187, arg13_1, buf14, arg17_1, buf188, 196608, grid=grid(196608), stream=stream0)
        del arg13_1
        del arg17_1
        del buf14
        del buf187
        buf189 = empty_strided((1, 96, 36, 68), (235008, 1, 6528, 96), device='cuda', dtype=torch.float32)
        triton_poi_fused_replication_pad2d_72.run(buf188, buf189, 235008, grid=grid(235008), stream=stream0)
        buf190 = empty_strided((1, 96, 36, 68), (235008, 1, 6528, 96), device='cuda', dtype=torch.float16)
        triton_poi_fused_replication_pad2d_73.run(buf189, buf190, 96, 2448, grid=grid(96, 2448), stream=stream0)
        del buf189
        buf191 = extern_kernels.convolution(buf190, arg18_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=96, bias=None)
        assert_size_stride(buf191, (1, 96, 32, 64), (196608, 1, 6144, 96))
        del arg18_1
        del buf190
        buf192 = buf191
        del buf191
        triton_poi_fused__to_copy_convolution_replication_pad2d_silu_74.run(buf192, arg19_1, 196608, grid=grid(196608), stream=stream0)
        del arg19_1
        buf193 = extern_kernels.convolution(buf192, arg20_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf193, (1, 96, 32, 64), (196608, 1, 6144, 96))
        del arg20_1
        del buf192
        buf197 = empty((1, 96, 40, 72), device='cuda', dtype=torch.float32)
        buf194 = reinterpret_tensor(buf197, (1, 96, 4, 72), (276480, 2880, 72, 1), 0)
        buf196 = reinterpret_tensor(buf197, (1, 96, 4, 72), (276480, 2880, 72, 1), 2592)
        triton_poi_fused_replication_pad2d_75.run(buf188, buf193, arg21_1, buf194, buf196, 96, 288, grid=grid(96, 288), stream=stream0)
        buf195 = reinterpret_tensor(buf197, (1, 96, 32, 72), (276480, 2880, 72, 1), 288)
        triton_poi_fused_replication_pad2d_76.run(buf188, buf193, arg21_1, buf195, 3072, 72, grid=grid(3072, 72), stream=stream0)
        buf198 = empty_strided((1, 96, 40, 72), (276480, 1, 6912, 96), device='cuda', dtype=torch.float16)
        triton_poi_fused_replication_pad2d_77.run(buf197, buf198, 96, 2880, grid=grid(96, 2880), stream=stream0)
        del buf194
        del buf195
        del buf196
        del buf197
        buf199 = extern_kernels.convolution(buf198, arg22_1, stride=(1, 1), padding=(0, 0), dilation=(2, 2), transposed=False, output_padding=(0, 0), groups=96, bias=None)
        assert_size_stride(buf199, (1, 96, 32, 64), (196608, 1, 6144, 96))
        del arg22_1
        del buf198
        buf200 = buf199
        del buf199
        triton_poi_fused__to_copy_convolution_replication_pad2d_silu_74.run(buf200, arg23_1, 196608, grid=grid(196608), stream=stream0)
        del arg23_1
        buf201 = extern_kernels.convolution(buf200, arg24_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf201, (1, 96, 32, 64), (196608, 1, 6144, 96))
        del arg24_1
        del buf200
        buf202 = buf193
        del buf193
        triton_poi_fused__to_copy_add_convolution_replication_pad2d_silu_78.run(buf202, buf188, arg21_1, buf201, arg25_1, 196608, grid=grid(196608), stream=stream0)
        del arg21_1
        del arg25_1
        del buf201
        buf203 = extern_kernels.convolution(buf202, arg26_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf203, (1, 80, 32, 64), (163840, 1, 5120, 80))
        del arg26_1
        del buf202
        buf204 = empty_strided((1, 96, 16, 32), (49152, 1, 3072, 96), device='cuda', dtype=torch.float16)
        triton_poi_fused__to_copy_avg_pool2d_silu_79.run(buf188, buf204, 49152, grid=grid(49152), stream=stream0)
        del buf188
        buf205 = extern_kernels.convolution(buf204, arg28_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf205, (1, 64, 16, 32), (32768, 1, 2048, 64))
        del arg28_1
        del buf204
        buf209 = empty((1, 64, 20, 36), device='cuda', dtype=torch.float32)
        buf206 = reinterpret_tensor(buf209, (1, 64, 2, 36), (46080, 720, 36, 1), 0)
        buf208 = reinterpret_tensor(buf209, (1, 64, 2, 36), (46080, 720, 36, 1), 648)
        triton_poi_fused_replication_pad2d_80.run(buf205, arg29_1, buf206, buf208, 64, 72, grid=grid(64, 72), stream=stream0)
        buf207 = reinterpret_tensor(buf209, (1, 64, 16, 36), (46080, 720, 36, 1), 72)
        triton_poi_fused_replication_pad2d_81.run(buf205, arg29_1, buf207, 64, 576, grid=grid(64, 576), stream=stream0)
        buf210 = empty_strided((1, 64, 20, 36), (46080, 1, 2304, 64), device='cuda', dtype=torch.float16)
        triton_poi_fused_replication_pad2d_82.run(buf209, buf210, 64, 720, grid=grid(64, 720), stream=stream0)
        del buf206
        del buf207
        del buf208
        del buf209
        buf211 = extern_kernels.convolution(buf210, arg30_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf211, (1, 64, 16, 32), (32768, 1, 2048, 64))
        del arg30_1
        buf212 = buf211
        del buf211
        triton_poi_fused__to_copy_convolution_replication_pad2d_silu_83.run(buf212, arg31_1, 32768, grid=grid(32768), stream=stream0)
        del arg31_1
        buf213 = extern_kernels.convolution(buf212, arg32_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf213, (1, 64, 16, 32), (32768, 1, 2048, 64))
        del arg32_1
        del buf212
        buf214 = empty_strided((1, 64, 16, 32), (32768, 1, 2048, 64), device='cuda', dtype=torch.float32)
        triton_poi_fused__to_copy_add_avg_pool2d_convolution_replication_pad2d_silu_84.run(buf205, arg29_1, buf213, arg33_1, buf214, 32768, grid=grid(32768), stream=stream0)
        del arg29_1
        del arg33_1
        del buf205
        del buf213
        buf215 = empty_strided((1, 64, 8, 16), (8192, 1, 1024, 64), device='cuda', dtype=torch.float16)
        triton_poi_fused__to_copy_avg_pool2d_85.run(buf214, buf215, 8192, grid=grid(8192), stream=stream0)
        buf216 = extern_kernels.convolution(buf215, arg34_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf216, (1, 96, 8, 16), (12288, 1, 1536, 96))
        del arg34_1
        del buf215
        buf220 = empty((1, 96, 12, 20), device='cuda', dtype=torch.float32)
        buf217 = reinterpret_tensor(buf220, (1, 96, 2, 20), (23040, 240, 20, 1), 0)
        buf219 = reinterpret_tensor(buf220, (1, 96, 2, 20), (23040, 240, 20, 1), 200)
        triton_poi_fused_replication_pad2d_86.run(buf216, arg35_1, buf217, buf219, 96, 40, grid=grid(96, 40), stream=stream0)
        buf218 = reinterpret_tensor(buf220, (1, 96, 8, 20), (23040, 240, 20, 1), 40)
        triton_poi_fused_replication_pad2d_87.run(buf216, arg35_1, buf218, 96, 160, grid=grid(96, 160), stream=stream0)
        buf221 = empty_strided((1, 96, 12, 20), (23040, 1, 1920, 96), device='cuda', dtype=torch.float16)
        triton_poi_fused_replication_pad2d_88.run(buf220, buf221, 96, 240, grid=grid(96, 240), stream=stream0)
        del buf217
        del buf218
        del buf219
        del buf220
        buf222 = extern_kernels.convolution(buf221, arg36_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=96, bias=None)
        assert_size_stride(buf222, (1, 96, 8, 16), (12288, 1, 1536, 96))
        del arg36_1
        buf223 = buf222
        del buf222
        triton_poi_fused__to_copy_convolution_replication_pad2d_silu_89.run(buf223, arg37_1, 12288, grid=grid(12288), stream=stream0)
        del arg37_1
        buf224 = extern_kernels.convolution(buf223, arg38_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf224, (1, 96, 8, 16), (12288, 1, 1536, 96))
        del arg38_1
        del buf223
        buf225 = empty_strided((1, 96, 8, 16), (12288, 1, 1536, 96), device='cuda', dtype=torch.float32)
        triton_poi_fused__to_copy_add_avg_pool2d_convolution_replication_pad2d_silu_90.run(buf216, arg35_1, buf224, arg39_1, buf225, 12288, grid=grid(12288), stream=stream0)
        del arg35_1
        del arg39_1
        del buf216
        buf226 = empty_strided((1, 96, 4, 8), (3072, 1, 768, 96), device='cuda', dtype=torch.float16)
        triton_poi_fused__to_copy_avg_pool2d_91.run(buf225, buf226, 3072, grid=grid(3072), stream=stream0)
        buf227 = extern_kernels.convolution(buf226, arg40_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf227, (1, 128, 4, 8), (4096, 1, 1024, 128))
        del arg40_1
        del buf226
        buf231 = empty((1, 128, 8, 12), device='cuda', dtype=torch.float32)
        buf228 = reinterpret_tensor(buf231, (1, 128, 2, 12), (12288, 96, 12, 1), 0)
        buf230 = reinterpret_tensor(buf231, (1, 128, 2, 12), (12288, 96, 12, 1), 72)
        triton_poi_fused_replication_pad2d_92.run(buf227, arg41_1, buf228, buf230, 128, 24, grid=grid(128, 24), stream=stream0)
        buf229 = reinterpret_tensor(buf231, (1, 128, 4, 12), (12288, 96, 12, 1), 24)
        triton_poi_fused_replication_pad2d_93.run(buf227, arg41_1, buf229, 128, 48, grid=grid(128, 48), stream=stream0)
        buf232 = reinterpret_tensor(buf224, (1, 128, 8, 12), (12288, 1, 1536, 128), 0)
        del buf224
        triton_poi_fused_replication_pad2d_94.run(buf231, buf232, 128, 96, grid=grid(128, 96), stream=stream0)
        del buf228
        del buf229
        del buf230
        buf233 = extern_kernels.convolution(buf232, arg42_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=128, bias=None)
        assert_size_stride(buf233, (1, 128, 4, 8), (4096, 1, 1024, 128))
        del arg42_1
        del buf232
        buf234 = buf233
        del buf233
        triton_poi_fused__to_copy_convolution_replication_pad2d_silu_95.run(buf234, arg43_1, 4096, grid=grid(4096), stream=stream0)
        del arg43_1
        buf235 = extern_kernels.convolution(buf234, arg44_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf235, (1, 128, 4, 8), (4096, 1, 1024, 128))
        del arg44_1
        del buf234
        buf236 = buf227
        del buf227
        triton_poi_fused__to_copy_add_avg_pool2d_convolution_replication_pad2d_silu_96.run(buf236, arg41_1, buf235, arg45_1, 4096, grid=grid(4096), stream=stream0)
        del arg41_1
        del arg45_1
        del buf235
        buf237 = extern_kernels.convolution(buf236, arg46_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf237, (1, 96, 4, 8), (3072, 1, 768, 96))
        del arg46_1
        del buf236
        buf238 = reinterpret_tensor(buf231, (1, 96, 8, 16), (12288, 128, 16, 1), 0)
        del buf231
        buf242 = buf238
        del buf238
        triton_poi_fused__to_copy__unsafe_index_add_arange_avg_pool2d_clamp_convolution_mul_replication_pad2d_rsub_silu_sub_97.run(buf242, buf237, arg47_1, 12288, grid=grid(12288), stream=stream0)
        del arg47_1
        del buf237
        buf244 = buf221
        del buf221
        triton_poi_fused_replication_pad2d_98.run(buf225, buf242, buf244, 96, 240, grid=grid(96, 240), stream=stream0)
        buf245 = extern_kernels.convolution(buf244, arg48_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=96, bias=None)
        assert_size_stride(buf245, (1, 96, 8, 16), (12288, 1, 1536, 96))
        del arg48_1
        del buf244
        buf246 = buf245
        del buf245
        triton_poi_fused__to_copy_convolution_replication_pad2d_silu_89.run(buf246, arg49_1, 12288, grid=grid(12288), stream=stream0)
        del arg49_1
        buf247 = extern_kernels.convolution(buf246, arg50_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf247, (1, 96, 8, 16), (12288, 1, 1536, 96))
        del arg50_1
        del buf246
        buf248 = buf247
        del buf247
        triton_poi_fused__to_copy_add_convolution_replication_pad2d_silu_99.run(buf248, buf225, buf242, arg51_1, 128, 96, grid=grid(128, 96), stream=stream0)
        del arg51_1
        del buf225
        buf249 = extern_kernels.convolution(buf248, arg52_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf249, (1, 64, 8, 16), (8192, 1, 1024, 64))
        del arg52_1
        buf250 = empty((1, 64, 16, 32), device='cuda', dtype=torch.float32)
        buf254 = buf250
        del buf250
        triton_poi_fused__to_copy__unsafe_index_add_arange_clamp_convolution_mul_replication_pad2d_rsub_silu_sub_100.run(buf254, buf249, arg53_1, 32768, grid=grid(32768), stream=stream0)
        del arg53_1
        buf256 = buf210
        del buf210
        triton_poi_fused_replication_pad2d_101.run(buf214, buf254, buf256, 64, 720, grid=grid(64, 720), stream=stream0)
        buf257 = extern_kernels.convolution(buf256, arg54_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf257, (1, 64, 16, 32), (32768, 1, 2048, 64))
        del arg54_1
        del buf256
        buf258 = buf257
        del buf257
        triton_poi_fused__to_copy_convolution_replication_pad2d_silu_83.run(buf258, arg55_1, 32768, grid=grid(32768), stream=stream0)
        del arg55_1
        buf259 = extern_kernels.convolution(buf258, arg56_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf259, (1, 64, 16, 32), (32768, 1, 2048, 64))
        del arg56_1
        del buf258
        buf260 = buf259
        del buf259
        triton_poi_fused__to_copy_add_convolution_replication_pad2d_silu_102.run(buf260, buf214, buf254, arg57_1, 512, 64, grid=grid(512, 64), stream=stream0)
        del arg57_1
        buf261 = extern_kernels.convolution(buf260, arg58_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf261, (1, 80, 16, 32), (40960, 1, 2560, 80))
        del arg58_1
        buf262 = empty((1, 80, 32, 64), device='cuda', dtype=torch.float32)
        buf266 = buf262
        del buf262
        triton_poi_fused__to_copy__unsafe_index_add_arange_clamp_convolution_mul_replication_pad2d_rsub_silu_sub_103.run(buf266, buf261, arg59_1, 163840, grid=grid(163840), stream=stream0)
        del arg59_1
        del buf261
        buf268 = buf13
        del buf13
        extern_kernels.mm(buf267, reinterpret_tensor(arg62_1, (256, 32), (1, 256), 0), out=buf268)
        del arg62_1
        del buf267
        buf269 = buf268
        del buf268
        triton_poi_fused__to_copy_silu_5.run(buf269, arg63_1, 32, grid=grid(32), stream=stream0)
        del arg63_1
        buf270 = empty((1, 128), device='cuda', dtype=torch.float16)
        extern_kernels.mm(buf269, reinterpret_tensor(arg64_1, (32, 128), (1, 32), 0), out=buf270)
        del arg64_1
        del buf269
        buf271 = reinterpret_tensor(buf183, (32, 64, 2), (1, 64, 32), 0)
        del buf183
        buf272 = reinterpret_tensor(buf181, (32, 64, 2), (1, 64, 32), 0)
        del buf181
        buf274 = reinterpret_tensor(buf89, (32, 64, 2), (1, 64, 32), 0)
        del buf89
        triton_per_fused_mean_var_104.run(buf84, buf271, buf272, buf274, 4096, 20, grid=grid(4096), stream=stream0)
        buf276 = reinterpret_tensor(buf85, (32, 64, 2), (128, 2, 1), 0)
        del buf85
        buf280 = reinterpret_tensor(buf103, (32, 64, 2), (128, 2, 1), 0)
        del buf103
        buf305 = empty((32, 64, 2), device='cuda', dtype=torch.float32)
        buf330 = empty((32, 64, 2, 22), device='cuda', dtype=torch.float32)
        buf323 = reinterpret_tensor(buf330, (32, 64, 2, 1), (2816, 44, 22, 1), 15)
        triton_per_fused__to_copy_eq_mean_stack_var_105.run(arg173_1, arg170_1, arg157_1, arg171_1, buf276, buf280, buf305, buf323, 4096, 20, grid=grid(4096), stream=stream0)
        buf283 = empty((32, 64, 2), device='cuda', dtype=torch.float32)
        buf286 = empty((32, 64, 2), device='cuda', dtype=torch.float32)
        triton_per_fused_var_106.run(arg173_1, arg170_1, arg157_1, arg171_1, buf283, buf286, 4096, 10, grid=grid(4096), stream=stream0)
        buf288 = empty((32, 64, 2), device='cuda', dtype=torch.float32)
        triton_per_fused_mean_pow_sub_107.run(arg173_1, arg170_1, arg157_1, arg171_1, buf288, 4096, 19, grid=grid(4096), stream=stream0)
        buf289 = empty_strided((1, 2, 34, 66), (4488, 1, 132, 2), device='cuda', dtype=torch.float32)
        triton_poi_fused_replication_pad2d_108.run(buf280, buf289, 4488, grid=grid(4488), stream=stream0)
        buf290 = empty((1, 2, 32, 64), device='cuda', dtype=torch.float32)
        triton_poi_fused_avg_pool2d_109.run(buf289, buf290, 2, 2048, grid=grid(2, 2048), stream=stream0)
        buf291 = buf289
        del buf289
        triton_poi_fused_replication_pad2d_108.run(buf276, buf291, 4488, grid=grid(4488), stream=stream0)
        buf292 = empty((1, 2, 32, 64), device='cuda', dtype=torch.float32)
        triton_poi_fused_avg_pool2d_109.run(buf291, buf292, 2, 2048, grid=grid(2, 2048), stream=stream0)
        buf294 = empty((32, 64, 2), device='cuda', dtype=torch.float32)
        buf301 = empty((32, 64, 2), device='cuda', dtype=torch.float32)
        buf299 = buf294
        del buf294
        buf300 = empty((32, 64, 2), device='cuda', dtype=torch.float32)
        buf319 = reinterpret_tensor(buf330, (32, 64, 2, 1), (2816, 44, 22, 1), 11)
        buf320 = reinterpret_tensor(buf330, (32, 64, 2, 1), (2816, 44, 22, 1), 12)
        buf302 = buf301
        del buf301
        buf303 = empty((32, 64, 2), device='cuda', dtype=torch.float32)
        buf321 = reinterpret_tensor(buf330, (32, 64, 2, 1), (2816, 44, 22, 1), 13)
        buf322 = reinterpret_tensor(buf330, (32, 64, 2, 1), (2816, 44, 22, 1), 14)
        buf308 = reinterpret_tensor(buf330, (32, 64, 2, 1), (2816, 44, 22, 1), 0)
        buf310 = reinterpret_tensor(buf330, (32, 64, 2, 1), (2816, 44, 22, 1), 2)
        buf311 = reinterpret_tensor(buf330, (32, 64, 2, 1), (2816, 44, 22, 1), 3)
        buf312 = reinterpret_tensor(buf330, (32, 64, 2, 1), (2816, 44, 22, 1), 4)
        buf309 = reinterpret_tensor(buf330, (32, 64, 2, 1), (2816, 44, 22, 1), 1)
        buf313 = reinterpret_tensor(buf330, (32, 64, 2, 1), (2816, 44, 22, 1), 5)
        buf314 = reinterpret_tensor(buf330, (32, 64, 2, 1), (2816, 44, 22, 1), 6)
        buf315 = reinterpret_tensor(buf330, (32, 64, 2, 1), (2816, 44, 22, 1), 7)
        buf316 = reinterpret_tensor(buf330, (32, 64, 2, 1), (2816, 44, 22, 1), 8)
        buf317 = reinterpret_tensor(buf330, (32, 64, 2, 1), (2816, 44, 22, 1), 9)
        buf318 = reinterpret_tensor(buf330, (32, 64, 2, 1), (2816, 44, 22, 1), 10)
        triton_per_fused_add_clamp_div_log_mean_mul_stack_sub_var_110.run(buf299, buf302, arg207_1, arg174_1, arg152_1, arg153_1, arg154_1, arg210_1, arg155_1, arg211_1, buf272, buf276, arg173_1, arg170_1, arg157_1, arg171_1, buf274, buf280, buf283, buf286, buf288, buf290, buf292, buf300, buf319, buf320, buf303, buf321, buf322, buf308, buf310, buf311, buf312, buf309, buf313, buf314, buf315, buf316, buf317, buf318, 4096, 20, grid=grid(4096), stream=stream0)
        del arg157_1
        del arg170_1
        del arg171_1
        del arg173_1
        del arg207_1
        del arg210_1
        del arg211_1
        del buf272
        buf306 = buf291
        del buf291
        triton_poi_fused_replication_pad2d_108.run(buf305, buf306, 4488, grid=grid(4488), stream=stream0)
        buf307 = reinterpret_tensor(buf305, (1, 2, 32, 64), (4096, 2048, 64, 1), 0)
        del buf305
        triton_poi_fused_avg_pool2d_109.run(buf306, buf307, 2, 2048, grid=grid(2, 2048), stream=stream0)
        del buf306
        buf324 = reinterpret_tensor(buf330, (32, 64, 2, 1), (2816, 44, 22, 1), 16)
        triton_poi_fused_stack_111.run(buf307, buf324, 2048, 2, grid=grid(2048, 2), stream=stream0)
        buf325 = reinterpret_tensor(buf330, (32, 64, 2, 1), (2816, 44, 22, 1), 17)
        triton_poi_fused_stack_112.run(buf274, buf325, 32, 128, grid=grid(32, 128), stream=stream0)
        buf326 = reinterpret_tensor(buf330, (32, 64, 2, 1), (2816, 44, 22, 1), 18)
        triton_poi_fused_stack_113.run(buf326, 4096, grid=grid(4096), stream=stream0)
        buf327 = reinterpret_tensor(buf330, (32, 64, 2, 1), (2816, 44, 22, 1), 19)
        triton_poi_fused_stack_114.run(buf327, 4096, grid=grid(4096), stream=stream0)
        buf328 = reinterpret_tensor(buf330, (32, 64, 2, 1), (2816, 44, 22, 1), 20)
        triton_poi_fused_stack_115.run(arg172_1, buf328, 4096, grid=grid(4096), stream=stream0)
        del arg172_1
        buf329 = reinterpret_tensor(buf330, (32, 64, 2, 1), (2816, 44, 22, 1), 21)
        triton_poi_fused_stack_116.run(buf329, 4096, grid=grid(4096), stream=stream0)
        buf331 = empty((32, 64, 2, 22), device='cuda', dtype=torch.float16)
        triton_poi_fused__to_copy_clamp_117.run(buf330, buf331, 90112, grid=grid(90112), stream=stream0)
        del buf308
        del buf309
        del buf310
        del buf311
        del buf312
        del buf313
        del buf314
        del buf315
        del buf316
        del buf317
        del buf318
        del buf319
        del buf320
        del buf321
        del buf322
        del buf323
        del buf324
        del buf325
        del buf326
        del buf327
        del buf328
        del buf329
        del buf330
        buf332 = empty((4096, 64), device='cuda', dtype=torch.float16)
        extern_kernels.mm(reinterpret_tensor(buf331, (4096, 22), (22, 1), 0), reinterpret_tensor(arg6_1, (22, 64), (1, 22), 0), out=buf332)
        del arg6_1
        del buf331
        buf333 = reinterpret_tensor(buf332, (32, 64, 2, 64), (8192, 128, 64, 1), 0)
        del buf332
        triton_poi_fused__to_copy_silu_118.run(buf333, arg7_1, 262144, grid=grid(262144), stream=stream0)
        del arg7_1
        buf334 = empty((4096, 64), device='cuda', dtype=torch.float16)
        extern_kernels.mm(reinterpret_tensor(buf333, (4096, 64), (64, 1), 0), reinterpret_tensor(arg8_1, (64, 64), (1, 64), 0), out=buf334)
        del arg8_1
        del buf333
        buf335 = reinterpret_tensor(buf334, (32, 64, 2, 64), (8192, 128, 64, 1), 0)
        del buf334
        triton_poi_fused__to_copy_silu_118.run(buf335, arg9_1, 262144, grid=grid(262144), stream=stream0)
        del arg9_1
        buf336 = reinterpret_tensor(buf249, (4096, 2), (2, 1), 0)
        del buf249
        extern_kernels.mm(reinterpret_tensor(buf335, (4096, 64), (64, 1), 0), reinterpret_tensor(arg10_1, (64, 2), (1, 64), 0), out=buf336)
        del arg10_1
        del buf335
        buf405 = buf164
        del buf164
        buf407 = empty((20, 32, 64, 2), device='cuda', dtype=torch.float32)
        triton_poi_fused_add_mul_sub_119.run(buf162, arg154_1, arg153_1, buf62, buf161, buf60, buf203, arg27_1, buf266, buf405, buf407, 40960, 2, grid=grid(40960, 2), stream=stream0)
        buf406 = buf68
        del buf68
        buf408 = empty((1, 20, 32, 64, 3), device='cuda', dtype=torch.float32)
        triton_poi_fused_copy_div_select_scatter_slice_scatter_sub_120.run(buf405, buf161, arg154_1, arg153_1, buf60, buf407, buf162, buf62, buf406, buf408, 122880, grid=grid(122880), stream=stream0)
        del buf405
        buf337 = buf407
        del buf407
        buf409 = buf161
        del buf161
        buf410 = buf162
        del buf162
        triton_poi_fused_add_div_exp_mean_mul_sub_tanh_121.run(buf409, buf410, buf271, buf336, arg11_1, arg155_1, buf84, arg154_1, arg153_1, buf406, buf408, buf60, buf62, buf337, 640, 128, grid=grid(640, 128), stream=stream0)
        del arg11_1
        del buf336
        buf401 = reinterpret_tensor(buf271, (32, 64, 2), (128, 2, 1), 0)
        del buf271
        buf403 = reinterpret_tensor(buf274, (32, 64, 2), (128, 2, 1), 0)
        del buf274
        triton_per_fused_add_mean_mul_var_122.run(buf337, buf83, arg153_1, arg154_1, buf401, buf403, 4096, 20, grid=grid(4096), stream=stream0)
        buf411 = reinterpret_tensor(buf307, (32, 64, 2), (128, 2, 1), 0)
        del buf307
        triton_per_fused_mean_123.run(buf410, buf411, 4096, 20, grid=grid(4096), stream=stream0)
        buf338 = reinterpret_tensor(buf409, (1, 20, 2, 32, 64), (81920, 4096, 1, 128, 2), 0)
        del buf409
        buf412 = buf410
        del buf410
        triton_poi_fused_add_div_mean_mul_sub_124.run(buf412, buf337, buf83, arg153_1, arg154_1, arg155_1, buf403, buf411, buf338, 81920, grid=grid(81920), stream=stream0)
        buf339 = reinterpret_tensor(buf292, (1, 2, 32, 64), (4096, 1, 128, 2), 0)
        del buf292
        buf344 = reinterpret_tensor(buf290, (1, 2, 32, 64), (4096, 1, 128, 2), 0)
        del buf290
        triton_per_fused_mean_std_125.run(buf338, buf339, buf344, 4096, 20, grid=grid(4096), stream=stream0)
        buf341 = reinterpret_tensor(buf288, (1, 2, 32, 64), (4096, 2048, 64, 1), 0)
        del buf288
        buf389 = reinterpret_tensor(buf286, (1, 1, 2, 32, 64), (4096, 4096, 2048, 64, 1), 0)
        del buf286
        buf391 = reinterpret_tensor(buf283, (1, 1, 2, 32, 64), (4096, 4096, 2048, 64, 1), 0)
        del buf283
        buf394 = reinterpret_tensor(buf280, (2, 32, 64), (2048, 64, 1), 0)
        del buf280
        buf396 = reinterpret_tensor(buf276, (2, 32, 64), (2048, 64, 1), 0)
        del buf276
        triton_per_fused_mean_mul_std_var_126.run(buf180, arg155_1, buf341, buf389, buf391, buf394, buf396, 4096, 20, grid=grid(4096), stream=stream0)
        buf347 = empty_strided((1, 333, 32, 64), (681984, 1, 21312, 333), device='cuda', dtype=torch.float16)
        triton_poi_fused__to_copy_cat_127.run(buf180, buf185, buf339, buf344, buf341, buf347, 333, 2048, grid=grid(333, 2048), stream=stream0)
        del buf339
        del buf341
        del buf344
        buf348 = extern_kernels.convolution(buf347, arg60_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf348, (1, 64, 32, 64), (131072, 1, 4096, 64))
        del arg60_1
        del buf347
        buf349 = empty_strided((1, 64, 32, 64), (131072, 1, 4096, 64), device='cuda', dtype=torch.float32)
        triton_poi_fused__to_copy_add_convolution_mul_tanh_128.run(buf348, arg61_1, buf270, arg65_1, buf349, 131072, grid=grid(131072), stream=stream0)
        del arg61_1
        del arg65_1
        del buf270
        del buf348
        buf350 = empty_strided((1, 64, 36, 68), (156672, 1, 4352, 64), device='cuda', dtype=torch.float32)
        triton_poi_fused_replication_pad2d_129.run(buf349, buf350, 156672, grid=grid(156672), stream=stream0)
        buf351 = empty_strided((1, 64, 36, 68), (156672, 1, 4352, 64), device='cuda', dtype=torch.float16)
        triton_poi_fused_replication_pad2d_130.run(buf350, buf351, 64, 2448, grid=grid(64, 2448), stream=stream0)
        del buf350
        buf352 = extern_kernels.convolution(buf351, arg66_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf352, (1, 64, 32, 64), (131072, 1, 4096, 64))
        del arg66_1
        del buf351
        buf353 = buf352
        del buf352
        triton_poi_fused__to_copy_convolution_replication_pad2d_silu_131.run(buf353, arg67_1, 131072, grid=grid(131072), stream=stream0)
        del arg67_1
        buf354 = extern_kernels.convolution(buf353, arg68_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf354, (1, 64, 32, 64), (131072, 1, 4096, 64))
        del arg68_1
        del buf353
        buf358 = empty((1, 64, 40, 72), device='cuda', dtype=torch.float32)
        buf355 = reinterpret_tensor(buf358, (1, 64, 4, 72), (184320, 2880, 72, 1), 0)
        buf357 = reinterpret_tensor(buf358, (1, 64, 4, 72), (184320, 2880, 72, 1), 2592)
        triton_poi_fused_replication_pad2d_132.run(buf349, buf354, arg69_1, buf355, buf357, 64, 288, grid=grid(64, 288), stream=stream0)
        buf356 = reinterpret_tensor(buf358, (1, 64, 32, 72), (184320, 2880, 72, 1), 288)
        triton_poi_fused_replication_pad2d_133.run(buf349, buf354, arg69_1, buf356, 2048, 72, grid=grid(2048, 72), stream=stream0)
        buf359 = empty_strided((1, 64, 40, 72), (184320, 1, 4608, 64), device='cuda', dtype=torch.float16)
        triton_poi_fused_replication_pad2d_134.run(buf358, buf359, 64, 2880, grid=grid(64, 2880), stream=stream0)
        del buf355
        del buf356
        del buf357
        del buf358
        buf360 = extern_kernels.convolution(buf359, arg70_1, stride=(1, 1), padding=(0, 0), dilation=(2, 2), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf360, (1, 64, 32, 64), (131072, 1, 4096, 64))
        del arg70_1
        del buf359
        buf361 = buf360
        del buf360
        triton_poi_fused__to_copy_convolution_replication_pad2d_silu_131.run(buf361, arg71_1, 131072, grid=grid(131072), stream=stream0)
        del arg71_1
        buf362 = extern_kernels.convolution(buf361, arg72_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf362, (1, 64, 32, 64), (131072, 1, 4096, 64))
        del arg72_1
        del buf361
        buf363 = empty_strided((1, 64, 8, 16), (8192, 1, 1024, 64), device='cuda', dtype=torch.float32)
        triton_poi_fused_avg_pool2d_silu_135.run(buf349, buf363, 8192, grid=grid(8192), stream=stream0)
        buf364 = empty_strided((1, 64, 12, 20), (15360, 1, 1280, 64), device='cuda', dtype=torch.float32)
        triton_poi_fused_replication_pad2d_136.run(buf363, buf364, 15360, grid=grid(15360), stream=stream0)
        buf365 = empty_strided((1, 64, 12, 20), (15360, 1, 1280, 64), device='cuda', dtype=torch.float16)
        triton_poi_fused_replication_pad2d_137.run(buf364, buf365, 64, 240, grid=grid(64, 240), stream=stream0)
        del buf364
        buf366 = extern_kernels.convolution(buf365, arg74_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf366, (1, 64, 8, 16), (8192, 1, 1024, 64))
        del arg74_1
        del buf365
        buf367 = buf366
        del buf366
        triton_poi_fused__to_copy_convolution_replication_pad2d_silu_138.run(buf367, arg75_1, 8192, grid=grid(8192), stream=stream0)
        del arg75_1
        buf368 = extern_kernels.convolution(buf367, arg76_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf368, (1, 64, 8, 16), (8192, 1, 1024, 64))
        del arg76_1
        del buf367
        buf372 = empty((1, 64, 16, 24), device='cuda', dtype=torch.float32)
        buf369 = reinterpret_tensor(buf372, (1, 64, 4, 24), (24576, 384, 24, 1), 0)
        buf371 = reinterpret_tensor(buf372, (1, 64, 4, 24), (24576, 384, 24, 1), 288)
        triton_poi_fused_replication_pad2d_139.run(buf363, buf368, arg77_1, buf369, buf371, 64, 96, grid=grid(64, 96), stream=stream0)
        buf370 = reinterpret_tensor(buf372, (1, 64, 8, 24), (24576, 384, 24, 1), 96)
        triton_poi_fused_replication_pad2d_140.run(buf363, buf368, arg77_1, buf370, 64, 192, grid=grid(64, 192), stream=stream0)
        buf373 = empty_strided((1, 64, 16, 24), (24576, 1, 1536, 64), device='cuda', dtype=torch.float16)
        triton_poi_fused_replication_pad2d_141.run(buf372, buf373, 64, 384, grid=grid(64, 384), stream=stream0)
        del buf369
        del buf370
        del buf371
        del buf372
        buf374 = extern_kernels.convolution(buf373, arg78_1, stride=(1, 1), padding=(0, 0), dilation=(2, 2), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf374, (1, 64, 8, 16), (8192, 1, 1024, 64))
        del arg78_1
        del buf373
        buf375 = buf374
        del buf374
        triton_poi_fused__to_copy_convolution_replication_pad2d_silu_138.run(buf375, arg79_1, 8192, grid=grid(8192), stream=stream0)
        del arg79_1
        buf376 = extern_kernels.convolution(buf375, arg80_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf376, (1, 64, 8, 16), (8192, 1, 1024, 64))
        del arg80_1
        del buf375
        buf377 = buf363
        del buf363
        triton_poi_fused__to_copy_add_convolution_replication_pad2d_silu_142.run(buf377, buf368, arg77_1, buf376, arg81_1, 8192, grid=grid(8192), stream=stream0)
        del arg77_1
        del arg81_1
        del buf368
        del buf376
        buf378 = empty((1, 64, 32, 64), device='cuda', dtype=torch.float32)
        buf382 = buf378
        del buf378
        triton_poi_fused__to_copy__unsafe_index_add_arange_clamp_mul_rsub_sub_143.run(buf382, buf377, 131072, grid=grid(131072), stream=stream0)
        buf383 = buf354
        del buf354
        triton_poi_fused__to_copy_add_convolution_replication_pad2d_silu_144.run(buf383, buf349, arg69_1, buf362, arg73_1, buf382, 2048, 64, grid=grid(2048, 64), stream=stream0)
        del arg69_1
        del arg73_1
        del buf349
        del buf362
        del buf382
        buf384 = extern_kernels.convolution(buf383, arg82_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf384, (1, 4, 32, 64), (8192, 1, 256, 4))
        del arg82_1
        del buf383
        buf386 = empty((1, 64), device='cuda', dtype=torch.float16)
        extern_kernels.mm(buf385, reinterpret_tensor(arg90_1, (256, 64), (1, 256), 0), out=buf386)
        del arg90_1
        del buf385
        buf387 = buf386
        del buf386
        triton_poi_fused__to_copy_silu_145.run(buf387, arg91_1, 64, grid=grid(64), stream=stream0)
        del arg91_1
        buf388 = empty((1, 64), device='cuda', dtype=torch.float16)
        extern_kernels.mm(buf387, reinterpret_tensor(arg92_1, (64, 64), (1, 64), 0), out=buf388)
        del arg92_1
        del buf387
        buf393 = buf185
        del buf185
        triton_poi_fused_cat_69.run(buf180, buf389, buf391, buf393, 327680, grid=grid(327680), stream=stream0)
        buf398 = reinterpret_tensor(buf377, (2, 2, 32, 64), (4096, 1, 128, 2), 0)
        del buf377
        triton_poi_fused_stack_146.run(buf394, buf396, buf398, 4, 2048, grid=grid(4, 2048), stream=stream0)
        buf439 = empty((), device='cuda', dtype=torch.float32)
        triton_poi_fused_add_147.run(arg228_1, buf439, 1, grid=grid(1), stream=stream0)
        del arg228_1
        buf399 = empty_strided((2, 2, 32, 64), (4096, 1, 128, 2), device='cuda', dtype=torch.float32)
        buf433 = empty_strided((2, 2, 32, 64), (4096, 1, 128, 2), device='cuda', dtype=torch.float32)
        buf435 = empty_strided((2, 2, 32, 64), (4096, 1, 128, 2), device='cuda', dtype=torch.float32)
        buf440 = empty_strided((2, 2, 32, 64), (4096, 1, 128, 2), device='cuda', dtype=torch.float32)
        triton_poi_fused_add_div_mul_sub_148.run(arg219_1, buf398, arg220_1, arg221_1, arg218_1, buf439, buf399, buf433, buf435, buf440, 4096, 2, grid=grid(4096, 2), stream=stream0)
        del arg218_1
        del arg219_1
        del arg220_1
        del arg221_1
        buf413 = reinterpret_tensor(buf396, (32, 64, 2), (128, 2, 1), 0)
        del buf396
        buf415 = reinterpret_tensor(buf394, (32, 64, 2), (128, 2, 1), 0)
        del buf394
        triton_per_fused_mean_sub_var_149.run(buf412, buf413, buf415, 4096, 20, grid=grid(4096), stream=stream0)
        buf417 = reinterpret_tensor(buf338, (20, 32, 64, 2), (4096, 128, 2, 1), 0)
        del buf338
        triton_poi_fused_add_bitwise_and_clamp_min_div_gt_mean_mul_sqrt_sub_var_where_150.run(buf401, buf415, buf412, buf413, buf337, buf83, arg153_1, arg154_1, buf403, buf417, 81920, grid=grid(81920), stream=stream0)
        buf418 = buf415
        del buf415
        triton_per_fused_add_mean_mul_151.run(buf403, buf417, buf411, buf418, 4096, 20, grid=grid(4096), stream=stream0)
        buf419 = buf417
        del buf417
        triton_poi_fused_add_exp_mean_mul_sub_152.run(buf419, buf418, buf384, arg83_1, arg155_1, buf403, buf411, 81920, grid=grid(81920), stream=stream0)
        del arg83_1
        buf420 = buf62
        del buf62
        triton_poi_fused_copy_div_select_scatter_slice_scatter_sub_153.run(buf419, arg154_1, arg153_1, buf337, buf83, buf420, 122880, grid=grid(122880), stream=stream0)
        buf421 = reinterpret_tensor(buf418, (2, 32, 64), (1, 128, 2), 0)
        del buf418
        buf424 = reinterpret_tensor(buf411, (2, 32, 64), (1, 128, 2), 0)
        del buf411
        buf452 = reinterpret_tensor(buf403, (1, 2, 32, 64), (4096, 1, 128, 2), 0)
        del buf403
        buf454 = reinterpret_tensor(buf413, (1, 2, 32, 64), (4096, 1, 128, 2), 0)
        del buf413
        buf427 = reinterpret_tensor(buf401, (2, 32, 64), (1, 128, 2), 0)
        del buf401
        buf429 = reinterpret_tensor(buf391, (2, 32, 64), (1, 128, 2), 0)
        del buf391
        triton_per_fused_mean_var_154.run(buf420, arg153_1, arg154_1, arg174_1, arg152_1, buf421, buf424, buf452, buf454, buf427, buf429, 4096, 20, grid=grid(4096), stream=stream0)
        buf422 = reinterpret_tensor(buf389, (2, 32, 64), (2048, 64, 1), 0)
        del buf389
        triton_poi_fused_cat_64.run(buf421, buf422, 2, 2048, grid=grid(2, 2048), stream=stream0)
        buf426 = reinterpret_tensor(buf421, (2, 32, 64), (2048, 64, 1), 0)
        del buf421
        triton_poi_fused_cat_64.run(buf424, buf426, 2, 2048, grid=grid(2, 2048), stream=stream0)
        buf437 = empty_strided((2, 2, 32, 64), (4096, 1, 128, 2), device='cuda', dtype=torch.float32)
        triton_poi_fused_clone_155.run(arg227_1, buf437, 4, 2048, grid=grid(4, 2048), stream=stream0)
        del arg227_1
        buf438 = empty_strided((2, 2, 32, 64), (4096, 1, 128, 2), device='cuda', dtype=torch.float32)
        triton_poi_fused_clone_155.run(arg226_1, buf438, 4, 2048, grid=grid(4, 2048), stream=stream0)
        del arg226_1
        buf432 = empty_strided((2, 2, 32, 64), (4096, 1, 128, 2), device='cuda', dtype=torch.float32)
        buf434 = empty_strided((2, 2, 32, 64), (4096, 1, 128, 2), device='cuda', dtype=torch.float32)
        buf436 = empty_strided((2, 2, 32, 64), (4096, 1, 128, 2), device='cuda', dtype=torch.float32)
        buf451 = reinterpret_tensor(buf419, (40, 32, 64), (2048, 64, 1), 0)
        del buf419
        buf441 = reinterpret_tensor(buf451, (4, 32, 64), (2048, 64, 1), 0)
        buf443 = reinterpret_tensor(buf451, (4, 32, 64), (2048, 64, 1), 16384)
        buf445 = reinterpret_tensor(buf451, (4, 32, 64), (2048, 64, 1), 32768)
        buf447 = reinterpret_tensor(buf451, (4, 32, 64), (2048, 64, 1), 49152)
        buf448 = reinterpret_tensor(buf451, (4, 32, 64), (2048, 64, 1), 57344)
        buf449 = reinterpret_tensor(buf451, (4, 32, 64), (2048, 64, 1), 65536)
        buf450 = reinterpret_tensor(buf451, (4, 32, 64), (2048, 64, 1), 73728)
        triton_poi_fused_add_cat_mul_stack_sub_156.run(buf427, arg216_1, arg155_1, buf429, arg217_1, arg222_1, arg223_1, arg224_1, buf399, buf422, buf426, buf433, buf435, buf437, buf438, arg225_1, buf440, buf432, buf434, buf436, buf441, buf443, buf445, buf447, buf448, buf449, buf450, 4, 2048, grid=grid(4, 2048), stream=stream0)
        del arg216_1
        del arg217_1
        del arg222_1
        del arg223_1
        del arg224_1
        del arg225_1
        buf442 = reinterpret_tensor(buf451, (4, 32, 64), (2048, 64, 1), 8192)
        triton_poi_fused_cat_157.run(buf432, buf442, 4, 2048, grid=grid(4, 2048), stream=stream0)
        buf444 = reinterpret_tensor(buf451, (4, 32, 64), (2048, 64, 1), 24576)
        triton_poi_fused_cat_157.run(buf434, buf444, 4, 2048, grid=grid(4, 2048), stream=stream0)
        buf446 = reinterpret_tensor(buf451, (4, 32, 64), (2048, 64, 1), 40960)
        triton_poi_fused_cat_157.run(buf436, buf446, 4, 2048, grid=grid(4, 2048), stream=stream0)
        buf465 = empty((1, 48, 32, 64), device='cuda', dtype=torch.float32)
        buf463 = reinterpret_tensor(buf465, (1, 2, 32, 64), (98304, 2048, 64, 1), 90112)
        buf464 = reinterpret_tensor(buf465, (1, 2, 32, 64), (98304, 2048, 64, 1), 94208)
        triton_per_fused_mean_std_158.run(buf180, buf463, buf464, 4096, 20, grid=grid(4096), stream=stream0)
        del buf441
        del buf442
        del buf443
        del buf444
        del buf445
        del buf446
        del buf447
        del buf448
        del buf449
        del buf450
        buf460 = reinterpret_tensor(buf465, (1, 40, 32, 64), (98304, 2048, 64, 1), 0)
        triton_poi_fused_cat_159.run(buf451, buf460, 81920, grid=grid(81920), stream=stream0)
        buf461 = reinterpret_tensor(buf465, (1, 2, 32, 64), (98304, 2048, 64, 1), 81920)
        buf462 = reinterpret_tensor(buf465, (1, 2, 32, 64), (98304, 2048, 64, 1), 86016)
        triton_poi_fused_clamp_min_div_mean_sqrt_var_160.run(buf452, arg155_1, buf454, buf461, buf462, 2, 2048, grid=grid(2, 2048), stream=stream0)
        buf466 = empty_strided((1, 375, 32, 64), (768000, 1, 24000, 375), device='cuda', dtype=torch.float16)
        triton_poi_fused__to_copy_cat_161.run(buf180, buf393, buf465, buf466, 375, 2048, grid=grid(375, 2048), stream=stream0)
        del buf393
        del buf460
        del buf461
        del buf462
        del buf463
        del buf464
        del buf465
        buf467 = extern_kernels.convolution(buf466, arg84_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf467, (1, 32, 32, 64), (65536, 1, 2048, 32))
        del arg84_1
        del buf466
        buf468 = empty_strided((1, 40, 32, 64), (81920, 1, 2560, 40), device='cuda', dtype=torch.float16)
        triton_poi_fused__to_copy_162.run(buf420, arg153_1, arg154_1, buf452, buf454, buf468, 81920, grid=grid(81920), stream=stream0)
        buf469 = extern_kernels.convolution(buf468, arg86_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf469, (1, 32, 32, 64), (65536, 1, 2048, 32))
        del arg86_1
        del buf468
        buf470 = buf203
        del buf203
        triton_poi_fused__to_copy_cat_163.run(buf470, arg27_1, buf266, 2048, 80, grid=grid(2048, 80), stream=stream0)
        del arg27_1
        del buf266
        buf471 = extern_kernels.convolution(buf470, arg88_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf471, (1, 32, 32, 64), (65536, 1, 2048, 32))
        del arg88_1
        del buf470
        buf472 = empty_strided((1, 32, 32, 64), (65536, 1, 2048, 32), device='cuda', dtype=torch.float32)
        buf473 = empty_strided((1, 32, 32, 64), (65536, 1, 2048, 32), device='cuda', dtype=torch.float16)
        triton_poi_fused__to_copy_add_cat_convolution_mul_silu_tanh_164.run(buf467, arg85_1, buf469, arg87_1, buf471, arg89_1, buf388, arg93_1, buf472, buf473, 65536, grid=grid(65536), stream=stream0)
        del arg85_1
        del arg87_1
        del arg89_1
        del arg93_1
        del buf388
        del buf467
        del buf469
        del buf471
        buf474 = empty_strided((32, 32, 3, 3), (288, 1, 96, 32), device='cuda', dtype=torch.float16)
        triton_poi_fused__to_copy_convolution_silu_165.run(arg94_1, buf474, 1024, 9, grid=grid(1024, 9), stream=stream0)
        del arg94_1
        buf475 = extern_kernels.convolution(buf473, buf474, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf475, (1, 32, 32, 64), (65536, 1, 2048, 32))
        del buf473
        buf476 = buf475
        del buf475
        triton_poi_fused__to_copy_convolution_silu_166.run(buf476, arg95_1, 65536, grid=grid(65536), stream=stream0)
        del arg95_1
        buf477 = buf474
        del buf474
        triton_poi_fused__to_copy_convolution_silu_165.run(arg96_1, buf477, 1024, 9, grid=grid(1024, 9), stream=stream0)
        del arg96_1
        buf478 = extern_kernels.convolution(buf476, buf477, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf478, (1, 32, 32, 64), (65536, 1, 2048, 32))
        buf479 = buf472
        del buf472
        triton_poi_fused__to_copy_add_convolution_mul_silu_167.run(buf479, buf478, arg97_1, 65536, grid=grid(65536), stream=stream0)
        del arg97_1
        buf480 = empty_strided((1, 32, 16, 32), (16384, 1, 1024, 32), device='cuda', dtype=torch.float16)
        triton_poi_fused__to_copy_avg_pool2d_168.run(buf479, buf480, 16384, grid=grid(16384), stream=stream0)
        buf481 = extern_kernels.convolution(buf480, arg98_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf481, (1, 64, 16, 32), (32768, 1, 2048, 64))
        del arg98_1
        del buf480
        buf482 = buf260
        del buf260
        triton_poi_fused__to_copy_avg_pool2d_convolution_silu_169.run(buf481, arg99_1, buf482, 32768, grid=grid(32768), stream=stream0)
        buf483 = empty_strided((64, 64, 3, 3), (576, 1, 192, 64), device='cuda', dtype=torch.float16)
        triton_poi_fused__to_copy_avg_pool2d_convolution_silu_170.run(arg100_1, buf483, 4096, 9, grid=grid(4096, 9), stream=stream0)
        del arg100_1
        buf484 = extern_kernels.convolution(buf482, buf483, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf484, (1, 64, 16, 32), (32768, 1, 2048, 64))
        del buf482
        buf485 = buf484
        del buf484
        triton_poi_fused__to_copy_convolution_replication_pad2d_silu_83.run(buf485, arg101_1, 32768, grid=grid(32768), stream=stream0)
        del arg101_1
        buf486 = buf483
        del buf483
        triton_poi_fused__to_copy_avg_pool2d_convolution_silu_170.run(arg102_1, buf486, 4096, 9, grid=grid(4096, 9), stream=stream0)
        del arg102_1
        buf487 = extern_kernels.convolution(buf485, buf486, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf487, (1, 64, 16, 32), (32768, 1, 2048, 64))
        del buf485
        buf488 = reinterpret_tensor(buf254, (1, 64, 16, 32), (32768, 1, 2048, 64), 0)
        del buf254
        triton_poi_fused__to_copy_add_avg_pool2d_convolution_mul_silu_171.run(buf481, arg99_1, buf487, arg103_1, buf488, 32768, grid=grid(32768), stream=stream0)
        del arg103_1
        del arg99_1
        del buf481
        buf489 = reinterpret_tensor(buf384, (1, 64, 8, 16), (8192, 1, 1024, 64), 0)
        del buf384
        triton_poi_fused__to_copy_avg_pool2d_85.run(buf488, buf489, 8192, grid=grid(8192), stream=stream0)
        buf490 = extern_kernels.convolution(buf489, arg104_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf490, (1, 96, 8, 16), (12288, 1, 1536, 96))
        del arg104_1
        del buf489
        buf491 = buf248
        del buf248
        triton_poi_fused__to_copy_avg_pool2d_convolution_silu_172.run(buf490, arg105_1, buf491, 12288, grid=grid(12288), stream=stream0)
        buf492 = empty_strided((96, 96, 3, 3), (864, 1, 288, 96), device='cuda', dtype=torch.float16)
        triton_poi_fused__to_copy_avg_pool2d_convolution_silu_173.run(arg106_1, buf492, 9216, 9, grid=grid(9216, 9), stream=stream0)
        del arg106_1
        buf493 = extern_kernels.convolution(buf491, buf492, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf493, (1, 96, 8, 16), (12288, 1, 1536, 96))
        del buf491
        buf494 = buf493
        del buf493
        triton_poi_fused__to_copy_convolution_replication_pad2d_silu_89.run(buf494, arg107_1, 12288, grid=grid(12288), stream=stream0)
        del arg107_1
        buf495 = buf492
        del buf492
        triton_poi_fused__to_copy_avg_pool2d_convolution_silu_173.run(arg108_1, buf495, 9216, 9, grid=grid(9216, 9), stream=stream0)
        del arg108_1
        buf496 = extern_kernels.convolution(buf494, buf495, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf496, (1, 96, 8, 16), (12288, 1, 1536, 96))
        buf497 = buf494
        del buf494
        triton_poi_fused__to_copy_add_avg_pool2d_convolution_mul_silu_174.run(buf490, arg105_1, buf496, arg109_1, buf497, 12288, grid=grid(12288), stream=stream0)
        buf498 = buf495
        del buf495
        triton_poi_fused__to_copy_avg_pool2d_convolution_silu_173.run(arg110_1, buf498, 9216, 9, grid=grid(9216, 9), stream=stream0)
        del arg110_1
        buf499 = extern_kernels.convolution(buf497, buf498, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf499, (1, 96, 8, 16), (12288, 1, 1536, 96))
        del buf497
        buf500 = buf499
        del buf499
        triton_poi_fused__to_copy_convolution_replication_pad2d_silu_89.run(buf500, arg111_1, 12288, grid=grid(12288), stream=stream0)
        del arg111_1
        buf501 = buf498
        del buf498
        triton_poi_fused__to_copy_avg_pool2d_convolution_silu_173.run(arg112_1, buf501, 9216, 9, grid=grid(9216, 9), stream=stream0)
        del arg112_1
        buf502 = extern_kernels.convolution(buf500, buf501, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf502, (1, 96, 8, 16), (12288, 1, 1536, 96))
        del buf501
        buf504 = buf500
        del buf500
        triton_poi_fused__to_copy_add_avg_pool2d_convolution_mul_silu_175.run(buf490, arg105_1, buf496, arg109_1, buf502, arg113_1, buf504, 12288, grid=grid(12288), stream=stream0)
        del arg105_1
        del arg109_1
        del arg113_1
        del buf490
        del buf496
        del buf502
        buf505 = extern_kernels.convolution(buf504, arg114_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf505, (1, 64, 8, 16), (8192, 1, 1024, 64))
        del arg114_1
        buf506 = reinterpret_tensor(buf214, (1, 64, 16, 32), (32768, 512, 32, 1), 0)
        del buf214
        buf510 = buf506
        del buf506
        triton_poi_fused__to_copy__unsafe_index_add_arange_clamp_convolution_mul_replication_pad2d_rsub_silu_sub_100.run(buf510, buf505, arg115_1, 32768, grid=grid(32768), stream=stream0)
        del arg115_1
        del buf505
        buf511 = buf487
        del buf487
        triton_poi_fused__to_copy_add_176.run(buf488, buf510, buf511, 512, 64, grid=grid(512, 64), stream=stream0)
        buf512 = buf486
        del buf486
        triton_poi_fused__to_copy_avg_pool2d_convolution_silu_170.run(arg116_1, buf512, 4096, 9, grid=grid(4096, 9), stream=stream0)
        del arg116_1
        buf513 = extern_kernels.convolution(buf511, buf512, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf513, (1, 64, 16, 32), (32768, 1, 2048, 64))
        del buf511
        buf514 = buf513
        del buf513
        triton_poi_fused__to_copy_convolution_replication_pad2d_silu_83.run(buf514, arg117_1, 32768, grid=grid(32768), stream=stream0)
        del arg117_1
        buf515 = buf512
        del buf512
        triton_poi_fused__to_copy_avg_pool2d_convolution_silu_170.run(arg118_1, buf515, 4096, 9, grid=grid(4096, 9), stream=stream0)
        del arg118_1
        buf516 = extern_kernels.convolution(buf514, buf515, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf516, (1, 64, 16, 32), (32768, 1, 2048, 64))
        del buf514
        del buf515
        buf517 = buf516
        del buf516
        triton_poi_fused__to_copy_add_convolution_mul_silu_177.run(buf517, buf488, buf510, arg119_1, 512, 64, grid=grid(512, 64), stream=stream0)
        del arg119_1
        del buf488
        del buf510
        buf518 = extern_kernels.convolution(buf517, arg120_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf518, (1, 32, 16, 32), (16384, 1, 1024, 32))
        del arg120_1
        del buf517
        buf519 = empty((1, 32, 32, 64), device='cuda', dtype=torch.float32)
        buf523 = buf519
        del buf519
        triton_poi_fused__to_copy__unsafe_index_add_arange_clamp_convolution_mul_rsub_silu_sub_178.run(buf523, buf518, arg121_1, 65536, grid=grid(65536), stream=stream0)
        del arg121_1
        del buf518
        buf524 = buf478
        del buf478
        triton_poi_fused__to_copy_add_179.run(buf479, buf523, buf524, 2048, 32, grid=grid(2048, 32), stream=stream0)
        buf525 = buf477
        del buf477
        triton_poi_fused__to_copy_convolution_silu_165.run(arg122_1, buf525, 1024, 9, grid=grid(1024, 9), stream=stream0)
        del arg122_1
        buf526 = extern_kernels.convolution(buf524, buf525, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf526, (1, 32, 32, 64), (65536, 1, 2048, 32))
        buf527 = buf526
        del buf526
        triton_poi_fused__to_copy_convolution_silu_166.run(buf527, arg123_1, 65536, grid=grid(65536), stream=stream0)
        del arg123_1
        buf528 = buf525
        del buf525
        triton_poi_fused__to_copy_convolution_silu_165.run(arg124_1, buf528, 1024, 9, grid=grid(1024, 9), stream=stream0)
        del arg124_1
        buf529 = extern_kernels.convolution(buf527, buf528, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf529, (1, 32, 32, 64), (65536, 1, 2048, 32))
        del buf528
        buf530 = buf527
        del buf527
        buf532 = buf524
        del buf524
        buf534 = buf476
        del buf476
        triton_poi_fused__to_copy_add_convolution_mul_silu_180.run(buf479, buf523, buf529, arg125_1, buf530, buf532, buf534, 2048, 32, grid=grid(2048, 32), stream=stream0)
        del arg125_1
        del buf529
        buf531 = extern_kernels.convolution(buf530, arg130_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf531, (1, 80, 32, 64), (163840, 1, 5120, 80))
        del arg130_1
        del buf530
        buf533 = extern_kernels.convolution(buf532, arg126_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf533, (1, 3, 32, 64), (6144, 1, 192, 3))
        del arg126_1
        del buf532
        buf535 = extern_kernels.convolution(buf534, arg128_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf535, (1, 40, 32, 64), (81920, 1, 2560, 40))
        del arg128_1
        del buf534
        buf536 = reinterpret_tensor(buf337, (1, 20, 2, 32, 64), (81920, 4096, 1, 128, 2), 0)
        del buf337
        triton_poi_fused_add_mul_sub_tanh_181.run(buf420, arg153_1, arg154_1, buf452, buf454, buf535, arg129_1, buf536, 81920, grid=grid(81920), stream=stream0)
        del arg129_1
        del buf535
        buf537 = reinterpret_tensor(buf429, (1, 1, 2, 32, 64), (4096, 4096, 1, 128, 2), 0)
        del buf429
        buf539 = reinterpret_tensor(buf427, (1, 2, 32, 64), (4096, 1, 128, 2), 0)
        del buf427
        triton_per_fused_mean_sub_var_149.run(buf536, buf537, buf539, 4096, 20, grid=grid(4096), stream=stream0)
        buf541 = reinterpret_tensor(buf412, (1, 20, 2, 32, 64), (81920, 4096, 1, 128, 2), 0)
        del buf412
        triton_poi_fused_mean_mul_sub_where_182.run(buf454, buf539, buf536, buf537, buf420, arg153_1, arg154_1, buf452, buf541, 81920, grid=grid(81920), stream=stream0)
        buf542 = reinterpret_tensor(buf536, (1, 40, 32, 64), (81920, 2048, 64, 1), 0)
        del buf536
        triton_poi_fused_where_183.run(buf420, arg153_1, arg154_1, buf452, arg155_1, buf533, arg127_1, buf541, buf542, 2048, 40, grid=grid(2048, 40), stream=stream0)
        del arg127_1
        del buf533
        del buf541
        buf543 = reinterpret_tensor(buf452, (1, 2, 32, 64), (4096, 2048, 64, 1), 0)
        del buf452
        buf545 = reinterpret_tensor(buf539, (1, 2, 32, 64), (4096, 2048, 64, 1), 0)
        del buf539
        buf590 = empty((1, 78, 32, 64), device='cuda', dtype=torch.float32)
        buf572 = reinterpret_tensor(buf590, (1, 2, 32, 64), (159744, 2048, 64, 1), 81920)
        buf573 = reinterpret_tensor(buf590, (1, 2, 32, 64), (159744, 2048, 64, 1), 86016)
        triton_per_fused_clamp_min_div_mean_sqrt_var_184.run(buf542, arg154_1, arg153_1, buf420, arg155_1, buf543, buf545, buf572, buf573, 4096, 20, grid=grid(4096), stream=stream0)
        buf548 = reinterpret_tensor(buf537, (1, 2, 32, 64), (4096, 2048, 64, 1), 0)
        del buf537
        buf569 = reinterpret_tensor(buf454, (1, 2, 32, 64), (4096, 2048, 64, 1), 0)
        del buf454
        triton_per_fused_mean_var_185.run(buf180, buf548, buf569, 4096, 20, grid=grid(4096), stream=stream0)
        buf550 = reinterpret_tensor(buf590, (1, 2, 32, 64), (159744, 2048, 64, 1), 98304)
        triton_poi_fused_mean_sub_186.run(buf180, buf550, 4096, grid=grid(4096), stream=stream0)
        buf551 = empty((1, 1, 32, 64), device='cuda', dtype=torch.float32)
        triton_poi_fused_mul_pow_sum_var_187.run(buf548, arg155_1, buf551, 2048, grid=grid(2048), stream=stream0)
        del arg155_1
        buf552 = empty((1, 1, 32, 64), device='cuda', dtype=torch.float32)
        triton_poi_fused_sum_var_188.run(buf545, buf552, 2048, grid=grid(2048), stream=stream0)
        buf556 = reinterpret_tensor(buf424, (1, 2, 32, 64), (4096, 2048, 64, 1), 0)
        del buf424
        buf575 = reinterpret_tensor(buf590, (1, 2, 32, 64), (159744, 2048, 64, 1), 94208)
        buf579 = reinterpret_tensor(buf590, (1, 2, 32, 64), (159744, 2048, 64, 1), 114688)
        triton_poi_fused_avg_pool2d_clamp_min_sqrt_var_189.run(buf548, buf556, buf575, buf579, 4096, grid=grid(4096), stream=stream0)
        del buf548
        buf559 = aten.avg_pool2d(buf552, [7, 7], [1, 1], [3, 3], False, False, None)
        buf560 = buf559
        del buf559
        buf561 = aten.avg_pool2d(buf551, [7, 7], [1, 1], [3, 3], False, False, None)
        buf562 = buf561
        del buf561
        buf565 = aten.avg_pool2d(buf552, [15, 15], [1, 1], [7, 7], False, False, None)
        buf566 = buf565
        del buf565
        buf567 = aten.avg_pool2d(buf551, [15, 15], [1, 1], [7, 7], False, False, None)
        buf568 = buf567
        del buf567
        buf576 = reinterpret_tensor(buf590, (1, 1, 32, 64), (159744, 2048, 64, 1), 102400)
        buf577 = reinterpret_tensor(buf590, (1, 1, 32, 64), (159744, 2048, 64, 1), 104448)
        buf580 = reinterpret_tensor(buf590, (1, 1, 32, 64), (159744, 2048, 64, 1), 118784)
        buf581 = reinterpret_tensor(buf590, (1, 1, 32, 64), (159744, 2048, 64, 1), 120832)
        buf583 = reinterpret_tensor(buf590, (1, 1, 32, 64), (159744, 2048, 64, 1), 126976)
        buf584 = reinterpret_tensor(buf590, (1, 1, 32, 64), (159744, 2048, 64, 1), 129024)
        buf586 = reinterpret_tensor(buf590, (1, 1, 32, 64), (159744, 2048, 64, 1), 135168)
        buf587 = reinterpret_tensor(buf590, (1, 1, 32, 64), (159744, 2048, 64, 1), 137216)
        triton_poi_fused_add_avg_pool2d_div_log_190.run(buf551, buf552, buf560, buf562, buf566, buf568, buf576, buf577, buf580, buf581, buf583, buf584, buf586, buf587, 2048, grid=grid(2048), stream=stream0)
        del buf560
        del buf562
        del buf566
        del buf568
        buf557 = aten.avg_pool2d(buf556, [7, 7], [1, 1], [3, 3], False, False, None)
        buf558 = buf557
        del buf557
        buf563 = aten.avg_pool2d(buf556, [15, 15], [1, 1], [7, 7], False, False, None)
        del buf556
        buf564 = buf563
        del buf563
        buf574 = reinterpret_tensor(buf590, (1, 2, 32, 64), (159744, 2048, 64, 1), 90112)
        buf588 = reinterpret_tensor(buf590, (1, 2, 32, 64), (159744, 2048, 64, 1), 139264)
        triton_red_fused_cat_mean_191.run(buf569, buf574, buf588, 2, 2048, grid=grid(2), stream=stream0)
        buf571 = reinterpret_tensor(buf590, (1, 40, 32, 64), (159744, 2048, 64, 1), 0)
        triton_poi_fused_cat_159.run(buf451, buf571, 81920, grid=grid(81920), stream=stream0)
        buf578 = reinterpret_tensor(buf590, (1, 4, 32, 64), (159744, 2048, 64, 1), 106496)
        triton_poi_fused_cat_192.run(buf180, buf578, 8192, grid=grid(8192), stream=stream0)
        buf582 = reinterpret_tensor(buf590, (1, 2, 32, 64), (159744, 2048, 64, 1), 122880)
        triton_poi_fused_clamp_min_sqrt_193.run(buf558, buf582, 4096, grid=grid(4096), stream=stream0)
        del buf558
        buf585 = reinterpret_tensor(buf590, (1, 2, 32, 64), (159744, 2048, 64, 1), 131072)
        triton_poi_fused_clamp_min_sqrt_193.run(buf564, buf585, 4096, grid=grid(4096), stream=stream0)
        del buf564
        buf589 = reinterpret_tensor(buf590, (1, 8, 32, 64), (159744, 2048, 64, 1), 143360)
        triton_poi_fused_cat_194.run(buf180, buf569, buf589, 16384, grid=grid(16384), stream=stream0)
        del buf180
        buf591 = empty_strided((1, 78, 32, 64), (159744, 1, 4992, 78), device='cuda', dtype=torch.float16)
        triton_poi_fused__to_copy_clamp_195.run(buf590, buf591, 78, 2048, grid=grid(78, 2048), stream=stream0)
        del buf550
        del buf571
        del buf572
        del buf573
        del buf574
        del buf575
        del buf576
        del buf577
        del buf578
        del buf579
        del buf580
        del buf581
        del buf582
        del buf583
        del buf584
        del buf585
        del buf586
        del buf587
        del buf588
        del buf589
        del buf590
        buf592 = extern_kernels.convolution(buf591, arg132_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf592, (1, 32, 32, 64), (65536, 1, 2048, 32))
        del arg132_1
        del buf591
        buf593 = reinterpret_tensor(buf523, (1, 32, 32, 64), (65536, 1, 2048, 32), 0)
        del buf523
        triton_poi_fused__to_copy_clamp_convolution_silu_196.run(buf592, arg133_1, buf593, 65536, grid=grid(65536), stream=stream0)
        del arg133_1
        del buf592
        buf594 = empty_strided((1, 32, 36, 68), (78336, 1, 2176, 32), device='cuda', dtype=torch.float32)
        triton_poi_fused_replication_pad2d_197.run(buf593, buf594, 78336, grid=grid(78336), stream=stream0)
        buf595 = empty_strided((1, 32, 36, 68), (78336, 1, 2176, 32), device='cuda', dtype=torch.float16)
        triton_poi_fused_replication_pad2d_198.run(buf594, buf595, 32, 2448, grid=grid(32, 2448), stream=stream0)
        del buf594
        buf596 = extern_kernels.convolution(buf595, arg134_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf596, (1, 32, 32, 64), (65536, 1, 2048, 32))
        del arg134_1
        del buf595
        buf597 = buf596
        del buf596
        triton_poi_fused__to_copy_convolution_silu_166.run(buf597, arg135_1, 65536, grid=grid(65536), stream=stream0)
        del arg135_1
        buf598 = extern_kernels.convolution(buf597, arg136_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf598, (1, 32, 32, 64), (65536, 1, 2048, 32))
        del arg136_1
        del buf597
        buf602 = empty((1, 32, 40, 72), device='cuda', dtype=torch.float32)
        buf599 = reinterpret_tensor(buf602, (1, 32, 4, 72), (92160, 2880, 72, 1), 0)
        buf601 = reinterpret_tensor(buf602, (1, 32, 4, 72), (92160, 2880, 72, 1), 2592)
        triton_poi_fused_replication_pad2d_199.run(buf593, buf598, arg137_1, buf599, buf601, 32, 288, grid=grid(32, 288), stream=stream0)
        buf600 = reinterpret_tensor(buf602, (1, 32, 32, 72), (92160, 2880, 72, 1), 288)
        triton_poi_fused_replication_pad2d_200.run(buf593, buf598, arg137_1, buf600, 1024, 72, grid=grid(1024, 72), stream=stream0)
        buf603 = empty_strided((1, 32, 40, 72), (92160, 1, 2304, 32), device='cuda', dtype=torch.float16)
        triton_poi_fused_replication_pad2d_201.run(buf602, buf603, 32, 2880, grid=grid(32, 2880), stream=stream0)
        del buf599
        del buf600
        del buf601
        del buf602
        buf604 = extern_kernels.convolution(buf603, arg138_1, stride=(1, 1), padding=(0, 0), dilation=(2, 2), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf604, (1, 32, 32, 64), (65536, 1, 2048, 32))
        del arg138_1
        del buf603
        buf605 = buf604
        del buf604
        triton_poi_fused__to_copy_convolution_silu_166.run(buf605, arg139_1, 65536, grid=grid(65536), stream=stream0)
        del arg139_1
        buf606 = extern_kernels.convolution(buf605, arg140_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf606, (1, 32, 32, 64), (65536, 1, 2048, 32))
        del arg140_1
        del buf605
        buf607 = reinterpret_tensor(buf569, (1, 32, 8, 16), (4096, 1, 512, 32), 0)
        del buf569
        triton_poi_fused_avg_pool2d_202.run(buf593, buf607, 4096, grid=grid(4096), stream=stream0)
        buf608 = empty_strided((1, 32, 12, 20), (7680, 1, 640, 32), device='cuda', dtype=torch.float32)
        triton_poi_fused_replication_pad2d_203.run(buf607, buf608, 7680, grid=grid(7680), stream=stream0)
        buf609 = empty_strided((1, 32, 12, 20), (7680, 1, 640, 32), device='cuda', dtype=torch.float16)
        triton_poi_fused_replication_pad2d_204.run(buf608, buf609, 32, 240, grid=grid(32, 240), stream=stream0)
        del buf608
        buf610 = extern_kernels.convolution(buf609, arg142_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf610, (1, 32, 8, 16), (4096, 1, 512, 32))
        del arg142_1
        del buf609
        buf611 = buf610
        del buf610
        triton_poi_fused__to_copy_convolution_replication_pad2d_silu_205.run(buf611, arg143_1, 4096, grid=grid(4096), stream=stream0)
        del arg143_1
        buf612 = extern_kernels.convolution(buf611, arg144_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf612, (1, 32, 8, 16), (4096, 1, 512, 32))
        del arg144_1
        del buf611
        buf616 = reinterpret_tensor(buf242, (1, 32, 16, 24), (12288, 384, 24, 1), 0)
        del buf242
        buf613 = reinterpret_tensor(buf616, (1, 32, 4, 24), (12288, 384, 24, 1), 0)
        buf615 = reinterpret_tensor(buf616, (1, 32, 4, 24), (12288, 384, 24, 1), 288)
        triton_poi_fused_replication_pad2d_206.run(buf607, buf612, arg145_1, buf613, buf615, 32, 96, grid=grid(32, 96), stream=stream0)
        buf614 = reinterpret_tensor(buf616, (1, 32, 8, 24), (12288, 384, 24, 1), 96)
        triton_poi_fused_replication_pad2d_207.run(buf607, buf612, arg145_1, buf614, 32, 192, grid=grid(32, 192), stream=stream0)
        buf617 = reinterpret_tensor(buf504, (1, 32, 16, 24), (12288, 1, 768, 32), 0)
        del buf504
        triton_poi_fused_replication_pad2d_208.run(buf616, buf617, 32, 384, grid=grid(32, 384), stream=stream0)
        del buf613
        del buf614
        del buf615
        del buf616
        buf618 = extern_kernels.convolution(buf617, arg146_1, stride=(1, 1), padding=(0, 0), dilation=(2, 2), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf618, (1, 32, 8, 16), (4096, 1, 512, 32))
        del arg146_1
        del buf617
        buf619 = buf618
        del buf618
        triton_poi_fused__to_copy_convolution_replication_pad2d_silu_205.run(buf619, arg147_1, 4096, grid=grid(4096), stream=stream0)
        del arg147_1
        buf620 = extern_kernels.convolution(buf619, arg148_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf620, (1, 32, 8, 16), (4096, 1, 512, 32))
        del arg148_1
        del buf619
        buf621 = buf607
        del buf607
        triton_poi_fused__to_copy_add_convolution_replication_pad2d_silu_209.run(buf621, buf612, arg145_1, buf620, arg149_1, 4096, grid=grid(4096), stream=stream0)
        del arg145_1
        del arg149_1
        del buf612
        del buf620
        buf622 = reinterpret_tensor(buf479, (1, 32, 32, 64), (65536, 2048, 64, 1), 0)
        del buf479
        buf626 = buf622
        del buf622
        triton_poi_fused__to_copy__unsafe_index_add_arange_clamp_mul_rsub_sub_210.run(buf626, buf621, 65536, grid=grid(65536), stream=stream0)
        del buf621
        buf627 = buf598
        del buf598
        triton_poi_fused__to_copy_add_convolution_replication_pad2d_silu_211.run(buf627, buf593, arg137_1, buf606, arg141_1, buf626, 2048, 32, grid=grid(2048, 32), stream=stream0)
        del arg137_1
        del arg141_1
        del buf593
        del buf606
        del buf626
        buf628 = extern_kernels.convolution(buf627, arg150_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf628, (1, 1, 32, 64), (2048, 1, 64, 1))
        del arg150_1
        del buf627
        buf630 = buf552
        del buf552
        triton_poi_fused__to_copy_add_avg_pool2d_clamp_min_convolution_div_mul_replication_pad2d_silu_sum_tanh_var_212.run(buf630, buf551, buf545, buf628, arg151_1, 2048, grid=grid(2048), stream=stream0)
        del arg151_1
        del buf545
        del buf551
        del buf628
        buf631 = reinterpret_tensor(buf83, (20, 32, 64, 3), (6144, 192, 3, 1), 0)
        del buf83
        buf632 = reinterpret_tensor(buf631, (1, 20, 32, 64, 3), (122880, 6144, 192, 3, 1), 0)
        del buf631
        triton_poi_fused_copy_div_select_scatter_slice_scatter_sub_213.run(buf632, buf543, buf542, arg154_1, arg153_1, buf420, buf630, 40960, 3, grid=grid(40960, 3), stream=stream0)
        del buf543
        del buf630
        buf633 = reinterpret_tensor(buf542, (20, 32, 64, 2), (4096, 128, 2, 1), 0)
        del buf542
        buf636 = reinterpret_tensor(buf451, (20, 32, 64, 2), (4096, 128, 2, 1), 0)
        del buf451
        triton_poi_fused_add_mul_sub_214.run(buf408, buf406, buf531, arg131_1, buf633, buf636, 81920, grid=grid(81920), stream=stream0)
        del arg131_1
        del buf531
        buf634 = buf406
        del buf406
        buf635 = buf634
        del buf634
        triton_poi_fused_add_select_scatter_slice_scatter_215.run(buf635, buf633, 122880, grid=grid(122880), stream=stream0)
        del buf633
        buf637 = buf408
        del buf408
        buf638 = buf637
        del buf637
        triton_poi_fused_add_select_scatter_slice_scatter_215.run(buf638, buf636, 122880, grid=grid(122880), stream=stream0)
        del buf636
        buf639 = buf420
        del buf420
        buf641 = buf60
        del buf60
        triton_poi_fused_add_copy_mean_slice_scatter_216.run(arg175_1, buf21, buf37, buf38, buf639, buf641, 122880, grid=grid(122880), stream=stream0)
        del buf21
        del buf37
        del buf38
        buf642 = empty((1, 20, 32, 64, 2), device='cuda', dtype=torch.bool)
        buf646 = empty_strided((20, 32, 64, 2), (4096, 1, 64, 32), device='cuda', dtype=torch.bool)
        triton_poi_fused_bitwise_or_ne_217.run(arg183_1, arg174_1, arg156_1, arg206_1, arg152_1, arg153_1, arg154_1, buf642, buf646, 640, 128, grid=grid(640, 128), stream=stream0)
        del arg152_1
        del arg153_1
        del arg154_1
        del arg156_1
        del arg174_1
        del arg183_1
        del arg206_1
        return (buf632, buf635, buf638, arg175_1, buf639, buf640, buf641, buf46, buf50, buf55, buf642, buf65, buf52, buf643, buf72, buf81, buf644, buf3, buf17, buf19, buf0, buf4, buf41, buf44, buf51, buf63, buf97, buf99, buf100, buf101, buf645, buf128, buf129, buf646, buf84, buf299, buf302, buf300, buf303, buf163, buf165, buf169, buf647, buf422, buf426, buf440, buf399, buf433, buf435, buf432, buf434, buf436, buf438, buf437, buf398, buf439)
