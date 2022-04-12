# #%%
# import math


# def determine_padding(k, s, d, i, o):
#     total_padding = d * k + (o - 1) * s - d - i + 1
#     start_padding = total_padding // 2
#     final_padding = total_padding - start_padding
#     return (start_padding, final_padding)


# determine_padding(3, 2, 2, 225, 113)

# #%%


# class ConvArithmetic:
#     """ Convolution Arithmetic

#     This class is an implementation of the following paper and adapted for dilated convolution:
    
#         Dumoulin, V. & Visin, F. A guide to convolution arithmetic for deep learning. arXiv:1603.07285 [cs, stat] (2016).
#     """

#     @staticmethod
#     def size(base_size, down_scale):
#         return base_size // down_scale + base_size % down_scale

#     @staticmethod
#     def padding(i, o, k, d, s):
#         return (d * k + (o - 1) * s - d - i + 2) // 2

#     @staticmethod
#     def padding_transposed(it, ot, k, d, s):
#         p = ConvArithmetic.padding(ot, it, k, d, s)
#         a = ot - ((it - 1) * s - 2 * p + d * (k - 1) + 1)
#         return p, a


# base_size = 97
# i = ConvArithmetic.size(base_size, 1)
# o = ConvArithmetic.size(base_size, 2)
# d = 3
# k = 3
# s = 3
# p, a = ConvArithmetic.padding_transposed(
#     it=o,
#     ot=i,
#     k=k,
#     d=d,
#     s=s,
# )

# import torch

# x = torch.rand((1, 1, i, i))
# conv = torch.nn.Conv2d(1, 1, kernel_size=k, stride=s, dilation=d, padding=p)
# y = conv(x)
# print(x.shape)
# print(y.shape)

# x = torch.rand((1, 1, o, o))
# conv = torch.nn.ConvTranspose2d(
#     1, 1, kernel_size=k, stride=s, dilation=d, padding=p, output_padding=a
# )
# y = conv(x)
# print(x.shape)
# print(y.shape)

# # 交换率 f(s2)(f(s1)(x)) = f(s1)(f(s2)(x))
# # 等变性 f(s2)(f(s1)(x)) = f(s2*s1)(x)

# # import random

# # for f in (ConvArithmetic.size1, ConvArithmetic.size2):
# #     for _ in range(10):
# #         base_size = random.randint(1, 20000)
# #         down_scale =
# #         A = f(f(97, 3), 2)
# #         B = f(f(97, 2), 3)
# #         C = f(97, 6)

# # %%
