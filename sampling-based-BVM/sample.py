# import torch
# import numpy as np
# # x1=torch.arange(12).view(3,2,2).float()
# # x2=2*torch.arange(12).view(3,2,2).float()
# x1=torch.tensor(np.array([1,2,3])).float()
# x2=torch.tensor(np.array([4,5,6])).float()
#
#
# # print(x1.sum(dim=1).mean().shape)
# # print(x1)
# t1=x1.unsqueeze(0)
# t2=x2.unsqueeze(0)
# aleatoric_t = torch.cat((t1, t2), 0)
# print(aleatoric_t)
#
# #
# indices = torch.tensor([0, 1,0])
#
# aleatoric_op = torch.index_select(aleatoric_t, 0, indices)
# print('op',aleatoric_op)
# # x=torch.cat((x1,x2),0)
# # print(x.shape)
# #
# # x=x.sum(dim=1).mean(dim=1).mean(dim=1)
# # print(x)
# # print(x.shape)
# # a=torch.tensor(np.array([1000000]*5))
# # b=torch.tensor(np.array([100]*5))
# # c=torch.max(a,b)
# # print(c)
import numpy as np
condition = [0, 1, 0]


list1= [1, 1, 1]
list2 = [2, 2, 2]
# a=np.where(condition,  array2,array1)
# print(a)

# from itertools import chain
# from itertools import zip_longest
# a=list(filter(lambda x: x != '', chain.from_iterable(zip_longest(x, y, fillvalue = ''))))
# print(a)
list = [None]*(len(list1)+len(list2))
list[::2] = list1
list[1::2] = list2
print(list)