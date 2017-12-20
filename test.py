import torch
import torch.nn as nn
from torch.autograd import Variable

# TODO: proves that zero out operation doesn't have gradient

X = Variable(torch.FloatTensor([4,4]))
a = Variable(torch.FloatTensor([2,4]), requires_grad = True)
y = a * X
criterion = nn.MSELoss(size_average = True)
optimizer = torch.optim.Adam([a], lr = 1e-2)

threshold = 3
cond = torch.abs(a.data) < threshold
weight_ = a.data + 0
weight_[cond] = 0
a.data = weight_

# times gradient by 10
#h = a.register_hook(lambda grad: grad * 10)
#optimizer.zero_grad()
loss = criterion(y, Variable(torch.FloatTensor([7,7])))
loss.backward()

a.grad.data[cond] = 0

optimizer.step()

print(a)
print(a.grad)
