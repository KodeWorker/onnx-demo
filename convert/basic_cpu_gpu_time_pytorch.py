import torch
import time

img_size = 3000

###CPU
start_time = time.time()
a = torch.ones(img_size,img_size).cpu()
for _ in range(1000):
    a += a
elapsed_time = time.time() - start_time
print('CPU time = ',elapsed_time)

###GPU
start_time = time.time()
b = torch.ones(img_size,img_size).cuda()
for _ in range(1000):
    b += b
elapsed_time = time.time() - start_time

print('GPU time = ',elapsed_time)