import numpy as np

kp, od = None, None
with open('./data/kp.npy', 'rb') as f:
    kp = np.load(f, allow_pickle=True)
with open('./data/od_info.npy', 'rb') as f:
    od = np.load(f, allow_pickle=True)

print(kp)

left, right, curr = [], [], []

start = (564, 719)


