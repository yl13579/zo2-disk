import torch
import torch.nn as nn

device = 0
torch.cuda.set_device(device)

model = nn.Linear(1000, 1000).to(device)  # 初始占用显存
# print(model)
# print("显存占用（移动前）:", torch.cuda.memory_allocated())  # 输出非零值
# torch.save(model.state_dict(), f"test.pth")
# print("显存占用（移动前）:", torch.cuda.memory_allocated())
# model.to("meta")  # 转移到meta设备
# print("显存占用（移动后）:", torch.cuda.memory_allocated())  # 显存应减少或归零
# model.to_empty(device=device) 
# print("显存占用（移动回去）:", torch.cuda.memory_allocated())
# state_dict = torch.load("test.pth",map_location="cpu")
# model.load_state_dict(state_dict)
# del state_dict
# print("最大显存占用:", torch.cuda.max_memory_allocated())
print("显存占用（移动前）:", torch.cuda.memory_allocated())
from tensornvme import DiskOffloader
offloader = DiskOffloader('./offload_dir')
state_dict = model.state_dict()
# print(state_dict)
for key, tensor in state_dict.items():
    tensor.to("cpu")
    print(key)
    print(tensor.device)
    
    # offloader.sync_write(tensor)
print("显存占用（移动前）:", torch.cuda.memory_allocated())
tensor_test = torch.rand(size=[10,10])
offloader.sync_write(tensor_test)
print(tensor_test)
offloader.sync_read(tensor_test)
print(tensor_test)
