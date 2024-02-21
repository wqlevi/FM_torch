import torch

from torchcfm.models.unet import ProUNetModel, UNetModel

def count_param(model):
    if isinstance(model, torch.nn.Module):
        count = sum(para.data.nelement() for para in model.parameters())
        count /= 1024**2
        print(f"Num of params: {count=:.2f} M")

def save_model_arch(model: torch.nn.Module):
    with open("model_step1_28.txt", "w") as f:
        f.write(model.__repr__())

model = ProUNetModel(dim=(1,32,32), num_channels=32, num_res_blocks=1).to("cuda:0")
model_og = UNetModel(dim=(1,32,32), num_channels=32, num_res_blocks=1).to("cuda:0")

# step 0 (14, 14):
print(f"Trying step = 0")
x = torch.ones((4,1,16, 16), dtype=torch.float32, device=torch.device("cuda:0"))
t = torch.rand(x.shape[0]).type_as(x)
y = model(t,x)
count_param(model)
print(y.shape)

# step 1 (28, 28):
print(f"Trying step = 1")
x = torch.ones((4,1,32,32), dtype=torch.float32, device=torch.device("cuda:0"))
t = torch.rand(x.shape[0]).type_as(x)
y = model(t,x, step=1)
count_param(model)
print(y.shape)

print(f"Trying og model")
x = torch.ones((4,1,32,32), dtype=torch.float32, device=torch.device("cuda:0"))
t = torch.rand(x.shape[0]).type_as(x)
y = model_og(t,x)
#y = model(t,x)
count_param(model_og)
print(y.shape)
