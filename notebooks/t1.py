import torch 

if __name__ == "__main__":
    tensor = torch.Tensor([[1.0, 0.0], [0.0, 1.0]])
    print(torch.max(tensor, 1).indices)