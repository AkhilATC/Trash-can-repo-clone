import torch

if __name__ == "__main__":
    arranged = torch.arange(4)
    # print(arranged)
    #What does .unsqueeze(1) do?
    # It adds a new dimension at index 1.
    arranged = arranged.unsqueeze(1) # [0,1,2,3] -> [ [0],[1],[2],[3] ]
    # print(arranged)
    t = torch.arange(0,12,2)
    print(t)
