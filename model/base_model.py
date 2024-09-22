import torch
import torch.nn as nn
import numpy as np

class BaseModel(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device

    # Implement this function to generate data
    def sample_data():
        pass

    def assemble_batch(self, data, augment_shape=2):
        if isinstance(data, np.ndarray):
            data = torch.FloatTensor(data)
        if len(data.shape) == augment_shape:
            data = data.unsqueeze(0)
        data = data.to(self.device)
        return data

    def squeeze_batch(self,data):
        return data.squeeze(0)

    def detach_to_numpy(self,data):
        return data.detach().cpu().numpy()
