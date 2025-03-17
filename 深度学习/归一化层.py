import torch

class BatchNorm:
    def __init__(self, num_features, eps=1e-5, moentum=0.1):
        self.num_features = num_features
        self.eps = eps
        self.momentum = moentum

        self.gamma = torch.ones(num_features)
        self.beta = torch.zeros(num_features)

        self.running_mean = torch.zeros(num_features)
        self.running_var = torch.ones(num_features)

    def forward(self,x,training=True):
        if x.dim() != 4:
            raise ValueError("输入数据必须是4D张量，[B,C,H,W]")
        
        B,C,H,W=x.shape

        if training:
            mean = x.mean(dim=(0,2,3),keepdim=True)
            var = x.var(dim=(0,2,3),keepdim=True)

            self.running_mean = (1-self.momentum) * self.running_mean + self.momentum * mean.squeeze()
            self.running_var = (1-self.momentum) * self.running_var + self.momentum * var.squeeze()
        else:
            mean = self.running_mean.view(1,C,1,1)
            var = self.running_var.view(1,C,1,1)

        x_hat = (x-mean) / torch.sqrt(var + self.eps)

        y = self.gamma.view(1,C,1,1) * x_hat + self.beta.view(1,C,1,1)

        return y
    
batch_norm = BatchNorm(num_features=64)

input_data = torch.randn(32,64,128,128)

output = batch_norm.forward(input_data)
