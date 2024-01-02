from torch import nn


# class MyAwesomeModel(nn.Module):
#     """My awesome model."""

#     def __init__(self):
#         super().__init__()
#         self.fc_sizes = [784, 512, 256, 128, 10]
#         self.fc = nn.Sequential(
#             *[nn.Sequential(
#                 nn.Linear(self.fc_sizes[i], self.fc_sizes[i+1]), nn.ReLU(), nn.Dropout(0.2)
#             ) for i in range(len(self.fc_sizes)-2)],
#             nn.Linear(self.fc_sizes[-2], self.fc_sizes[-1])
#         )
    
        
#         self.out = nn.Softmax(dim = 1)
        

        
#     def forward(self, x):
#         x = x.view(-1, 784)
#         x = self.fc(x)
#         x = self.out(x)
#         return x


## create CNN for MNIST

class MyAwesomeModel(nn.Module):
    """My awesome model."""

    def __init__(self):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.Conv2d(1, 32, 3, padding = 1), 
            nn.ReLU(), 
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding = 1), 
            nn.ReLU(), 
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding = 1), 
            nn.ReLU(), 
            nn.MaxPool2d(2)
        )
        
        self.fc = nn.Sequential(
            nn.Linear(128*3*3, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 10)
        )
        
        self.out = nn.Softmax(dim = 1)
        
    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        x = self.conv(x)
        x = x.view(-1, 128*3*3)
        x = self.fc(x)
        x = self.out(x)
        return x