import torch
from torch import nn
from torch.autograd import Variable
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt

dataset = MNIST('./data',transform=transforms.ToTensor(), download=True)
dataloader = DataLoader(dataset,batch_size=100,shuffle=True)


class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder,self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(28*28,128),
            nn.ReLU(inplace=True),
            nn.Linear(128,64),
            nn.ReLU(inplace=True),
            nn.Linear(64,12),
            nn.ReLU(inplace=True),
            nn.Linear(12, 3)
        )

        self.decoder = nn.Sequential(
            nn.Linear(3,12),
            nn.ReLU(inplace=True),
            nn.Linear(12,64),
            nn.ReLU(inplace=True),
            nn.Linear(64,128),
            nn.ReLU(inplace=True),
            nn.Linear(128,28*28),
            nn.Tanh()
        )

    def forward(self,x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


model = autoencoder()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train = True

if train:
    for epoch in range(100):
        for data in dataloader:
            img, _ = data
            img = img.view(img.size(0), -1)
            img = Variable(img)
            #print(img.size())

            output = model(img)
            #print(output.size())
            loss = criterion(output,img)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print('Epoch :', epoch+1, 'Loss:',loss.item())

    torch.save(model, './sim_autoencoder.pth')

if not train:
    model = torch.load('./sim_autoencoder.pth')
    model.eval()

    img, _ = next(iter(dataloader))
    plt.imshow(img.numpy()[0].reshape(28,28), cmap='gray')
    plt.show()

    img = img.view(img.size(0), -1)
    img = Variable(img)
    #print(img.size())
    output = model(img)
    #print(output.size())
    plt.imshow(output.detach().numpy()[0].reshape(28,28), cmap='gray')
    plt.show()