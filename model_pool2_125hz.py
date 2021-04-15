import torch
import torch.nn as nn

INPUT_HEIGHT = 48
INPUT_WIDTH = 74

# q(z|x, y)
class Qz_xy(nn.Module):
    def __init__(self, z_dim=2, y_dim=10):
        super(Qz_xy, self).__init__()

        self.z_dim = z_dim

        # encode
        self.conv_e = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),  # 88*300 ⇒ 44*150
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 44*150 ⇒ 22*75
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )
        self.fc1 = nn.Sequential(
            nn.Linear((INPUT_HEIGHT//4) * (INPUT_WIDTH//4)* 128,  40),
        )
        self.fc2 = nn.Sequential(
            nn.Linear((INPUT_HEIGHT//4) * (INPUT_WIDTH//4)* 128,  y_dim),
        )

        self.fc = nn.Sequential(
            nn.Linear(40+y_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 2*self.z_dim),
        )

    def forward(self, x, y):
        x = self.conv_e(x)
        x = x.view(-1,(INPUT_HEIGHT//4) * (INPUT_WIDTH//4)* 128)
        x1 = self.fc1(x)
        x2 = self.fc2(x)
        x = torch.cat([x1, x2*y], dim=1)
        x = self.fc(x)
        mu = x[:, :self.z_dim]
        logvar = x[:, self.z_dim:]
        z = self.reparameterize(mu, logvar)
        self.mu = mu
        self.logvar = logvar
        return z

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = std.new(std.size()).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu


# q(y|x)
class Qy_x(nn.Module):
    def __init__(self, y_dim=10):
        super(Qy_x, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4,padding=2),   # 88x300 ⇒ 89x301
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)) # 89*301 ⇒ 44*150

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, padding=2), # 44x150 ⇒ 22x75
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2))

        self.fc =  nn.Sequential(
            nn.Linear(-(-INPUT_HEIGHT//4) * -(-INPUT_WIDTH//4)* 128, 256),
            nn.Dropout(p=0.4),
            nn.ReLU(),
            nn.Linear(256, y_dim),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        c1 = self.conv1(x)
        c2 = self.conv2(c1)
        c2_flat = c2.view(c2.size(0), -1)
        out = self.fc(c2_flat)
        return out

# p(z|y)
class Pz_y(nn.Module):
    def __init__(self, z_dim=2, y_dim=10):
        super(Pz_y, self).__init__()

        self.z_dim = z_dim

        # encode
        self.fc = nn.Sequential(
            nn.Linear(y_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 2*self.z_dim),
        )

    def forward(self, y):
        x = self.fc(y)
        mu = x[:, :self.z_dim]
        logvar = x[:, self.z_dim:]
        z = self.reparameterize(mu, logvar)
        self.mu = mu
        self.logvar = logvar
        return z

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = std.new(std.size()).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu

    def sample(self, y):
        x = self.fc(y)
        mu = x[:, :self.z_dim]
        logvar = x[:, self.z_dim:]
        std = logvar.mul(0.5).exp_()
        eps = std.new(std.size()).normal_()
        return eps.mul(std).add_(mu)

# p(x|z)
class Px_z(nn.Module):
    def __init__(self, z_dim=2):
        super(Px_z, self).__init__()

        self.z_dim = z_dim

        # decode
        self.fc = nn.Sequential(
            nn.Linear(self.z_dim, 40),
        )

        self.fc_d = nn.Sequential(
            nn.Linear(40, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024,(INPUT_HEIGHT//4) * (INPUT_WIDTH//4)* 128),
            nn.LeakyReLU(0.2)
        )
        self.conv_d = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, output_padding=(0,1)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, z):
        z = self.fc(z)
        h = self.fc_d(z)
        h = h.view(-1, 128, (INPUT_HEIGHT//4), (INPUT_WIDTH//4))
        #print(h.shape)
        return self.conv_d(h)

class MyDataset(torch.utils.data.Dataset):

    def __init__(self, data, label, datatime_idx, transform=None):
        self.transform = transform
        self.data = data
        self.data_num = len(data)
        self.label = label
        self.datatime_idx = datatime_idx

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        if self.transform:
            # print(self.data.shape)
            # print(self.data[idx].shape)
            out_data = self.transform(self.data[idx])
            out_label = int(self.label[idx])
            out_datatime_idx = int(self.datatime_idx[idx])
        else:
            out_data = self.data[idx]
            out_label =  self.label[idx]
            out_datatime_idx = self.datatime_idx[idx]

        return out_data, out_label, out_datatime_idx
