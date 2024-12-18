import datetime
import os
import json
from comet_ml import Experiment

import torch
import torch.nn as nn
from torch import optim
from torchvision import datasets
from torchvision import transforms

from sklearn.metrics import confusion_matrix, accuracy_score

with open('config.json','r') as f:
    env = json.load(f)

experiment = Experiment(api_key=env["COMET_API_KEY"], workspace=env["COMET_WORKSPACE"], project_name=env["COMET_PROJECT_NAME"], log_code=True, auto_output_logging="simple")
experiment.set_name("ViT FMNIST")

def get_loader(args):
    data_path = os.path.join('./data/', 'fmnist')
    os.makedirs(data_path, exist_ok=True)
    train_transform = transforms.Compose([transforms.RandomCrop(args['image_size'], padding=2, padding_mode='edge'), 
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(), 
                                            transforms.Normalize([0.5], [0.5])])
    train = datasets.FashionMNIST(data_path, train=True, download=True, transform=train_transform)
    test_transform = transforms.Compose([transforms.Resize([args['image_size'], args['image_size']]), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    test = datasets.FashionMNIST(data_path, train=False, download=True, transform=test_transform)

    train_loader = torch.utils.data.DataLoader(dataset=train,
                                                 batch_size=args['batch_size'],
                                                 shuffle=True,
                                                 num_workers=args['num_workers'],
                                                 drop_last=True)

    test_loader = torch.utils.data.DataLoader(dataset=test,
                                                batch_size=args['batch_size'] * 2,
                                                shuffle=False,
                                                num_workers=args['num_workers'],
                                                drop_last=False)
    return train_loader, test_loader


class EmbedLayer(nn.Module):
    def __init__(self, n_channels, embed_dim, image_size, patch_size):
        super().__init__()
        self.conv1 = nn.Conv2d(n_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=True)
        self.pos_embedding = nn.Parameter(torch.zeros(1, (image_size // patch_size) ** 2 + 1, embed_dim), requires_grad=True)

    def forward(self, x):
        x = self.conv1(x)
        x = x.reshape([x.shape[0], x.shape[1], -1])
        x = x.transpose(1, 2)
        x = torch.cat((torch.repeat_interleave(self.cls_token, x.shape[0], 0), x), dim=1)
        x = x + self.pos_embedding
        return x

class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.wq = nn.Linear(self.embed_dim, self.embed_dim)
        self.wk = nn.Linear(self.embed_dim, self.embed_dim)
        self.wv = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self, x, mask=None):
        B, S, E = x.shape

        xq = self.wq(x)
        xk = self.wk(x)
        xv = self.wv(x)

        xq = xq.view(B, S, self.num_heads, self.head_dim)
        xk = xk.view(B, S, self.num_heads, self.head_dim)
        xv = xv.view(B, S, self.num_heads, self.head_dim)
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        xk = xk.transpose(-1, -2)
        x_attn = torch.matmul(xq, xk)
        x_attn /= float(self.head_dim) ** 0.5
        if mask is not None:
            x_attn += mask.to(x_attn.dtype) * x_attn.new_tensor(-1e4)
        x_attn = torch.softmax(x_attn, dim=-1)
        x = torch.matmul(x_attn, xv)

        x = x.transpose(1, 2)
        x = x.reshape(B, S, E)
        return x


class Encoder(nn.Module):
    def __init__(self, embed_dim, n_attention_heads, forward_mul):
        super().__init__()
        self.attention = SelfAttention(embed_dim, n_attention_heads)
        self.fc1 = nn.Linear(embed_dim, embed_dim * forward_mul)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(embed_dim * forward_mul, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.fc2(self.activation(self.fc1(self.norm2(x))))
        return x


class Classifier(nn.Module):
    def __init__(self, embed_dim, n_classes):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, embed_dim)
        self.activation = nn.Tanh()
        self.fc2 = nn.Linear(embed_dim, n_classes)

    def forward(self, x):
        x = x[:, 0, :]
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x


class VisionTransformer(nn.Module):
    def __init__(self, n_channels, embed_dim, n_layers, n_attention_heads, forward_mul, image_size, patch_size, n_classes):
        super().__init__()
        self.embedding = EmbedLayer(n_channels, embed_dim, image_size, patch_size)
        self.encoder = nn.Sequential(*[Encoder(embed_dim, n_attention_heads, forward_mul) for _ in range(n_layers)], nn.LayerNorm(embed_dim))
        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = Classifier(embed_dim, n_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = self.encoder(x)
        x = self.norm(x)
        x = self.classifier(x)
        return x


class Solver(object):
    def __init__(self, args):
        self.args = args

        self.args['model_path'] = os.path.join(args['model_path'], 'fmnist')
        os.makedirs(self.args['model_path'], exist_ok=True)
        print('Model path: ', self.args['model_path'])
        self.args['n_patches'] = (args['image_size'] // args['patch_size']) ** 2
        
        
        self.train_loader, self.test_loader = get_loader(args)

        self.model = VisionTransformer(n_channels=self.args['n_channels'], embed_dim=self.args['embed_dim'], 
                                        n_layers=self.args['n_layers'], n_attention_heads=self.args['n_attention_heads'], 
                                        forward_mul=self.args['forward_mul'], image_size=self.args['image_size'], 
                                        patch_size=self.args['patch_size'], n_classes=self.args['n_classes'])
        
        if torch.cuda.is_available():  
            dev = "cuda:0"
            self.args['is_cuda'] = True
        else:
            dev = "cpu"
            self.args['is_cuda'] = False
        device = torch.device(dev)
        self.model = self.model.to(device)
        print(f"Using device {device}")

        if args['load_model']:
            print("Using pretrained model")
            self.model.load_state_dict(torch.load(os.path.join(self.args['model_path'], 'ViT_model.pt')))

        self.ce = nn.CrossEntropyLoss()

    def test_dataset(self, loader):
        self.model.eval()

        actual = []
        pred = []

        for (x, y) in loader:
            if self.args['is_cuda']:
                x = x.cuda()

            with torch.no_grad():
                logits = self.model(x)
            predicted = torch.max(logits, 1)[1]

            actual += y.tolist()
            pred += predicted.tolist()

        acc = accuracy_score(y_true=actual, y_pred=pred)
        cm = confusion_matrix(y_true=actual, y_pred=pred, labels=range(self.args['n_classes']))

        return acc, cm

    def test(self, train=True):
        if train:
            acc, cm = self.test_dataset(self.train_loader)
            print(f"Train acc: {acc:.2%}\nTrain Confusion Matrix:")
            print(cm)

        acc, cm = self.test_dataset(self.test_loader)
        print(f"Test acc: {acc:.2%}\nTest Confusion Matrix:")
        print(cm)

        return acc

    def train(self):
        iter_per_epoch = len(self.train_loader)
        #experiment.log_parameters()
        optimizer = optim.AdamW(self.model.parameters(), lr=self.args['lr'], weight_decay=1e-3)
        linear_warmup = optim.lr_scheduler.LinearLR(optimizer, start_factor=1/self.args['warmup_epochs'], end_factor=1.0, total_iters=self.args['warmup_epochs'], last_epoch=-1, verbose=True)
        cos_decay = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.args['epochs']-self.args['warmup_epochs'], eta_min=1e-5, verbose=True)

        best_acc = 0
        for epoch in range(self.args['epochs']):

            self.model.train()

            for i, (x, y) in enumerate(self.train_loader):
                if self.args['is_cuda']:
                    x, y = x.cuda(), y.cuda()

                logits = self.model(x)
                loss = self.ce(logits, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if i % 50 == 0 or i == (iter_per_epoch - 1):
                    print(f'Ep: {epoch+1}/{self.args["epochs"]}, It: {i+1}/{iter_per_epoch}, loss: {loss:.4f}')

            test_acc = self.test(train=((epoch+1)%25==0)) # Test training set every 25 epochs
            best_acc = max(test_acc, best_acc)
            print(f"Best test acc: {best_acc:.2%}\n")

            torch.save(self.model.state_dict(), os.path.join(self.args['model_path'], "ViT_model.pt"))
            
            if epoch < self.args['warmup_epochs']:
                linear_warmup.step()
            else:
                cos_decay.step()


def main(args):
    os.makedirs(args['model_path'], exist_ok=True)

    solver = Solver(args)
    solver.train()
    solver.test(train=True)

if __name__ == '__main__':
    torch.manual_seed(0)


    model_args ={"epochs": 200,
                 "batch_size": 256,
                 "num_workers": 4,
                 "lr": 5e-4,
                 "n_classes": 10,
                 "warmup_epochs": 10,
                 "image_size": 28,
                 "n_channels": 1,
                 "embed_dim": 64,
                 "n_attention_heads": 4,
                 "patch_size": 4,
                 "forward_mul": 2,
                 "n_layers": 6,
                 "model_path": './model',
                 "load_model": False}
    
    start_time = datetime.datetime.now()
    print("Started at " + str(start_time.strftime('%Y-%m-%d %H:%M:%S')))
    main(model_args)

    end_time = datetime.datetime.now()
    duration = end_time - start_time
    print("Ended at " + str(end_time.strftime('%Y-%m-%d %H:%M:%S')))
    print("Duration: " + str(duration))

experiment.end()
