import sys
import os
import json
import warnings
import logging
import base64
from PIL import Image
from fastapi import FastAPI, Request, status
from pydantic import BaseModel, ValidationError, root_validator
from fastapi.exceptions import HTTPException, RequestValidationError
from fastapi.responses import JSONResponse
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms

warnings.filterwarnings("ignore")


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
        x = self.classifier


class RequestBaseModel(BaseModel):
    @root_validator(pre=True)
    def body_params_case_insensitive(cls, values: dict):
        for field in cls.__fields__:
            in_fields = list(filter(lambda f: f.lower() == field.lower(), values.keys()))
            for in_field in in_fields:
                values[field] = values.pop(in_field)

        return values

class Question(BaseModel):
    question: str


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
             "model_path": os.path.join(os.path.join('./model', 'fmnist'), 'ViT_model.pt'),
             "load_model": False}

model = VisionTransformer(n_channels=model_args['n_channels'], embed_dim=model_args['embed_dim'],
                          n_layers=model_args['n_layers'], n_attention_heads=model_args['n_attention_heads'],
                          forward_mul=model_args['forward_mul'], image_size=model_args['image_size'],
                          patch_size=model_args['patch_size'], n_classes=model_args['n_classes'])

model.load_state_dict(torch.load(model_args['model_path'], weights_only=True))
if torch.cuda.is_available():  
    dev = "cuda:0"
else:
    dev = "cpu"
device = torch.device(dev)
model = model.to(device)
model.eval()

def predict(x):
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    x = transform(x.convert('L'))    
    x = x.to(device)
    print(x.size())
    with torch.no_grad():
        logits = model(x)    
    return torch.max(logits, 1)[1]


app = FastAPI()

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
	exc_str = f'{exc}'.replace('\n', ' ').replace('   ', ' ')
	logging.error(f"{request}: {exc_str}")
	content = {'status_code': 10422, 'message': exc_str, 'data': None}
	return JSONResponse(content=content, status_code=status.HTTP_422_UNPROCESSABLE_ENTITY)

@app.get("/")
def get_root():
    return ""

@app.post("/model")
def process_req(question:Question):
    user_input = question.question
    img = Image.open(BytesIO(base64.b64decode(user_input)))
    prediction = predict(img)
    return json.dumps({'result': prediction})
