from torchvision import transforms
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms
'''img_path = "box/data/imagenet_07_609.jpg"
transform = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor()])
img = transform(Image.open(img_path))
img = (img * 255)%255
img = img.numpy().astype(np.int16)
rd = 50
salt = (np.random.rand(*tuple(img.shape)) * 255).astype(np.int16)
print(img)
img = img+salt
plt.imshow(np.moveaxis(img, 0, 2))
plt.show()
array = np.around(img / rd, decimals=0) * rd
plt.imshow(np.moveaxis(array, 0, 2))
plt.show()
print(array)
print(salt)'''
'''mean = torch.tensor([0.485, 0.456, 0.406])
std = torch.tensor([0.229, 0.224, 0.225])

normalize = torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
transform_norm = transforms.Compose([normalize])
img2 = transform_norm(img)
mean = mean.reshape(1, -1)
std = std.reshape(1, -1)
print(img)
print(img.shape)
#img2 = (img-mean) / std
print(img2)
q = torch.randint(0, 9, (1,))
print(q)
a = torch.range(1, 10)
a = a.reshape(1, -1)
sa = torch.softmax(a, dim=1)
print(sa[0][q])'''


def compute_mse(x1, x2):
    dis = np.linalg.norm(x1.cpu().numpy() - x2.cpu().numpy())
    mse = dis ** 2 / np.prod(x1.cpu().numpy().shape)
    return mse


'''x_adv = img / 5
x = transform(Image.open("box/data/imagenet_08_915.jpg")) / 5
norm_dist = torch.linalg.norm(x_adv - x) / (x.shape[-1] * x.shape[-2] * x.shape[-3]) ** 0.5
mse = compute_mse(x_adv, x.unsqueeze(0))
print(norm_dist)
print(mse)'''
'''print(img)
img = transform(Image.open(img_path).convert("RGB"))
print(img.max())'''
'''xs = []
x = [1]
xs.extend(x)
xs.extend([mse])
print(xs)
print(np.mean(xs))
'''
'''from collections import Counter
x = Counter('abcdeabcdabcaba')#.most_common(3)
y = x.most_common(1)
print(x)
print(y)
x = [False, True, True]
y = np.count_nonzero(x)
print(y)
a = torch.range(0, 24).reshape(5, 5)
a = a.reshape(-1)
a = a.repeat(31, 1)
print(a.shape)
print(a)
c = torch.tensor(range(-15, 16)).reshape(31, 1).repeat(1, a.shape[1])
print(c.shape)
d = a+c
round = 50
array_discretized = torch.round(d / round) * round
e = torch.sum(torch.abs(d - array_discretized), dim=1)
print(array_discretized)
print(e)
print(e.shape)
max_id = e.argmax().numpy()
print(max_id-15)
print(type(max_id))

root = 'box/data'
img = torchvision.io.read_image(root+'/src19.jpg')
print(img/255)'''
dim = 3 * 255 * 255
import math
print(1500 / (math.sqrt(dim) * dim))