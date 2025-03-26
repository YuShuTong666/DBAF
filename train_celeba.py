import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torchvision
import torchvision.models as models
from torchvision import transforms
import celeba_dataset
from celeba_dataset import CelebADataset
from autoaugment import ImageNetPolicy
import argparse


transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.ToTensor()
])


class CelebAModel(nn.Module):
    def __init__(self, num_class, model='resnet', pretrained=False, gpu=True):
        super(CelebAModel, self).__init__()
        self.pretrained = pretrained
        self.gpu = gpu
        self.num_class = num_class
        self.is_vgg = False
        #self.resnet = models.resnet18(pretrained=pretrained)
        #self.resnet = models.wide_resnet50_2(pretrained=pretrained)
        if model == 'resnet':
            self.model = models.resnet50(pretrained=pretrained, num_classes=self.num_class)
            print("Train model: resnet50")
        elif model == 'vgg':
            self.model = models.vgg16_bn(pretrained=pretrained, num_classes=self.num_class)
            print("Train model: vgg16")
        elif model == 'inception':
            self.model = models.inception_v3(pretrained=pretrained, num_classes=self.num_class, transform_input=True, aux_logits=False)
            print("Train model: inception_v3")
        elif model == 'vit':
            print("Train model: ViT")
            self.is_vgg = True
            self.model = models.vit_b_16(pretrained=pretrained)
            self.output = nn.Linear(1000, self.num_class)
        else:
            raise Exception("Invalid model name!")


        if gpu:
            self.cuda()

    def forward(self, x):
        if self.gpu:
            x = x.cuda()
        x = self.model(x)
        if self.is_vgg:
            x = self.output(x)

        return x

    def loss(self, pred, label):
        if self.gpu:
            label = label.cuda()
        return F.cross_entropy(pred, label)


# class CelebADenseNet(nn.Module):
#     def __init__(self, num_class, pretrained=True, gpu=False):
#         super(CelebADenseNet, self).__init__()
#         self.pretrained = pretrained
#         self.gpu = gpu
#         self.num_class = num_class
#
#         self.densenet = models.densenet121(pretrained=pretrained)
#         self.output = nn.Linear(1000, self.num_class)
#
#         if gpu:
#             self.cuda()
#
#     def forward(self, x):
#         if self.gpu:
#             x = x.cuda()
#         #x = F.interpolate(x, [224,224])
#         #x = self.resnet(x)
#
#         x = self.densenet.features(x)
#         x = F.relu(x, inplace=True)
#         x = x.view(x.size(0), -1)
#         x = self.densenet.classifier(x)
#
#         x = self.output(x)
#
#         return x
#
#     def loss(self, pred, label):
#         if self.gpu:
#             label = label.cuda()
#         return F.cross_entropy(pred, label)


def epoch_train(model, optimizer, dataloader):
    model.train()

    cum_loss = 0.0
    cum_acc = 0.0
    tot_num = 0.0
    for X, y in dataloader:
        B = X.size()[0]
        if (B==1):
            continue
        pred = model(X)
        loss = model.loss(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        cum_loss += loss.item() * B
        pred_c = pred.max(1)[1].cpu()
        cum_acc += (pred_c.eq(y)).sum().item()
        tot_num = tot_num + B

    print(cum_loss / tot_num, cum_acc / tot_num)
    return cum_loss / tot_num, cum_acc / tot_num

def epoch_eval(model, dataloader):
    model.eval()

    cum_loss = 0.0
    cum_acc = 0.0
    tot_num = 0.0
    for X, y in dataloader:
        B = X.size()[0]
        with torch.no_grad():
            pred = model(X)
            loss = model.loss(pred, y)

        cum_loss += loss.item() * B
        pred_c = pred.max(1)[1].cpu()
        cum_acc += (pred_c.eq(y)).sum().item()
        tot_num = tot_num + B

    print(cum_loss / tot_num, cum_acc / tot_num)
    return cum_loss / tot_num, cum_acc / tot_num


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_id', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--wd', type=float, default=1e-5)
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--bs', type=int, default=64)
    parser.add_argument('--adam', action='store_true')
    parser.add_argument('--model', type=str, default='resnet')
    args = parser.parse_args()
    gpu = True
    num_class = args.n_id
    n_e = 500
    batch_size = args.bs

    root_dir = '../celeba/celeba'
    do_random_sample = False
    '''print("Preprocessing...")
    celeba_dataset.preprocess_data(root_dir, n_id=num_class, random_sample=do_random_sample)
    print("Preprocess finished.")
    print("Sorting...")
    celeba_dataset.sort_imgs(root_dir)
    print("Sort finished.")
    print("Getting dataset...")
    celeba_dataset.get_dataset(root_dir, n_id=num_class, random_sample=do_random_sample)
    print("Get dataset finished.")'''
    trainset = CelebADataset(root_dir=root_dir, is_train=True, transform=transforms.Compose(
                        [transforms.RandomResizedCrop(224),
                         transforms.RandomHorizontalFlip(), ImageNetPolicy(),
                         transforms.ToTensor(), torchvision.transforms.Resize((224, 224))]), preprocess=False,
                             random_sample=do_random_sample, n_id=num_class)
    '''trainset = CelebADataset(root_dir=root_dir, is_train=True, transform=transform, preprocess=False,
                             random_sample=do_random_sample, n_id=num_class)'''
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testset = CelebADataset(root_dir=root_dir, is_train=False, transform=transform, preprocess=False,
                             random_sample=do_random_sample, n_id=num_class)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
    #print (len(trainset), len(testset))
    #assert 0

    model = CelebAModel(num_class, model=args.model, pretrained=args.pretrained, gpu=gpu)
    '''if args.adam:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)

    for e in range(n_e):
        print("Epoch %d" %(e))
        epoch_train(model=model, optimizer=optimizer, dataloader=trainloader)
        print("Evaluate")
        epoch_eval(model=model, dataloader=testloader)
        torch.save(model.state_dict(), '../models/'+args.model+'celeba.model')'''
    model_path = '../models/'+args.model+'celeba.model'
    model_param = torch.load(model_path)
    model.load_state_dict(model_param)
    epoch_eval(model=model, dataloader=testloader)