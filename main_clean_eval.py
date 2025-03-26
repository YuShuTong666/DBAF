import torchvision.models as models
from models.statefuldefense import init_stateful_classifier_v2
import argparse
import json
import os
import warnings
import logging
import logging.handlers
#import multiprocessing
import torch
import numpy as np
from torchvision import transforms
import random
from torchvision import datasets
import torch.utils.data as Data
import os
from tqdm import tqdm
#os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.0002, help='lr.')
parser.add_argument('--seed', type=int, default=123, help='Random seed.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=2, help='Number of hidden units.')
parser.add_argument('--n_img', type=int, default=50, help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate (1 - keep probability).')
parser.add_argument('--epochs', type=int, default=500, help='Number of epochs to train.')
parser.add_argument('--k', type=int, default=5000, help='Number of nodes to train.')
parser.add_argument('--celeba', action='store_true', help='celeba')
parser.add_argument('--maxq', type=int, default=5000, help='Number of max queries.')
parser.add_argument('--method', default='HSJA', help='Options: "GM", "HSJA", "QEBA", "GF", "QEBAM", "JBF", "GIF", "RBD".')
parser.add_argument('--layers', type=int, default=16, help='Number of graph filter layers.')
parser.add_argument('--alpha', type=float, default=0.0, help='Initial residual connection.')
parser.add_argument('--T', type=float, default=0.0002, help='Initial residual connection.')
parser.add_argument('--delta', type=float, default=0.8, help='Initial residual connection.')
parser.add_argument('--model', type=str, default="resnet", help='Option: resnet, vgg, inception')
parser.add_argument('--r', type=int, default=3, help='.')
parser.add_argument('--iteration', type=int, default=8, help='.')
parser.add_argument('--init', type=int, default=100, help='.')
parser.add_argument('--bf', action='store_true', default=False, help='bi filter.')
parser.add_argument('--time', action='store_true', default=False, help='Time prior.')
parser.add_argument('--ntime', action='store_true', default=False, help='Naive Time Prior.')
parser.add_argument('--disable_logging', action='store_true')
parser.add_argument('--config', type=str, default='configs/imagenet/blacklight/hsja/targeted/fooler/config.json')
parser.add_argument('--num_images', type=int, default=50)
parser.add_argument('--start_idx', type=int, default=0)
parser.add_argument('--log_dir', type=str)
parser.add_argument('--bright', action='store_true')
parser.add_argument('--adap_c', action='store_true')
parser.add_argument('--eps', type=float, default=0.1, help='eps of ')
parser.add_argument('--nosalt', action='store_true')
parser.add_argument('--random_init', action='store_true')
parser.add_argument('--no_hash', action='store_true')
parser.add_argument('--accurate', action='store_true')
parser.add_argument('--normal', action='store_true')
parser.add_argument('--normal_eval', action='store_true')
args = parser.parse_args()


def normal_eval(model, dataloader):
    model.eval()

    cum_acc = 0.0
    tot_num = 0.0
    cache_num = 0.0
    for X, y in tqdm(dataloader):
        X = X.cuda()
        y = y.cuda()
        B = X.size()[0]
        with torch.no_grad():
            pred = model(X)
            #loss = model.loss(pred, y)

        #cum_loss += loss.item() * B
        pred_c = pred.max(1)[1].cpu()
        cum_acc += (pred_c.eq(y.cpu())).sum().item()
        tot_num = tot_num + B
        #cache_num += np.count_nonzero(is_cache)
        tqdm.write("Accuracy: ", end="")
        tqdm.write(str(cum_acc / tot_num))

    print(cum_acc / tot_num)
    # 0.74548
    return cum_acc / tot_num


def epoch_eval(model, dataloader):
    model.eval()

    cum_acc = 0.0
    tot_num = 0.0
    cache_num = 0.0
    for X, y in tqdm(dataloader):
        X = X.cuda()
        y = y.cuda()
        B = X.size()[0]
        with torch.no_grad():
            pred, is_cache = model.forward_batch(X)
            #loss = model.loss(pred, y)

        #cum_loss += loss.item() * B
        pred_c = pred.max(1)[1].cpu()
        cum_acc += (pred_c.eq(y.cpu())).sum().item()
        tot_num = tot_num + B
        cache_num += np.count_nonzero(is_cache)
        tqdm.write("Accuracy: ", end="")
        tqdm.write(str(cum_acc / tot_num), end='   ')
        tqdm.write("Cached Rate: ", end='')
        tqdm.write(str(cache_num / tot_num))

    print(cum_acc / tot_num)
    # 0.7455  0.00312  self.eps=0.2

    # 在3090-3上测试eps=0.05的防御效果
    # ImageNet resnet：
    # DPF: 0.00294  eps=0.1
    #    : 0.00122  eps=0.05
    # blacklight: 0.7455 0.00242
    # PIHA 0.00074
    # ImageNet vit：

    # Celeba resnet:
    # DPF : 0.0  eps=0.05 or 0.1
    # blacklight: 0.000413
    return cum_acc / tot_num


config = json.load(open(args.config))
model_config, attack_config = config["model_config"], config["attack_config"]
#model = models.resnet50(pretrained=True).eval()
if args.celeba:
    from main import CelebAModel
    model_root = 'path/to/your model'
    model = CelebAModel(100, model=args.model, pretrained=False, gpu=True)
    if args.model == 'resnet':
        stat = torch.load(model_root + "resnetceleba.model")
        model.load_state_dict(stat)
        print("Attacked model: resnet50")
    elif args.model == 'vit':
        stat = torch.load(model_root + "vitceleba.model")
        model.load_state_dict(stat)
    else:
        raise Exception("Invalid model name!")
    model.eval()
    # if args.use_gpu:
    model.cuda()
    if not args.normal:
        model = init_stateful_classifier_v2(model_config, model, args)
    root = '../celeba/celeba'
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    from celeba_dataset import CelebADataset
    testset = CelebADataset(root_dir=root, is_train=True, transform=transform, preprocess=False, random_sample=False, n_id=100)
    loader = Data.DataLoader(
        dataset=testset,
        batch_size=64,
        shuffle=False,
    )
else:
    if args.model == 'resnet':
        model = models.resnet50(pretrained=True).eval()
        print("Attacked model: resnet50")
    elif args.model == 'vgg':
        model = models.vgg16(pretrained=True).eval()
        print("Attacked model: vgg16")
    elif args.model == 'vit':
        model = models.vit_b_16(pretrained=True).eval()
    model.cuda()
    if not args.normal:
        model = init_stateful_classifier_v2(model_config, model, args)
    root = 'path/to/imagenet val'
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if args.normal:
        transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), normalize])
    else:
        transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    dataset = datasets.ImageFolder(root=root, transform=transform)
    loader = Data.DataLoader(
        dataset=dataset,
        batch_size=1024,
        shuffle=False,
    )
if not args.normal_eval:
    epoch_eval(model, loader)
else:
    normal_eval(model, loader)
