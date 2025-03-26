ImageNet 非自适应攻击
HSJA:
    PIHA:
    CUDA_VISIBLE_DEVICES=3 python main.py --config configs/imagenet/piha/hsja/targeted/standard/config.json
    blacklight:
    CUDA_VISIBLE_DEVICES=3 python main.py --config configs/imagenet/blacklight/hsja/targeted/standard/config.json
    DPF:
    CUDA_VISIBLE_DEVICES=3 python main.py --config configs/imagenet/blacklight/hsja/targeted/fooler/normalconfig.json

QEBA:
    PIHA:
    CUDA_VISIBLE_DEVICES=2 python main.py --config configs/imagenet/piha/hsja/targeted/standard/config.json --method QEBA
    blacklight:
    CUDA_VISIBLE_DEVICES=2 python main.py --config configs/imagenet/blacklight/hsja/targeted/standard/config.json --method QEBA
    DPF:
    CUDA_VISIBLE_DEVICES=2 python main.py --config configs/imagenet/blacklight/hsja/targeted/fooler/normalconfig.json --method QEBA


DBA-GP:
    PIHA:
    CUDA_VISIBLE_DEVICES=3 python main.py --config configs/imagenet/piha/hsja/targeted/standard/config.json --method GP
    blacklight:
    CUDA_VISIBLE_DEVICES=3 python main.py --config configs/imagenet/blacklight/hsja/targeted/standard/config.json --method GP
    DPF:
    CUDA_VISIBLE_DEVICES=3 python main.py --config configs/imagenet/blacklight/hsja/targeted/fooler/normalconfig.json --method GP

Surfree:
    PIHA:
    CUDA_VISIBLE_DEVICES=2 python main.py --config configs/imagenet/piha/surfree/targeted/standard/config.json
    blacklight:
    CUDA_VISIBLE_DEVICES=2 python main.py --config configs/imagenet/blacklight/surfree/targeted/standard/config.json
    DPF:
    CUDA_VISIBLE_DEVICES=2 python main.py --config configs/imagenet/blacklight/surfree/targeted/fooler/normalconfig.json

conda activate oars
cd ccs-23-oras
CUDA_VISIBLE_DEVICES=3 python main_imagenet.py --celeba --eps 0.05
CUDA_VISIBLE_DEVICES=1 python main_imagenet.py --eps

ImageNet 自适应攻击
HSJA:
    PIHA:
    CUDA_VISIBLE_DEVICES=3 python main.py --config configs/imagenet/piha/hsja/targeted/adaptive/config.json
    blacklight:
    CUDA_VISIBLE_DEVICES=3 python main.py --config configs/imagenet/blacklight/hsja/targeted/adaptive/config.json
    DPF:
    CUDA_VISIBLE_DEVICES=3 python main.py --config configs/imagenet/blacklight/hsja/targeted/fooler/config.json

QEBA:
    PIHA:
    CUDA_VISIBLE_DEVICES=3 python main.py --config configs/imagenet/piha/hsja/targeted/adaptive/config.json --method QEBA
    blacklight:
    CUDA_VISIBLE_DEVICES=3 python main.py --config configs/imagenet/blacklight/hsja/targeted/adaptive/config.json --method QEBA
    DPF:
    CUDA_VISIBLE_DEVICES=3 python main.py --config configs/imagenet/blacklight/hsja/targeted/fooler/config.json --method QEBA

# 3090 - 7： celeba上的前6个的resnet

DBA-GP:
    PIHA:
    CUDA_VISIBLE_DEVICES=2 python main.py --config configs/imagenet/piha/hsja/targeted/adaptive/config.json --method GP
    blacklight:
    CUDA_VISIBLE_DEVICES=2 python main.py --config configs/imagenet/blacklight/hsja/targeted/adaptive/config.json --method GP
    DPF:
    CUDA_VISIBLE_DEVICES=2 python main.py --config configs/imagenet/blacklight/hsja/targeted/fooler/config.json --method GP

Surfree:
    PIHA:
    CUDA_VISIBLE_DEVICES=2 python main.py --config configs/imagenet/piha/surfree/targeted/adaptive/config.json
    blacklight:
    CUDA_VISIBLE_DEVICES=2 python main.py --config configs/imagenet/blacklight/surfree/targeted/adaptive/config.json
    DPF:
    CUDA_VISIBLE_DEVICES=2 python main.py --config configs/imagenet/blacklight/surfree/targeted/fooler/config.json
