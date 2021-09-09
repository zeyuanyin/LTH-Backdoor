CUDA_VISIBLE_DEVICES=0 python mask_overlap_rate.py \
--dataset cifar10 \
--depth 18 \
--arch resnet18_cifar \
--percent 0.5 \
--save ./baseline/resnet18_cifar10/EB-30-35.pth.tar \
--save_1 ./backdoor_1/resnet18_cifar10/EB-30-35.pth.tar