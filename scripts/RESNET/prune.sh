CUDA_VISIBLE_DEVICES=0 python vggprune.py \
--dataset cifar10 \
--arch resnet18_cifar \
--test-batch-size 256 \
--depth 18 \
--percent 0.3 \
--model ./baseline/resnet18_cifar10/EB-30-35.pth.tar \
--save ./baseline/resnet18_cifar10/pruned/EB-30-35