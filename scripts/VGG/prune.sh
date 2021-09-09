CUDA_VISIBLE_DEVICES=0 python vggprune.py \
--dataset cifar10 \
--arch vgg \
--test-batch-size 256 \
--depth 16 \
--percent 0.3 \
--model ./baseline/vgg16_cifar10/EB-30-35.pth.tar \
--save ./baseline/vgg16_cifar10/pruned/EB-30-35