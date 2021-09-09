CUDA_VISIBLE_DEVICES=0 python main.py \
--dataset cifar10 \
--arch vgg \
--depth 16 \
--lr 0.1 \
--epochs 160 \
--schedule 80 120 \
--batch-size 256 \
--test-batch-size 256 \
--save ./baseline/vgg16_cifar10 \
--momentum 0.9 \
--sparsity-regularization \
--poison_on 1 \
--poison_method 1