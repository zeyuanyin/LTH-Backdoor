CUDA_VISIBLE_DEVICES=0 python mask_overlap_visualize.py \
--dataset cifar10 \
--depth 16 \
--arch vgg \
--percent 0.5 \
--path ./baseline/vgg16_cifar10/EB-30-35.pth.tar \
--path_1 ./backdoor_1/vgg16_cifar10/EB-30-35.pth.tar \
--path_2 ./backdoor_2/vgg16_cifar10/EB-30-35.pth.tar