import os


def file_name(file_dir='/home/zyyin/priml/CIFAR-10', p: int = 30):  #p=30/50/70

    EB_list = []
    for root, _, _ in os.walk(file_dir):

        if '0916X' in root:
            # print(root,':') #当前目录路径
            for _, _, files in os.walk(root):
                for f in files:
                    if 'EB-' + str(p) in f:
                        EB_list.append(root + '/' + f)

    # for i in EB_list:
    #     print(i)

    return EB_list


if __name__ == "__main__":
    file_name('/home/zyyin/priml/CIFAR-10', 30)
