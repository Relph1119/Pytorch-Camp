# coding: utf-8
import torch
import torchvision.utils as vutils
import numpy as np
import torchvision.models as models
from torchvision import datasets
from tensorboardX import SummaryWriter

resnet18 = models.resnet18(False)
writer = SummaryWriter('../../Result/runs')
sample_rate = 44100
freqs = [262, 294, 330, 349, 392, 440, 440, 440, 440, 440, 440]

true_positive_counts = [75, 64, 21, 5, 0]
false_positive_counts = [150, 105, 18, 0, 0]
true_negative_counts = [0, 45, 132, 150, 150]
false_negative_counts = [0, 11, 54, 70, 75]
precision = [0.3333333, 0.3786982, 0.5384616, 1.0, 0.0]
recall = [1.0, 0.8533334, 0.28, 0.0666667, 0.0]

for n_iter in range(100):
    s1 = torch.rand(1)  # value to keep
    s2 = torch.rand(1)
    # add_scalar()：在一个图表中记录一个标量的变化，常用于 Loss 和 Accuracy 曲线的记录
    # data grouping by `slash`
    writer.add_scalar('data/scalar_systemtime', s1[0], n_iter)
    # data grouping by `slash`
    writer.add_scalar('data/scalar_customtime', s1[0], n_iter, walltime=n_iter)

    # add_scalars()：在一个图表中记录多个标量的变化，常用于对比，如 trainLoss 和 validLoss 的比较等。
    writer.add_scalars('data/scalar_group', {"xsinx": n_iter * np.sin(n_iter),
                                             "xcosx": n_iter * np.cos(n_iter),
                                             "arctanx": np.arctan(n_iter)}, n_iter)
    x = torch.rand(32, 3, 64, 64)  # output from network
    if n_iter % 10 == 0:
        # torchvision.utils.make_grid()：将一组图片拼接成一张图片，便于可视化。
        x = vutils.make_grid(x, normalize=True, scale_each=True)
        # add_image()：绘制图片，可用于检查模型的输入，监测 feature map 的变化，或是观察 weight。
        writer.add_image('Image', x, n_iter)  # Tensor
        # writer.add_image('astronaut', skimage.data.astronaut(), n_iter) # numpy
        # writer.add_image('imread',
        # skimage.io.imread('screenshots/audio.png'), n_iter) # numpy
        x = torch.zeros(sample_rate * 2)
        for i in range(x.size(0)):
            # sound amplitude should in [-1, 1]
            x[i] = np.cos(freqs[n_iter // 10] * np.pi *
                          float(i) / float(sample_rate))
        writer.add_audio('myAudio', x, n_iter)
        writer.add_text('Text', 'text logged at step:' + str(n_iter), n_iter)
        writer.add_text('markdown Text', '''a|b\n-|-\nc|d''', n_iter)

        # add_histogram()：绘制直方图和多分位数折线图，常用于监测权值及梯度的分布变化情况，便于诊断网络更新方向是否正确。
        for name, param in resnet18.named_parameters():
            if 'bn' not in name:
                writer.add_histogram(name, param, n_iter)

        # add_pr_curve()：绘制 PR 曲线
        writer.add_pr_curve('xoxo', np.random.randint(2, size=100), np.random.rand(
            100), n_iter)  # needs tensorboard 0.4RC or later
        # add_pr_curve_raw()：从原始数据上绘制 PR 曲线
        writer.add_pr_curve_raw('prcurve with raw data', true_positive_counts,
                                false_positive_counts,
                                true_negative_counts,
                                false_negative_counts,
                                precision,
                                recall, n_iter)

# export_scalars_to_json()：将 scalars 信息保存到 json 文件，便于后期使用
# export scalar data to JSON for external processing
writer.export_scalars_to_json("../../Result/all_scalars.json")

# add_embedding()：在三维空间或二维空间展示数据分布，可选 T-SNE、 PCA 和 CUSTOM 方法。
# 展示 Mnist 中的 100 张图片的三维数据分布
dataset = datasets.MNIST('../../Data/mnist', train=False, download=True)
images = dataset.data[:100].float()
label = dataset.targets[:100]
features = images.view(100, 784)
writer.add_embedding(features, metadata=label, label_img=images.unsqueeze(1))

writer.add_embedding(features, global_step=1, tag='noMetadata')
dataset = datasets.MNIST('../../Data/mnist', train=True, download=True)
images_train = dataset.data[:100].float()
labels_train = dataset.targets[:100]
features_train = images_train.view(100, 784)

all_features = torch.cat((features, features_train))
all_labels = torch.cat((label, labels_train))
all_images = torch.cat((images, images_train))
dataset_label = ['test'] * 100 + ['train'] * 100
all_labels = list(zip(all_labels, dataset_label))

writer.add_embedding(all_features, metadata=all_labels, label_img=all_images.unsqueeze(1),
                     metadata_header=['digit', 'dataset'], global_step=2)

#GRAPH
# add_graph()：绘制网络结构拓扑图
dummy_input = torch.rand(6, 3, 224, 224)
writer.add_graph(resnet18, dummy_input)

# VIDEO
vid_images = dataset.data[:16 * 48]
vid = vid_images.view(16, 1, 48, 28, 28)  # BxCxTxHxW

writer.add_video('video', vid_tensor=vid)
writer.add_video('video_1_fps', vid_tensor=vid, fps=1)

writer.close()
