# 本次更新内容：
# version: 1.0.8
# 2、修改draw_confusion_mat函数，用来处理不输入模型，只输入真是标签和预测标签的情况。
# 3、修改de_to_one_hot_auto函数，支持如果此时是非onehot形式就不转变。


# version: 1.0.7
# 1、draw_result函数，绘制loss与acc时 对多分类的每一个值都绘制Acc时，图片文件保存为 acc_small.jpg。
# 2、draw_result函数，修改了函数中的acc的值，acc=history['sparse_categorical_accuracy']修改为acc=history['acc']

# version: 1.0.6
# 1、绘制loss与acc时，添加上仅有训练集没有测试集的绘图代码，并且给出选项，是否要对多分类的每一个值都绘制Acc。
# 2、绘制Se，Sp，+p，Acc的图
# 3、更改所有图片为600dpi,并且给所有保存图片的函数，添加了：picture_format字段，用来设置保存图片的格式

# version: 1.0.5
# 1、在draw_confusion_mat()函数中添加了是否归一化以及是否归一化与非归一化混淆矩阵都画的参数，
#               需求说明：因为绘制归一化之后的混淆矩阵数据保存不全
# 2、在1、的需求上有对两张混淆矩阵进行了保存是名字的修改，分别保存成"混淆矩阵.jpg"和"混淆矩阵_归一化.jpg"

# version：1.0.4
# 1、在draw_confusion_mat()函数中加入了try-except AttributeError，因为model.predict_classes()使用api创建模型时可能没有该函数

# version:1.0.3
# 1、help（）函数中加入      print('#de_to_one_hot_auto(labels)')
# 2、draw_result()函数中如果savepath=None，则不保存，只绘制


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
import random
import sys
from myselfutilsyjl import plot_draw_metric
import pprint


def help():
    print('#draw_result(history, savepath)')
    print('#draw_confusion_mat(model, test_features, test_labels, classes, savepath)')
    print('#to_one_hot(labels, dimension=2, begin=1)')
    print('#de_to_one_hot_3dim(labels)')
    print('#de_to_one_hot(labels)')
    print('#de_to_one_hot_auto(labels)')
    print('#random_split(all_num, train_ratio, validation_ratio, test_ratio)')


# 绘制loss以及acc的图
def draw_result(history, small_class=None, picture_format=".jpg", savepath=None, color_set=None):
    acc = history['acc']
    loss = history['loss']
    # 是否有验证集
    if "val_acc" in history.keys() and "val_loss" in history.keys():
        val_acc = history['val_acc']
        val_loss = history['val_loss']
        val_flag = True
    else:
        val_flag = False

    epochs = range(1, len(acc) + 1)
    plt.figure()
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.plot(epochs, acc, 'b', label='Training acc')
    # 如果给小类赋值，则将小类的Acc绘制出来
    if small_class:
        # 备选的几种绘图颜色，如果绘制小类的正确率，则需要创建一个颜色列表,默认的颜色列表为5中，超出则报错，也可自己输入颜色列表。
        # 颜色网址：https://blog.csdn.net/syyyy712/article/details/87426927
        if color_set == None and len(small_class) <= 5:
            color_set = ["springgreen", "deeppink", "blueviolet", "skyblue", "orangered"]
        elif color_set == None and len(small_class) > 5:
            print("请输入颜色列表！！！")
            # 主动退出程序
            sys.exit(1)
        elif color_set != None and len(small_class) > len(color_set):
            print("颜色列表的长度小于分类类别的个数！！！")
            # 主动退出程序
            sys.exit(1)
        for index, (key, value) in enumerate(small_class.items()):
            plt.plot(epochs, value, color_set[index], label=key)

    if val_flag:
        plt.plot(epochs, val_acc, 'r', label='Validation acc')
        plt.title('Training and validation accuracy')
    else:
        plt.title('Training accuracy')
    plt.legend()
    if savepath is not None:
        plt.savefig(savepath + ('acc_small' if small_class else "acc") + picture_format, dpi=600, bbox_inches='tight')

    plt.figure()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.plot(epochs, loss, 'b', label='Training loss')
    if val_flag:
        plt.plot(epochs, val_loss, 'r', label='Validation loss')
        plt.title('Training and validation loss')
    else:
        plt.title('Training loss')
    plt.legend()
    if savepath is not None:
        plt.savefig(savepath + 'loss' + picture_format, dpi=600, bbox_inches='tight')

    plt.show()


# 绘制混淆矩阵(用归一化之后的数据绘图)
# 此处的plt.cm.Blues的具体颜色映射参考网址：https://matplotlib.org/gallery/color/colormap_reference.html
def plot_confusion_matrix(cm, classes, title, savepath, picture_format=".jpg", normalize=False, cmap=plt.cm.Blues):
    img_name = "混淆矩阵"
    if normalize:
        img_name = "混淆矩阵_归一化"
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    # plt.xticks(tick_marks, classes, rotation=45)
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    plt.tick_params(axis='both', which='both', bottom=False, left=False)
    # plt.tick_params(axis='y', which='both', bottom=False)
    thresh = cm.max() / 2.0
    if normalize:
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, '{:>.4f}'.format(cm[i, j]), horizontalalignment='center',
                     color='white' if cm[i, j] > thresh else 'black')
    else:
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i, j], horizontalalignment='center',
                     color='white' if cm[i, j] > thresh else 'black')
    # plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predict label')
    plt.savefig(savepath + img_name + picture_format, dpi=600, bbox_inches='tight')
    plt.show()


# 整理绘制混淆矩阵所需的数据（当model为None时，模式即为只输入预测值和真实值，预测值输入到test_features,中）
def draw_confusion_mat(model, test_features, test_labels, classes, savepath, picture_format=".jpg", normalize=True,
                       istwo=True):
    """
    Args:
        model:              模型，如果设置成None，表示直接输入预测值和真实值
        test_features:      数据集x，如果模型设置成None，则该值为预测值 = model(x)
        test_labels:        数据集x对应的标签
        classes:            类别：e.g.   classes=['N', 'S', 'V', 'F', 'Q'],
        savepath:           存储路径
        picture_format:     图片格式
        normalize:          混淆矩阵中的值是百分比还是实际的值
        istwo:              是否存储两张混淆矩阵（百分比和实际的值）

    Returns:                直接输出图片，无返回值

    """
    if model == None:
        test_labels_predict = test_features
    else:
        try:
            test_labels_predict = model.predict_classes(test_features)  # shape(1094,20)
        except AttributeError:
            temp_test_predict = model.predict(test_features)
            test_labels_predict = np.argmax(temp_test_predict, axis=-1)  # shape(1094,20)
        test_labels_predict = test_labels_predict.reshape((-1, 1)).astype(np.int32)  # shape(21800,1)int型,范围为(0-4)
    test_labels_true = de_to_one_hot_auto(test_labels).reshape((-1, 1)).astype(np.int32)  # shape(21800,1)int型,范围为(0-4)

    # 画混淆矩阵
    confusion_mat = confusion_matrix(test_labels_true, test_labels_predict)
    # self.plot_confusion_matrix(confusion_mat, classes=range(5), title=f'Confusion matrix {class_dict}', normalize=True)
    plot_confusion_matrix(confusion_mat, classes, savepath=savepath, picture_format=picture_format,
                          title=f'Confusion matrix', normalize=normalize)
    if istwo:
        plot_confusion_matrix(confusion_mat, classes, savepath=savepath, picture_format=picture_format,
                              title=f'Confusion matrix', normalize=not normalize)


# 评价指标的计算
def calc_metrics(cm, num_class):
    metrics_dict = {}
    Recall = np.empty(shape=((0, num_class)))
    Specificity = np.empty(shape=((0, num_class)))
    Precision = np.empty(shape=((0, num_class)))
    Acc = np.empty(shape=((0, num_class)))
    support = np.empty(shape=((0, num_class)))

    for i in range(num_class):
        TP = cm[i, i]
        FP = np.sum(cm[:, i]) - cm[i, i]
        TN = np.sum(cm) - np.sum(cm[i, :]) - np.sum(cm[:, i]) + cm[i, i]
        FN = np.sum(cm[i, :]) - cm[i, i]

        Recall = np.append(Recall, TP / (TP + FN))
        Specificity = np.append(Specificity, TN / (TN + FP))
        Precision = np.append(Precision, TP / (TP + FP))
        Acc = np.append(Acc, (TP + TN) / (TP + TN + FP + FN))
        support = np.append(support, np.sum(cm[i, :]))

    metrics_dict["Recall"] = Recall
    metrics_dict["Specificity"] = Specificity
    metrics_dict["Precision"] = Precision
    metrics_dict["Acc"] = Acc
    metrics_dict["support"] = support

    metrics_array = np.vstack((Recall, Specificity, Precision, Acc, support)).T
    return metrics_array, metrics_dict


# 绘制评价指标图
def draw_metrics(model, test_features, test_labels, classes, savepath, picture_format=".jpg",
                 title='Classification report', cmap='YlGnBu', print_metrics=False):
    """
    Args:
        model:              模型，如果设置成None，表示直接输入预测值和真实值
        test_features:      数据集x，如果模型设置成None，则该值为预测值 = model(x)
        test_labels:        数据集x对应的标签
        classes:            类别：e.g.   classes=['N', 'S', 'V', 'F', 'Q'],
        savepath:           存储路径
        picture_format:     图片格式
        title:              图片标题
        cmap:               颜色带
        print_metrics:      是否打印出评价指标

    Returns:                直接输出图片，无返回值

    """
    if model == None:
        test_labels_predict = test_features
    else:
        try:
            test_labels_predict = model.predict_classes(test_features)  # shape(1094,20)
        except AttributeError:
            temp_test_predict = model.predict(test_features)
            test_labels_predict = np.argmax(temp_test_predict, axis=-1)  # shape(1094,20)
        test_labels_predict = test_labels_predict.reshape((-1, 1)).astype(np.int32)  # shape(21800,1)int型,范围为(0-4)
    test_labels_true = de_to_one_hot_auto(test_labels).reshape((-1, 1)).astype(np.int32)  # shape(21800,1)int型,范围为(0-4)
    # 画混淆矩阵
    confusion_mat = confusion_matrix(test_labels_true, test_labels_predict)
    # 计算评价指标
    metrics_array, metrics_dict = calc_metrics(cm=confusion_mat, num_class=len(classes))
    if print_metrics:
        print(pprint.pformat(metrics_dict))
    plot_draw_metric.plot_classification_report(classification_report=metrics_array,
                                                savepath=savepath,
                                                picture_format=picture_format,
                                                class_names=classes,
                                                title=title,
                                                cmap=cmap)


# 将样本转为one_hot形式，labels为（seample，1）或（seample,),dimension默认为2，也就是转成one_hot之后会有几列（几类）
# begin为labels中开始的下表，例如，labels：（1,2,3,1,3,2,1,2）其中为3分类的话dimension为3，而begin为1.
def to_one_hot(labels, dimension=2, begin=1):
    results = np.zeros((len(labels), dimension))
    for i, label in enumerate(labels):
        results[i, label - begin] = 1
    return results


# 将one_hot形式转化为非one_hot的形式，也即是从（seample,5)转化为(seample)此处的函数为处理三维转二维的，
# 也即是（seample/20,20,5)转化为（seample/20,20)
def de_to_one_hot_3dim(labels):
    if len(labels.shape) != 3:
        print('de_to_one_hot_3dim此方法仅适用于三维转二维')
        exit()
    results = np.zeros((labels.shape[:-1]))
    for i in range(labels.shape[0]):
        for j in range(labels.shape[1]):
            results[i][j] = np.argwhere(labels[i][j] == 1).ravel()[0]
    return results


# 将one_hot形式转化为非one_hot的形式，也即是从（seample,5)转化为(seample,)，
def de_to_one_hot(labels):
    if len(labels.shape) != 2:
        print('de_to_one_hot此方法仅适用于二维转一维')
        exit()
    results = np.zeros((labels.shape[0],))
    for i in range(labels.shape[0]):
        results[i] = np.argwhere(labels[i] == 1).ravel()[0]
    return results


# 可以支持三维转二维的或者二维转一维的
def de_to_one_hot_auto(labels):
    if len(labels.shape) == 2:
        return de_to_one_hot(labels)
    elif len(labels.shape) == 3:
        return de_to_one_hot_3dim(labels)
    elif len(labels.shape) == 1:
        return labels
    else:
        print("目前仅支持三维转二维或者二维转一维!")
        exit()


# 做随机分割样本，all_num为样本总数，train_ratio为要分出训练集的比例，例如（124000,7,2,1）
# 返回值分别为训练集，测试集，验证集在all_num这样一个列表中的下表值
# 比如:(10,6,2,2)——>[2,5,6,8,1,9],[4,3],[7,0]
def random_split(all_num, train_ratio, validation_ratio, test_ratio):
    if train_ratio + validation_ratio + test_ratio != 10:
        print("请输入正确的比例，比例之和为10")
        exit()
    temp = list(range(all_num))
    random.shuffle(temp)
    return temp[:int(train_ratio / 10 * all_num)], temp[int(train_ratio / 10 * all_num):int(
        (train_ratio + validation_ratio) / 10 * all_num)], temp[int(-test_ratio / 10 * all_num):]
