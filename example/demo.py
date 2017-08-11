# 第一步：原始图片为拿画板随便写的几个数字
import cv2
import selectivesearch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


def my_camera():
    # 从摄像头中取得视频
    cap = cv2.VideoCapture(0)

    # 获取视频播放界面长宽
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)

    # 定义编码器 创建 VideoWriter 对象
    # Be sure to use the lower case
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (width, height))

    while cap.isOpened():
        # 读取帧摄像头
        ret, frame = cap.read()
        if ret:
            # 输出当前帧
            # out.write(frame)
            cv2.imshow('My Camera', frame)

            # 键盘按 Q 退出
            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                break
        else:
            break

    cv2.imwrite("a.png", frame)
    # 释放资源
    # out.release()
    cap.release()
    cv2.destroyAllWindows()


def show_fig(img, data):
    figure, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
    ax.imshow(img)
    for x, y, w, h in data:
        rect = mpatches.Rectangle(
            (x, y), w, h, fill=False, edgecolor='red', linewidth=1)
        ax.add_patch(rect)
    plt.show()


def first_filter(area, regions):
    candidates = []
    for r in regions:
        x, y, w, h = r['rect']
        # 重复的不要
        if r['rect'] in candidates:
            continue

        # 太小和太大的不要
        if (w <= 5) or (h <= 5):
            continue
        if (w * h) > area:
            continue
        # if (w * h) < (area / 20):
        #     continue
        # 保留1
        if ((w * h) < (area / 20)) and ((h / w) < 3):
            continue

        candidates.append((x, y, w, h))
    return candidates


def second_filter(regions):
    candidates = []
    flag = np.ones([len(regions)])
    for index1, i in enumerate(regions):
        for index2, j in enumerate(regions):
            if (i != j) and (flag[index1] == 1):
                if i[0] >= j[0] and i[0] + i[2] <= j[0] + j[2] and i[1] >= j[1] and i[1] + i[3] <= j[1] + j[3]:
                    flag[index1] = 0
                    break
                if i[0] <= j[0] and i[0] + i[2] >= j[0] + j[2] and i[1] <= j[1] and i[1] + i[3] >= j[1] + j[3]:
                    flag[index1] = 1
                    flag[index2] = 0
    for i in range(len(flag)):
        if flag[i]:
            candidates.append(regions[i])
    return candidates


def order_number(img, data):
    h_list = [r[1] for r in data]
    aver = np.mean(h_list)
    v = np.var(h_list)
    print('行高均值：%.4f, 方差：%.4f' % (aver, v))
    flag = v > 500
    l1, l2 = [], []
    # 划分所属行
    if flag:
        for r in data:
            if r[1] <= aver:
                l1.append(r)
            else:
                l2.append(r)

        # 从左到右排列
        for i in range(len(l2)):
            for j in range(i + 1, len(l2)):
                if l2[i][0] > l2[j][0]:
                    l2[i], l2[j] = l2[j], l2[i]
    else:
        l1 = data
    for i in range(len(l1)):
        for j in range(i + 1, len(l1)):
            if l1[i][0] > l1[j][0]:
                l1[i], l1[j] = l1[j], l1[i]

    image_data = []
    image_data.extend(make_pic(img, l1))
    if flag:
        image_data.extend(make_pic(img, l2))
    return image_data


# 制作成28*28的黑底白字图片
def make_pic(img, data):
    padding = 4
    size = 20
    image_data = []
    for r in data:
        temp = np.zeros([28, 28], dtype='float32')
        if r[3] < r[2]:
            t = (r[2] - r[3]) // 2
            temp_img = img[(r[1]-t):(r[1]+r[3]+t), r[0]:(r[0]+r[2])]
        else:
            t = (r[3] - r[2]) // 2
            temp_img = img[r[1]:(r[1] + r[3]), (r[0]-t):(r[0] + r[2]+t)]

        temp_img = cv2.resize(temp_img, (size, size), cv2.INTER_CUBIC)
        temp_img = cv2.cvtColor(temp_img, cv2.COLOR_BGR2GRAY)
        temp_img = np.array(temp_img, dtype='float32') / 255
        # temp_img = np.array(cv2.threshold(temp_img, 0, 1, cv2.THRESH_BINARY)[1], dtype='float32')
        temp[padding:(padding+size), padding:(padding+size)] = temp_img
        image_data.append(temp)
        plt.imshow(temp, cmap="gray")
        plt.show()
    return image_data


def extract_images():
    # 第二步：执行搜索工具,展示搜索结果
    image_path = "./data/1.png"
    # 用cv2读取图片
    img = cv2.imread(image_path)
    # 白底黑字图 改为黑底白字图
    img = 255 - img

    # selectivesearch 调用selectivesearch函数 对图片目标进行搜索
    img_lbl, regions = selectivesearch.selective_search(
        img, scale=200, sigma=0.9, min_size=20)

    # print(regions[0])  # {'labels': [0.0], 'rect': (0, 0, 585, 301), 'size': 160699}  第一个为原始图的区域
    print("totally searched %d regions" % len(regions))  # 共搜索到199个区域

    # 接下来我们把窗口和图像打印出来，对它有个直观认识
    cv2.imshow('image', img)
    # 计算区域面积均值
    area = 0
    for reg in regions:
        x, y, w, h = reg['rect']
        area += w * h
    area /= len(regions)
    # 展示区域分布
    show_fig(img, [data['rect'] for data in regions])

    # 第三步：过滤掉冗余的窗口
    # 1）第一过滤
    candidates = first_filter(area, regions)
    print('after first filter left %d regions' % len(candidates))

    # 展示第一次过滤后的区域分布
    show_fig(img, candidates)

    # 2)第二次过滤 大圈套小圈的目标 只保留大圈
    num_array = second_filter(candidates)

    # 窗口过滤完之后的数量
    print('after second filter left %d regions' % len(num_array))

    # 3) 展示第二次过滤后的区域分布
    show_fig(img, num_array)

    # 第四步：将数字进行排序（从上往下，从左到右）
    image_data = order_number(img, num_array)
    # 保存结果
    np.save('./data/image_data.npy', image_data)


if __name__ == '__main__':
    extract_images()