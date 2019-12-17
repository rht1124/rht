'''
抠图
'''

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt


def get_image(path):
    # 获取图片
    img = cv2.imread(path)
    try:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except:
        print("error:",path)

    return img

def show(image):
    plt.imshow(image)
    plt.axis('off')
    plt.show()

def horizontal_flip(image, axis):
    # axis 0 垂直翻转，1水平翻转 ，-1水平垂直翻转，2不翻转，各自以25%的可能性
    if axis != 2:
        image = cv2.flip(image, axis)
    return image

#图片旋转
def rotation(image):
    h, w, c = image.shape
    box = [0, 0, w, h]
    center = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
    rot_mat = cv2.getRotationMatrix2D(center, -5, 1)
    img_rotated_by_alpha = cv2.warpAffine(image, rot_mat, (image.shape[1], image.shape[0]),borderValue = (255,255,255))
    return img_rotated_by_alpha

#随机翻转函数（未使用）
def random_flip(img,foreimg):
    axis1 = np.random.randint(low=-1, high=3)
    outimg = horizontal_flip(img,axis1)
    outforeimg = horizontal_flip(foreimg, axis1)

    return outimg,outforeimg

#抠图函数，注意：需要选出大致抠图位置
def GetgrabCut(img):
    height, width = img.shape[:2]
    #rect为选出的抠图位置
    rect = (0 + 10, 0 + 10, width - 10, height - 10)
    # rect = (275, 120, 170, 320)

    mask = np.zeros(img.shape[:2], np.uint8)
    bgModel = np.zeros((1, 65), np.float64)
    fgModel = np.zeros((1, 65), np.float64)
    # grabCut选择前景
    cv2.grabCut(img, mask, rect, bgModel, fgModel, 5, cv2.GC_INIT_WITH_RECT)
    # np.where（a,b,c）满足a为b，不满足为c，此处mask=0/2做前景
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype(np.uint8)
    # np.newaxis把矩阵增加一个维度
    out = img * mask2[:, :, np.newaxis]

    # cv2.imshow('output', out)
    # cv2.waitKey()
    return out

#对图片进行变换
def transImg(img):
    # img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    lower_blue = np.array([0, 0, 0])  # 获取最小阈值
    upper_blue = np.array([180, 255, 46])  # 获取最大阈值
    mask = cv2.inRange(img, lower_blue, upper_blue)  # 创建遮罩
    # show(mask)
    erode = cv2.erode(mask, None, iterations=3)  # 图像腐蚀
    dilate = cv2.dilate(erode, None, iterations=1)  # 图像膨胀
    opening = cv2.morphologyEx(dilate, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8)))  # 开运算

    return opening

#随机选择融合位置，center为融合图片左上角位置
def GetCenter(img,back_img):
    rows, cols, _ = img.shape
    brows, bcols, _ = back_img.shape
    # print(brows,bcols)
    r = np.random.uniform(0.5,0.9)# 在0.5-0.9范围内取随机数
    print('r',r)
    scale = min(r * bcols / cols, r * brows / rows)
    w=int(cols*scale)
    h=int(rows*scale)

    center = [np.random.randint(0, brows - h), np.random.randint(0, bcols - w)]
    # print(center)

    return center, scale

#融合图片函数
def fusionImg(img_path, back_path):

    img = get_image(img_path)
    backImg = get_image(back_path)

    rows, cols, channels = img.shape #y,x,_
    img = rotation(img)
    center,scale = GetCenter(img, backImg) #左上角为center
    print('scale',scale)
    resizeImg = cv2.resize(img,(int(scale*cols),int(scale*rows)),interpolation=cv2.INTER_CUBIC)
    # foreImg = GetgrabCut(resizeImg)  #GetgrabCut抠图函数,获得mask
    # # show(foreImg)
    # # rotaimg,rotafore = random_flip(resizeImg,foreImg)
    # # show(rotafore)
    #
    # transimg = transImg(foreImg) #使mask更加平滑？

    x=[]
    y=[]
    for i in range(int(scale*rows)):
        for j in range(int(scale*cols)):
            # if transimg[i, j] == 0:
            backImg[center[0] + i, center[1] + j] = resizeImg[i, j]
            y.append(center[0] + i)
            x.append(center[1] + j)
    # .sort()从小到大
    x.sort()
    y.sort()
    # cv2.rectangle(backImg, (x[0], y[0]), (x[-1], y[-1]), (0, 255, 0), 4)
    # show(backImg)
    if len(x) and len(y):
        box = [x[0], y[0],(x[-1]-x[0]),(y[-1]-y[0])]
    else:
        box = []

    return backImg,box

#融合函数
def prod(banner_dir,raw_dir,save_dir,save_file):
    banner_list = os.listdir(banner_dir)
    raw_list = os.listdir(raw_dir)
    os.makedirs(save_dir,exist_ok=True)
    n = 1
    with open(save_file,"w") as f:
        print("need cycle {0} times".format(max(len(banner_list),len(raw_list))))
        for i in range(max(len(banner_list),len(raw_list))):
            banner_name = banner_list[i%len(banner_list)]
            raw_name = raw_list[i%len(raw_list)]
            banner_path = os.path.join(banner_dir,banner_name)
            raw_path = os.path.join(raw_dir,raw_name)
            # print(banner_path,"\n",raw_path)

            try:
                prod_img,box = fusionImg(banner_path,raw_path)
                print('prod_img,box is below')


                if len(box):
                    filepath = os.path.join(save_dir,"%d.jpg" %n)
                    # print(banner_path)
                    # print(raw_path)
                    print(filepath)
                    print(box)
                    prod_img = cv2.cvtColor(prod_img, cv2.COLOR_RGB2BGR)
                    print(0)
                    cv2.imwrite(filepath, prod_img)
                    show(prod_img)
                    print(1)
                    # cv2.imshow('prod_img',prod_img)
                    # cv2.waitKey(0)
                    # if(box[2]>50 and box[3]>50):
                    f.write("{0}\t{1},{2},{3},{4},{5}\n".format(filepath,box[0],box[1],box[2],box[3],1))
                    n += 1
            except:
                print(banner_path,raw_path)

if __name__ == '__main__':
    # img =cv2.imread('./2dcode/2dcode_img1.jpg')
    # cv2.imshow('img',img)
    #前景
    # banner_dir = "I:/mapDetect/images/world_map/poisson_world"
    banner_dir = "./2dcode"
    # raw_dir = "I:/mapDetect/images/pur_book"
    # 后景
    # raw_dir = "I:/mapDetect/images/fusionImage"
    raw_dir = "./background"
    # 保存图片路径
    # save_dir = "I:/mapDetect/images/world_map/fussion_world"
    save_dir = "./mix_img"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # 保存目标位置信息
    save_file = "./annotions.txt"

    prod(banner_dir, raw_dir, save_dir, save_file)


    # img_path = "I:/mapDetect/images/30.jpg"
    # back_path = "I:/mapDetect/fusionImage/all_souls_000005.jpg"
    #
    # fusionImg(img_path,back_path)



