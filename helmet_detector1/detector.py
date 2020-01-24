

import cv2
import numpy as np
from cv2 import Algorithm

dataPath = "d://TranImages"
# bow训练数目
samples = 2


def path(cls, i):
    return "%s/%s%d.jpg" % (dataPath, cls, i + 1)


def get_flann_matcher():
    flann_params = dict(algorithm=1, trees=5)
    return cv2.FlannBasedMatcher(flann_params, {})


def get_bow_extractor(extractor, flann):
    return cv2.BOWImgDescriptorExtractor(extractor, flann)


def get_extract_detect():
    return cv2.xfeatures2d.SIFT_create(), cv2.xfeatures2d.SIFT_create()


def extract_sift(fn, extractor, detector):
    im = cv2.imread(fn, 0)
    return extractor.compute(im, detector.detect(im))[1]


# 提取BOW特性
def bow_features(img, extractor_bow, detector):
    return extractor_bow.compute(img, detector.detect(img))


def helmet_detector():
    # 创建所需对象
    pos, neg = "pos-", "neg-"
    detect, extract = get_extract_detect()
    matcher = get_flann_matcher()
    print("生成BOWKMEANS...")
    bow_kmean_trainer = cv2.BOWKMeansTrainer(1000)
    extract_bow = cv2.BOWImgDescriptorExtractor(extract, matcher)

    # 加入特征
    print("特征加入训练器")
    for i in range(samples):
        print(i)
        bow_kmean_trainer.add(extract_sift(path(pos, i), extract, detect))
        bow_kmean_trainer.add(extract_sift(path(neg, i), extract, detect))
    voc = bow_kmean_trainer.cluster()
    extract_bow.setVocabulary(voc)

    # 训练数据和类关联
    traindata, trainlabels = [], []
    print("添加训练数据...")
    for i in range(samples):
        print(i)
        traindata.extend(bow_features(cv2.imread(path(pos, i), 0), extract_bow, detect))
        trainlabels.append(1)
        traindata.extend(bow_features(cv2.imread(path(neg, i), 0), extract_bow, detect))
        trainlabels.append(-1)
    # 初始化SVM
    svm = cv2.ml.SVM_create()
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setGamma(0.5)
    svm.setC(30)
    svm.setKernel(cv2.ml.SVM_LINEAR)

    svm.train(np.array(traindata), cv2.ml.ROW_SAMPLE, np.array(trainlabels))
    return svm, extract_bow
