import os
import numpy as np
import glob
import openpifpaf
import cv2
import PIL
import torch
import matplotlib.pyplot as plt
from math import sqrt, acos, degrees, atan, degrees


def angle_gor(a, b, c, d):
    ab = [a[0] - b[0], a[1] - b[1]]
    ab1 = [c[0] - d[0], c[1] - d[1]]
    cos = abs(ab[0] * ab1[0] + ab[1] * ab1[1]) / (sqrt(ab[0] ** 2 + ab[1] ** 2) * sqrt(ab1[0] ** 2 + ab1[1] ** 2))
    ang = acos(cos)
    return ang * 180 / np.pi


def sit_ang(a, b, c, d):
    ang = angle_gor(a, b, c, d)
    s1 = 0
    if ang != None:
        # print("Angle",ang)
        if ang < 120 and ang > 40:
            s1 = 1
    return s1


def sit_rec(a, b, c, d):
    ab = [a[0] - b[0], a[1] - b[1]]
    ab1 = [c[0] - d[0], c[1] - d[1]]
    l1 = sqrt(ab[0] ** 2 + ab[1] ** 2)
    l2 = sqrt(ab1[0] ** 2 + ab1[1] ** 2)
    s = 0
    if l1 != 0 and l2 != 0:
        # print(l1,l2, "---------->>>")
        if l2 / l1 >= 1.5:
            s = 1
    return s


def get_pose(image, net, device):
    standing_c = 0
    sitting_c = 0
    pil_im = PIL.Image.open(image).convert('RGB')
    # im = np.asarray(pil_im)

    openpifpaf.decoder.CifSeeds.threshold = 0.5
    openpifpaf.decoder.nms.Keypoints.keypoint_threshold = 0.2
    openpifpaf.decoder.nms.Keypoints.instance_threshold = 0.2
    processor = openpifpaf.decoder.factory_decode(net.head_nets, basenet_stride=net.base_net.stride)

    preprocess = openpifpaf.transforms.Compose([
        openpifpaf.transforms.NormalizeAnnotations(),
        openpifpaf.transforms.CenterPadTight(16),
        openpifpaf.transforms.EVAL_TRANSFORM,
    ])
    data = openpifpaf.datasets.PilImageList([pil_im], preprocess=preprocess)

    loader = torch.utils.data.DataLoader(
        data, batch_size=1, pin_memory=True,
        collate_fn=openpifpaf.datasets.collate_images_anns_meta)
    for images_batch, _, __ in loader:
        predictions = processor.batch(net, images_batch, device=device)[0]
        for pred in predictions:
            pred_json = pred.json_data()
            if pred_json["score"] < 0.4:
                continue
            kps = pred_json["keypoints"]
            kps = np.reshape(kps, (-1, 3))
            s = 0
            s1 = 0
            if (kps[11][:2] == [0, 0]).all() or (kps[12][:2] == [0, 0]).all() or (kps[13][:2] == [0, 0]).all() or (kps[14][:2] == [0, 0]).all() or (kps[15][:2] == [0, 0]).all() or (kps[16][:2] == [0, 0]).all():
                s += 1
                s1 += 1
            else:
                s += sit_rec(kps[12][:2], kps[14][:2], kps[14][:2],
                             kps[16][:2])
                s += sit_rec(kps[11][:2], kps[13][:2], kps[13][:2],
                             kps[15][:2])
                s1 += sit_ang(kps[12][:2], kps[14][:2], kps[14][:2],
                              kps[16][:2])
                s1 += sit_ang(kps[11][:2], kps[13][:2], kps[13][:2],
                              kps[15][:2])
            if s > 0 or s1 > 0:
                sitting_c += 1
            if s == 0 and s1 == 0:
                standing_c += 1
    return sitting_c, standing_c


def main():
    # device = torch.device('cpu')
    device = torch.device('cuda')  # if cuda is available

    net_cpu, _ = openpifpaf.network.factory(checkpoint='shufflenetv2k30w', download_progress=False)
    net = net_cpu.to(device)
    images = glob.glob("./images/*.jpg")
    sitting_inframe = np.zeros(len(images))
    for image in images:
        sitting_c, standing_c = get_pose(image, net, device)
        sitting_inframe[int(os.path.basename(image).replace(".jpg", "").split("_")[1])] = sitting_c
    # print(sitting_inframe)
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    frame_num = np.arange(len(sitting_inframe))
    ax.bar(frame_num, sitting_inframe)
    ax.set_ylabel('Count')
    ax.set_title('Number of persons sitting')
    # ax.set_xticks(frame_num)
    # ax.set_yticks(np.arange(0, 10, 2))
    # plt.show()
    plt.savefig("fig.jpg")


if __name__ == "__main__":
    main()