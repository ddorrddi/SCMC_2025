#!/usr/bin/python3


import os
import sys
from typing import Tuple, Union, List

import torch
import cv2
from torchvision.transforms import ToTensor
import numpy as np
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
from delivery.msg import DetectYolo

import torch.nn as nn

import time

import torchvision

def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def rescale(ori_shape: Tuple[int, int], boxes: Union[torch.Tensor, np.ndarray],
            target_shape: Tuple[int, int]):
    """Rescale the output to the original image shape
    :param ori_shape: original width and height [width, height].
    :param boxes: original bounding boxes as a torch.Tensor or np.array or shape
        [num_boxes, >=4], where the first 4 entries of each element have to be
        [x1, y1, x2, y2].
    :param target_shape: target width and height [width, height].
    """
    xscale = target_shape[1] / ori_shape[1]
    yscale = target_shape[0] / ori_shape[0]

    boxes[:, [0, 2]] *= xscale
    boxes[:, [1, 3]] *= yscale

    return boxes

def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


def attempt_load(weights, map_location):
    # Loads an ensemble of models weights=[a,b,c] or a single model weights=[a] or weights=a
    # weights = Path(str(weights).strip().replace("'", '').lower())
    ckpt = torch.load(weights, map_location=map_location)  # load
    model = ckpt['ema' if ckpt.get('ema') else 'model'].float().fuse().eval()  # FP32 model
    
    # Compatibility updates
    for m in model.modules():
        if type(m) in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
            m.inplace = True  # pytorch 1.7.0 compatibility
        elif type(m) is nn.Upsample:
            m.recompute_scale_factor = None  # torch 1.11.0 compatibility
        elif type(m) is Conv:
            m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility
    
    return model

def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)


def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False,
                        labels=()):
    """Runs Non-Maximum Suppression (NMS) on inference results

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """

    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_det = 300  # maximum number of detections per image
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            l = labels[xi]
            v = torch.zeros((len(l), nc + 5), device=x.device)
            v[:, :4] = l[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(l)), l[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        if nc == 1:
            x[:, 5:] = x[:, 4:5] # for models with one class, cls_loss is 0 and cls_conf is always 0.5,
                                 # so there is no need to multiplicate.
        else:
            x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break  # time limit exceeded

    return output

def get_random_color(seed):
    gen = np.random.default_rng(seed)
    color = tuple(gen.choice(range(256), size=3))
    color = tuple([int(c) for c in color])
    return color

def draw_detections_delivery_sign(img: np.array, bboxes: List[List[int]], classes: List[int],
                    class_labels: Union[List[str], None]):
    for bbox, cls in zip(bboxes, classes):
        if cls == 14 or cls == 16 or cls == 17 or cls == 18 or cls == 19 or cls == 20:
            x1, y1, x2, y2 = bbox
            # print("cls : ", cls, "classes : ", classes)
            color = get_random_color(int(cls))
            img = cv2.rectangle(
                img, (int(x1), int(y1)), (int(x2), int(y2)), color, 3
            )
            
            # print(class_labels)
            if class_labels:
                label = class_labels[int(cls)]

                x_text = int(x1)
                y_text = max(15, int(y1 - 10))
                img = cv2.putText(
                    img, label, (x_text, y_text), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, color, 1, cv2.LINE_AA
                )

    return img


class YoloV7:
    def __init__(self):
        self.conf_thresh = 0.9
        self.iou_thresh = 0.45
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.weights_path = '/home/foscar/SCMC_2025/src/camera/delivery/weight/delivery2400.pt'
        self.model = attempt_load(self.weights_path, map_location=self.device)
        self.model.eval()

    @torch.no_grad()
    def inference(self, img: torch.Tensor):
        """
        :param img: tensor [c, h, w]
        :returns: tensor of shape [num_boxes, 6], where each item is represented as
            [x1, y1, x2, y2, confidence, class_id]
        """
        img = img.unsqueeze(0)
        pred_results = self.model(img)[0]
        detections = non_max_suppression(
            pred_results, conf_thres=self.conf_thresh, iou_thres=self.iou_thresh
        )
        if detections:
            detections = detections[0]
        return detections


class Yolov7Publisher:
    def __init__(self):
        """
            A1 - 18, A2 - 19, A3 - 20
            B1 - 14, B2 - 16, B3 - 17
        """
        self.sign_mapping = {
            "A1": "1", "A2": "2", "A3": "3",
            "B1": "1", "B2": "2", "B3": "3"
        }

        self.class_labels = ["None", "None", "None", "None", "None", "None", "None", "None", "None", "None", "None", "None", "None", "None",
                             "B1", "None", "B2", "B3", "A1", "A2", "A3"]
        self.img_size = (640, 480)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.visualization_publisher = rospy.Publisher(
            "/delivery_sign_yolo_result", Image, queue_size=1
        )

        # === 수정된 부분 1: Publisher 변경 ===
        # 토픽 이름을 더 명확하게 바꾸고, 메시지 타입을 새로 만든 DetectedObject로 변경
        self.delivery_sign_publisher = rospy.Publisher(
            "/delivery_sign_info", DetectYolo, queue_size=1
        )
        # ====================================

        self.bridge = CvBridge()
        self.tensorize = ToTensor()
        self.model = YoloV7()
        self.img_subscriber = rospy.Subscriber(
            "/usb_cam_right/image_raw", Image, self.process_img_msg
        )

        self.delivery_target = None
        self.delivery_target_subscriber = rospy.Subscriber("/delivery_target", String, self.delivery_target_callback, queue_size=1)

    def delivery_target_callback(self, msg):
        self.delivery_target = msg.data

    def process_img_msg(self, img_msg):
        """ callback function for publisher """
        np_img_orig = self.bridge.imgmsg_to_cv2(
            img_msg, desired_encoding='bgr8'
        )
        if len(np_img_orig.shape) == 2:
            np_img_orig = np.stack([np_img_orig] * 3, axis=2)
        h_orig, w_orig, c = np_img_orig.shape
        w_scaled, h_scaled = self.img_size
        np_img_resized = cv2.resize(np_img_orig, (w_scaled, h_scaled))

        img = np_img_resized.transpose((2, 0, 1))[::-1]
        img = torch.from_numpy(np.ascontiguousarray(img))
        img = img.float() / 255.0
        img = img.to(self.device)

        detections = self.model.inference(img)
        
        # 시각화 이미지를 먼저 만듭니다.
        # detections가 비어있을 경우를 대비합니다.
        vis_img = np_img_orig.copy() # 원본 이미지 복사
        if detections is not None and len(detections) > 0:
            detections[:, :4] = rescale(
                [h_scaled, w_scaled], detections[:, :4], [h_orig, w_orig]
            )
            detections[:, :4] = detections[:, :4].round()
            
            bboxes = [[int(x) for x in box] for box in detections[:, :4].tolist()]
            classes = [int(c) for c in detections[:, 5].tolist()]

            vis_img = draw_detections_delivery_sign(vis_img, bboxes, classes, self.class_labels)

        vis_msg = self.bridge.cv2_to_imgmsg(vis_img, encoding="bgr8")
        self.visualization_publisher.publish(vis_msg)


        # === 수정된 부분 2: 메시지 발행 로직 변경 (타겟 우선 매칭) ===
        if detections is None or len(detections) == 0:
            msg = DetectYolo()
            msg.class_name = "None"
            msg.center_x = 0.0
            msg.center_y = 0.0
            msg.width = 0.0
            msg.height = 0.0
            self.delivery_sign_publisher.publish(msg)
            return

        # 좌표를 원본 해상도로 되돌림
        detections[:, :4] = rescale([h_scaled, w_scaled], detections[:, :4], [h_orig, w_orig])
        detections[:, :4] = detections[:, :4].round()

        rows = detections.tolist()

        # 관심 클래스만 유지 (B1:14, B2:16, B3:17, A1:18, A2:19, A3:20)
        allow_ids = {14, 16, 17, 18, 19, 20}
        candidates = []
        for r in rows:
            x1, y1, x2, y2, conf, cls_id = r
            cls_id = int(cls_id)
            if cls_id not in allow_ids:
                continue
            label = self.class_labels[cls_id]  # "A1"/"B2"/...
            if label not in self.sign_mapping:
                continue
            mapped_num = self.sign_mapping[label]  # "1"/"2"/"3"
            candidates.append({
                "row": r,
                "label": label,
                "num": mapped_num,
                "conf": float(conf),
                "y1": float(y1)
            })

        if not candidates:
            msg = DetectYolo()
            msg.class_name = "None"
            msg.center_x = 0.0
            msg.center_y = 0.0
            msg.width = 0.0
            msg.height = 0.0
            self.delivery_sign_publisher.publish(msg)
            return

        selected = None
        # 타겟이 지정된 경우: 타겟과 라벨(A1/B2/...) 또는 매핑숫자(1/2/3)가 같은 것만 선택
        if self.delivery_target and self.delivery_target not in ("NONE", ""):
            tgt = self.delivery_target  # 이미 upper()
            matches = [c for c in candidates if (c["label"] == tgt or c["num"] == tgt)]
            if matches:
                # 여러 개면 conf 최고 선택
                selected = max(matches, key=lambda c: c["conf"])
            else:
                # 타겟이 있지만 해당 라벨이 안 보임 -> None 발행
                msg = DetectYolo()
                msg.class_name = "None"
                msg.center_x = 0.0
                msg.center_y = 0.0
                msg.width = 0.0
                msg.height = 0.0
                self.delivery_sign_publisher.publish(msg)
                return
        else:
            # 타겟이 없으면 기존 정책 유지: 화면 맨 위(y1 최소) 1개 선택
            selected = min(candidates, key=lambda c: c["y1"])

        # 선택된 박스를 메시지로 변환해 publish
        x1, y1, x2, y2, conf, cls_id = selected["row"]
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        w  = (x2 - x1)
        h  = (y2 - y1)

        msg = DetectYolo()
        # class_name은 downstream(Fusion/Planner)에서 숫자 비교하기 쉽게 "1"/"2"/"3"으로 유지
        msg.class_name = selected["num"]
        msg.center_x = float(cx)
        msg.center_y = float(cy)
        msg.width    = float(w)
        msg.height   = float(h)
        self.delivery_sign_publisher.publish(msg)
        # =========================================


if __name__ == "__main__":
    rospy.init_node("delivery_sign_yolo")
    publisher = Yolov7Publisher()
    rospy.spin()