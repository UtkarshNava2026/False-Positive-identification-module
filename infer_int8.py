import cv2
import numpy as np
import onnxruntime as ort

MODEL = "/home/navavision/Utkarsh-Workspace/Quant/artifacts/gate/int8.onnx"
IMAGE = "/home/navavision/Utkarsh-Workspace/Quant/calibration/gate/193.jpg"
CLASSES = "/home/navavision/Utkarsh-Workspace/Quant/models/gate/classes.txt"

INPUT_SIZE = 640

CONF_THRES = 0.25
NMS_THRES = 0.45


with open(CLASSES, "r") as f:
    class_names = [x.strip() for x in f.readlines()]


session = ort.InferenceSession(
    MODEL,
    providers=["CPUExecutionProvider"],
)

input_name = session.get_inputs()[0].name


def preprocess(img):

    h, w = img.shape[:2]

    ratio = min(INPUT_SIZE / h, INPUT_SIZE / w)

    new_h = int(h * ratio)
    new_w = int(w * ratio)

    resized = cv2.resize(img, (new_w, new_h))

    padded = np.full(
        (INPUT_SIZE, INPUT_SIZE, 3),
        114,
        dtype=np.uint8,
    )

    padded[:new_h, :new_w] = resized

    blob = padded.transpose(2, 0, 1)
    blob = np.expand_dims(blob, axis=0)
    blob = blob.astype(np.float32)

    return blob, ratio


def multiclass_nms(boxes, scores, nms_thr):

    final_dets = []

    num_classes = scores.shape[1]

    for cls_ind in range(num_classes):

        cls_scores = scores[:, cls_ind]

        keep = cls_scores > CONF_THRES

        if keep.sum() == 0:
            continue

        valid_scores = cls_scores[keep]
        valid_boxes = boxes[keep]

        indices = cv2.dnn.NMSBoxes(
            valid_boxes.tolist(),
            valid_scores.tolist(),
            CONF_THRES,
            nms_thr,
        )

        if len(indices) > 0:

            for i in indices.flatten():

                final_dets.append(
                    [
                        *valid_boxes[i],
                        valid_scores[i],
                        cls_ind,
                    ]
                )

    return final_dets


def postprocess(outputs, ratio):

    predictions = outputs[0][0]

    grids = []
    expanded_strides = []

    strides = [8, 16, 32]

    hsizes = [INPUT_SIZE // s for s in strides]
    wsizes = [INPUT_SIZE // s for s in strides]

    for hsize, wsize, stride in zip(hsizes, wsizes, strides):

        xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))

        grid = np.stack((xv, yv), 2).reshape(1, -1, 2)

        grids.append(grid)

        shape = grid.shape[:2]

        expanded_strides.append(
            np.full((*shape, 1), stride)
        )

    grids = np.concatenate(grids, 1)
    expanded_strides = np.concatenate(expanded_strides, 1)

    predictions[..., :2] = (
        predictions[..., :2] + grids
    ) * expanded_strides

    predictions[..., 2:4] = np.exp(
        predictions[..., 2:4]
    ) * expanded_strides

    boxes = predictions[:, :4]

    scores = predictions[:, 4:5] * predictions[:, 5:]

    boxes_xyxy = np.zeros_like(boxes)

    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2

    boxes_xyxy /= ratio

    dets = multiclass_nms(
        boxes_xyxy,
        scores,
        NMS_THRES,
    )

    return dets


img = cv2.imread(IMAGE)

orig = img.copy()

blob, ratio = preprocess(img)

outputs = session.run(
    None,
    {
        input_name: blob
    },
)

dets = postprocess(outputs, ratio)

print("TOTAL DETECTIONS:", len(dets))

for det in dets:

    x1, y1, x2, y2, score, cls_id = det

    x1 = int(x1)
    y1 = int(y1)
    x2 = int(x2)
    y2 = int(y2)

    label = class_names[int(cls_id)]

    print(
        f"{label} {score:.3f} "
        f"({x1},{y1},{x2},{y2})"
    )

    cv2.rectangle(
        orig,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2,
    )

    cv2.putText(
        orig,
        f"{label} {score:.2f}",
        (x1, y1 - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
    )

cv2.imwrite("/home/navavision/Utkarsh-Workspace/Quant/output_int8_193.jpg", orig)

print("Saved output_int8_193.jpg")