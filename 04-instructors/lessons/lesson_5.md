---
marp: true
theme: dsi-certificates-theme
_class: invert
paginate: true
---

# Deep Learning

## Convolutional Neural Networks - Part II

Adapted from material by Charles Ollion & Olivier Grisel

---
# CNNs for computer Vision

.center[
          <img src="./images/vision.png" style="width: 600px;" />
]

---
# Beyond Image Classification

### CNNs
- Previous lecture: image classification

---

### Limitations
- Mostly on centered images
- Only a single object per image
- Not enough for many real world vision tasks

---
# Beyond Image Classification


.center[
          <br/>
          <img src="./images/cls_1.png" style="width: 800px;" />
]

---
# Beyond Image Classification


.center[
          <br/>
          <img src="./images/cls_2.png" style="width: 800px;" />
]
---
# Beyond Image Classification

.center[
          <br/>
          <img src="./images/cls_3.png" style="width: 800px;" />
]
---
# Beyond Image Classification


.center[
          <br/>
          <img src="./images/cls_4_2.png" style="width: 800px;" />
]
---
# Beyond Image Classification

.center[
          <br/>
          <img src="./images/cls_4_3.png" style="width: 800px;" />
]
---
# Outline

* Simple Localisation as regression
* Detection Algorithms
* Fully convolutional Networks
* Semantic & Instance Segmentation

---

# Localisation

---

# Localisation

.center[
          <img src="./images/dog.jpg" style="width: 400px;" />
]

---

- Single object per image
- Predict coordinates of a bounding box `(x, y, w, h)`
- Evaluate via Interection over Union (IoU)

---

# Localisation as regression

.center[
          <img src="./images/regression_dog_1.svg" style="width: 700px;" />
]

---

# Localisation as regression

.center[
          <img src="./images/regression_dog.svg" style="width: 700px;" />
]

---

# Classification + Localisation

.center[
          <img src="./images/doublehead_1.svg" style="width: 600px;" />
]

---

# Classification + Localisation

.center[
          <img src="./images/doublehead.svg" style="width: 600px;" />
]

---

- Use a pre-trained CNN on ImageNet (ex. ResNet)
- The "localisation head" is trained seperately with regression
- Possible end-to-end finetuning of both tasks
- At test time, use both heads

---

# Classification + Localisation

.center[
          <img src="./images/doublehead.svg" style="width: 600px;" />
]

$C$ classes, $4$ output dimensions ($1$ box)

---

**Predict exactly $N$ objects:** predict $(N \times 4)$ coordinates and $(N \times K)$ class scores

---

# Object detection

We don't know in advance the number of objects in the image. Object detection relies on *object proposal* and *object classification*

**Object proposal:** find regions of interest (RoIs) in the image
**Object classification:** classify the object in these regions

---

### Two main families:

- Single-Stage: A grid in the image where each cell is a proposal (SSD, YOLO, RetinaNet)
- Two-Stage: Region proposal then classification (Faster-RCNN)

---

# YOLO
.center[
          <img src="./images/yolo0.png" style="width: 500px;" />
]

.footnote.small[
Redmon, Joseph, et al. "You only look once: Unified, real-time object detection." CVPR (2016)
]

---

For each cell of the $S \times S$ predict:
- $B$ **boxes** and **confidence scores** $C$ ($5 \times B$ values) + **classes** $c$

---

# YOLO
.center[
          <img src="./images/yolo1.png" style="width: 500px;" />
]

For each cell of the $S \times S$ predict:
- $B$ **boxes** and **confidence scores** $C$ ($5 \times B$ values) + **classes** $c$

.footnote.small[
Redmon, Joseph, et al. "You only look once: Unified, real-time object detection." CVPR (2016)
]
---

# YOLO
.center[
          <img src="./images/yolo1.png" style="width: 500px;" />
]

Final detections: $C_j * prob(c) > \text{threshold}$

.footnote.small[
Redmon, Joseph, et al. "You only look once: Unified, real-time object detection." CVPR (2016)
]

---

# YOLO

.footnote.small[
Redmon, Joseph, et al. "You only look once: Unified, real-time object detection." CVPR (2016)
]

- After ImageNet pretraining, the whole network is trained end-to-end
- The loss is a weighted sum of different regressions

---

.center[
          <img src="./images/yolo_loss.png" style="width: 400px;" />
]

---

# Box Proposals

Instead of having a predefined set of box proposals, find them on the image:
- **Selective Search** - from pixels (not learnt, no longer used)
- **Faster - RCNN** - Region Proposal Network (RPN)

.footnote.small[
Girshick, Ross, et al. "Fast r-cnn." ICCV 2015
]

---

**Crop-and-resize** operator (**RoI-Pooling**):
- Input: convolutional map + $N$ regions of interest
- Output: tensor of $N \times 7 \times 7 \times \text{depth}$ boxes
- Allows to propagate gradient only on interesting regions, and efficient computation

---

# Faster-RCNN

.center[
          <img src="./images/fasterrcnn.png" style="width: 270px;" />
]

.footnote.small[
Ren, Shaoqing, et al. "Faster r-cnn: Towards real-time object detection with region proposal networks." NIPS 2015
]

---

- Train jointly **RPN** and other head
- 200 box proposals, gradient propagated only in positive boxes
- Region proposal is translation invariant, compared to YOLO
- Region proposal input is a fully convolutional network: shares weights across spatial dimensions

---
## Measuring performance

.center[
          <img src="./images/sotaresults2.png" style="width: 720px;" />
]

Measures: mean Average Precision **mAP** at given **IoU** thresholds

.footnote.small[
Zeming Li et al. Light-Head R-CNN: In Defense of Two-Stage Object Detector 2017
]

---

- AP @0.5 for class "cat": average precision for the class, where $IoU(box^{pred}, box^{true}) > 0.5$

---

## State-of-the-art

.center[
          <img src="./images/sotaresults3.png" style="width: 600px;" />
]

.footnote.small[
Ghiasi G. et al. Simple Copy-Paste is a Strong Data Augmentation Method for Instance Segmentation, 2020
]

---

- Larger image sizes, larger and better models, better augmented data
- https://paperswithcode.com/sota/object-detection-on-coco


---

# Segmentation

---

# Segmentation

Output a class map for each pixel (here: dog vs background)

.center[
          <img src="./images/dog_segment.jpg" style="width: 400px;" />
]

---

- **Instance segmentation**: specify each object instance as well (two dogs have different instances)
- This can be done through **object detection** + **segmentation**

---

# Convolutionize

.center[
          <img src="./images/convolutionalization.png" style="width: 400px;" />
]


.footnote.small[
Long, Jonathan, et al. "Fully convolutional networks for semantic segmentation." CVPR 2015
]

- Slide the network with an input of `(224, 224)` over a larger image. Output of varying spatial size
- **Convolutionize**: change Dense `(4096, 1000)` to $1 \times 1$ Convolution, with `4096, 1000` input and output channels
- Gives a coarse **segmentation** (no extra supervision)

---

# Fully Convolutional Network

.center[
          <img src="./images/densefc.png" style="width: 500px;" />
]

.footnote.small[
Long, Jonathan, et al. "Fully convolutional networks for semantic segmentation." CVPR 2015
]

---

- Predict / backpropagate for every output pixel
- Aggregate maps from several convolutions at different scales for more robust results

---

## Mask-RCNN

.center[
          <img src="./images/maskrcnn.png" style="width: 760px;" />
]

.footnote.small[
K. He and al. Mask Region-based Convolutional Network (Mask R-CNN) NIPS 2017
]

---

Faster-RCNN architecture with a third, binary mask head

---

# Results

.center[
          <img src="./images/maskrcnnresults.png" style="width: 760px;" />
]

.footnote.small[
K. He and al. Mask Region-based Convolutional Network (Mask R-CNN) NIPS 2017
]

---

- Mask results are still coarse (low mask resolution)
- Excellent instance generalization

---

## Results

.center[
          <img src="./images/maskrcnnresults2.png" style="width: 760px;" />
]

.footnote.small[
He, Kaiming, et al. "Mask r-cnn." Internal Conference on Computer Vision (ICCV), 2017.
]

---

## State-of-the-art & links

Most benchmarks and recent architectures are reported here:

.center[
https://paperswithcode.com/area/computer-vision
]

---

### Tensorflow

[object detection API](https://github.com/tensorflow/models/tree/master/research/object_detection)

---

### Pytorch

Detectron https://github.com/facebookresearch/Detectron

- Mask-RCNN, Retina Net and other architectures
- Focal loss, Feature Pyramid Networks, etc.

---

## Take away NN for Vision

### Pre-trained features as a basis

- ImageNet: centered objects, very broad image domain
- 1M+ labels and many different classes resulting in very **general** and **disentangling** representations
- Better Networks (i.e. ResNet vs VGG) have **a huge impact**

---

### Fine tuning

- Add new layers on top of convolutional or dense layer of CNNs
- **Fine tune** the whole architecture end-to-end
- Make use of a smaller dataset but with richer labels (bounding boxes, masks...)

---

# Next: Lab 5!