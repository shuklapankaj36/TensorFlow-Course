# README

## Project Title: Multi-Class Image Segmentation with U-Net

This project implements a U-Net architecture for semantic segmentation on a multi-class dataset using TensorFlow/Keras. The aim is to predict pixel-level class labels from input images.

---

## Key Features:
1. **Model Architecture**:
   - A U-Net-based convolutional neural network with encoder-decoder structure.
   - Includes dropout layers for regularization.
   - Designed for 10 output classes using a softmax activation.

2. **Dataset**:
   - Images and masks from a directory structure:
     - Images: `/image/`
     - Masks: `/mask/`
   - Data is resized to `(256, 256)` for consistency.

3. **Training**:
   - Uses the `Adam` optimizer with a learning rate of `0.001`.
   - Categorical cross-entropy as the loss function.
   - Metrics: Accuracy.

4. **Evaluation**:
   - Mean Intersection over Union (IoU) computed using Keras's `MeanIoU` metric.
   - IoU is calculated per class and averaged.

5. **Visualization**:
   - Training and validation loss/accuracy plotted across epochs.
   - Predicted segmentation masks compared with ground truth.

---

## Results:
- **Training**: Achieved 62% accuracy after 5 epochs.
- **IoU**:
  - Mean IoU across 10 classes: ~5.9%.
  - IoU for individual classes ranges from `0.0` to `0.59`.

---

## How to Run:
1. **Dataset Preparation**:
   - Mount Google Drive in Colab.
   - Organize images and masks in `/Train/Labeled/Non-Flooded/`.

2. **Execution**:
   - Run the script in Google Colab.
   - Adjust paths and parameters as needed.

3. **Dependencies**:
   - TensorFlow, Keras, OpenCV, Matplotlib, scikit-learn, and NatSort.

---

## Limitations:
- Low IoU indicates underperformance, requiring hyperparameter tuning, additional data preprocessing, or enhanced model architecture.
- Classes with low representation might cause class imbalance issues.

---

## Future Improvements:
- Introduce class weights to address imbalance.
- Experiment with advanced architectures like ResNet-based U-Net.
- Fine-tune learning rates and increase epochs.

