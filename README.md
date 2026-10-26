# HW2 - Best Accuracy notebook
https://www.kaggle.com/code/aneaandrei/dl-genai-2026-fii-competition-2-baseline?scriptVersionId=360732107


# HW2 – Deep Learning Improvements

Image classification with VGG13 on a 100-class, 32×32 dataset (500 training images per class), trained on Kaggle with **2× T4 GPUs** under a **25-minute** limit.

The baseline reached **74.9%** validation accuracy. Four changes took it to **79.08%**.

## Results

| Step | Validation accuracy |
|---|---|
| Baseline: single VGG13, flip, affine, CutMix | 74.9% |
| Single model with the new training setup | ~77% |
| Two models trained together, averaged | 78.0% |
| + test-time augmentation | 78.4% |
| + class-balanced predictions | **79.08%** |

## The four changes

### 1. Two VGG13s with mutual distillation
[`1-two-VGG13s-mutual-distillation.ipynb`](1-two-VGG13s-mutual-distillation.ipynb)

One VGG13 per GPU, both trained on the same augmented batch. Each model learns from the true labels and from the other model's softened predictions (KL divergence, temperature 3, weight 2). At test time the two models' probabilities are averaged. The GPUs run in parallel, so the second model costs no extra time.

### 2. Stronger augmentation with a clean finish
[`2-stronger-augmentation-clean-finish.ipynb`](2-stronger-augmentation-clean-finish.ipynb) (final version of the code)

- One random colour operation per image, at a random strength, done on the GPU: brightness, contrast, saturation, sharpness, solarize, posterize, autocontrast, equalize
- CutMix probability lowered from 0.5 to 0.3
- The last 10% of training runs without CutMix or colour changes, so the model finishes on images that look like the test set

The clean finish gave the biggest late gain: about +2 points in the last ~2 minutes.

### 3. A training schedule that follows the clock
[`3-clock-driven-training-schedule.ipynb`](3-clock-driven-training-schedule.ipynb)

- The OneCycle learning-rate and momentum schedule is driven by elapsed time instead of a fixed number of epochs, so training always stops on time with the learning rate fully lowered
- Validation every 10 epochs and cuDNN autotuning leave more of the budget for training
- About 200–218 epochs fit in the time limit; the final run trains on all 50,000 images

### 4. Test-time augmentation and class-balanced predictions
[`4-TTA-and-class-balanced-predictions.ipynb`](4-TTA-and-class-balanced-predictions.ipynb)

- Each test image is predicted 6 times per model: the original, a mirrored copy, and 4 copies shifted by one pixel
- Sinkhorn normalisation rebalances the averaged probabilities so every class gets about the same number of predictions, matching the balanced training set

Before balancing, some classes were predicted 82 times and others 278 times, where about 125 is expected.

## Running

Open a notebook on Kaggle, select **GPU T4 x2**, and run all cells.
