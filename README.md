The improvements

Two VGG13s trained together with mutual distillation
 One VGG13 runs on each T4 GPU, and both see the same augmented batch at every step. Each model learns from the true labels and also from the other model’s softened predictions: a KL-divergence term at temperature T = 3, weight 2. That makes each model better than if trained alone. The test prediction averages the two models’ probabilities.
 Effect: single models reached about 77% on validation, versus 74.9% for the baseline. The two-model average reached 78.4–78.5%. Since the two GPUs run in parallel, the second model costs no extra time.
2. Stronger data augmentation, with a clean finish
 Your existing flip, affine and CutMix are now joined by one random colour operation per image, at a random strength, done on the GPU. The operations are brightness, contrast, saturation, sharpness, solarize, posterize, autocontrast and equalize. CutMix was lowered from 0.5 to 0.3. For the last 10% of training, all CutMix and colour changes are off, so the model finishes on images that look like the test set.
 Effect: the clean finish gave the biggest late jump in every run. In the last ~2 minutes, validation rose from about 74.5% to 76.5% for single models.
3. A training schedule that follows the clock
 The OneCycle learning-rate and momentum curve is driven by elapsed time instead of a fixed number of epochs. Training always stops at exactly TRAIN_MINUTES with the learning rate fully lowered. Validation runs every 10 epochs instead of every epoch, and cuDNN autotuning is on, so more of the budget goes to training.
 Effect: about 200–218 epochs fit into 21.5–23.5 minutes, and the whole notebook runs in about 23.8 minutes, under the 25-minute limit. The final run also trains on all 50,000 images (VAL_FRAC = 0).
4. Test-time augmentation and class-balanced predictions
 Each test image is predicted six times, averaged across both models: the original, a mirrored copy, and four copies shifted by one pixel. The averaged probabilities are then rebalanced so every class receives about the same share of predictions. This uses Sinkhorn normalisation, and it relies on the training set having exactly 500 images per class.
 Effect: on validation, test-time augmentation improved the two-model average from 78.00% to 78.40%, and balancing raised it to 79.08%. Before balancing, some classes were predicted 82 times and others 278, where about 125 is expected.

