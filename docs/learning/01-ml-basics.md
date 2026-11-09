# 01: Machine learning basics

This note covers the handful of ideas you need before we train anything. Each idea gets a plain explanation, then what it means for arabic-ocr.

---

## 1. Tensors: the data

A **tensor** is a grid of numbers. The word sounds scary, but you already know the shapes:

| Dimensions | Everyday name | Example in this project |
|---|---|---|
| 0 | a single number | the loss value after one training step |
| 1 | a list | the scores for each of our ~150 characters |
| 2 | a table | one grayscale line image: `height × width` pixels |
| 3 | a stack of tables | a batch of line images: `batch × height × width` |
| 4 | — | what the model actually receives: `batch × channels × height × width` |

The list of sizes is called the tensor's **shape**. A batch of 32 grayscale line images that are 48 pixels tall and 512 wide has shape `(32, 1, 48, 512)`. Most bugs in ML code are shape mismatches, so printing `x.shape` is the most useful debugging habit you can build.

Tensors are like NumPy arrays, with two extras: they can live on a **GPU** (where the math runs 10–100× faster), and they can remember how they were computed so the computer can work out **gradients** (see §4).

```python
import torch
x = torch.zeros(32, 1, 48, 512)   # a batch of 32 blank line images
print(x.shape)                     # torch.Size([32, 1, 48, 512])
```

---

## 2. Models: the function

A **model** is a function with millions of adjustable numbers inside it, called **weights** (or **parameters**). It takes an input tensor and returns an output tensor.

- Input: a line image.
- Output: for each horizontal slice of the image, a score for every character in our charset.

At first the weights are random, so the output is garbage. **Training** means adjusting the weights until the output is right.

Models are built from **layers** stacked in order, like an assembly line:

- **Convolutional layers (CNN)** slide small filters over the image to find visual patterns: edges first, then strokes, then letter shapes.
- **Sequence layers (BiLSTM or Transformer)** read those patterns left-to-right and right-to-left to use context, which matters a lot for joined Arabic letters.
- A final **output layer** turns each position into character scores.

Model size is measured in parameters. Our recognizer target is ≤ 20 MB, roughly 5 million parameters stored as 32-bit numbers.

---

## 3. Loss: the score of how wrong we are

The **loss** is one number that says how wrong the model's output is compared with the correct answer (the **label**). Lower is better; zero would be perfect.

Different tasks use different loss functions. Our recognizer uses **CTC loss**, which lets the model learn to read a whole line without anyone telling it where each letter starts and ends. We cover CTC properly in note 03.

The loss is only a training signal. The number we actually care about is the **CER** (character error rate: the percentage of characters we get wrong). They usually move together, but not always, which is why we measure both.

---

## 4. The training loop: how the model learns

Training repeats the same four steps thousands of times:

```python
for images, labels in train_loader:        # 1. take a small batch of examples
    outputs = model(images)                #    run the model ("forward pass")
    loss = loss_fn(outputs, labels)        # 2. measure how wrong it was
    optimizer.zero_grad()
    loss.backward()                        # 3. work out which way to nudge each weight
    optimizer.step()                       # 4. nudge every weight a little
```

- **Batch:** we feed a small group of examples (e.g. 64) at a time, not the whole dataset. That keeps memory use manageable and makes progress smoother.
- **Gradient:** for every weight, the direction and size of the change that would reduce the loss. `loss.backward()` computes all of them automatically ("backpropagation").
- **Optimizer** (e.g. Adam): the rule that turns gradients into weight updates.
- **Learning rate:** how big each nudge is. Too big and training jumps around and never settles; too small and it takes forever. It is the most important setting to tune.
- **Epoch:** one full pass over the training data. With millions of synthetic images we may count training in steps instead of epochs.
- **Checkpoint:** a saved copy of the weights. We save often so training can **resume** after a crash or a Kaggle/Colab session timeout.
- **Random seed:** fixing it makes runs repeatable, so when results change we know it was our change and not luck.

---

## 5. Train / validation / test split: honest measurement

A model can **memorize** its training examples instead of learning the general skill. This is called **overfitting**: perfect on data it has seen, bad on new data. To catch it, we split our data into three separate sets:

| Split | Typical share | Used for | The model learns from it? |
|---|---|---|---|
| **Train** | ~90–98% | adjusting the weights | yes |
| **Validation** (val) | ~1–5% | checking progress during training, picking settings, choosing the best checkpoint | no |
| **Test** | ~1–5% | the final score we report, measured once at the end | no |

Rules we follow:

1. **No overlap.** The same text line must never appear in two splits, even rendered in a different font. Otherwise the test score is too optimistic.
2. **Don't tune on the test set.** If we keep adjusting things until the test score looks good, the test set stops being "new" data. That is what the validation set is for.
3. **Keep a real-world test set.** Our training data is synthetic (rendered fonts). The test that matters most is on **real** scans and screenshots, which the model has never seen in any form. The gap between synthetic and real performance is called the **domain gap**, and it is the biggest risk in this project (PRD §14).

**Reading the curves:** during training we plot train loss and validation loss.

- Both going down → still learning.
- Train loss down, validation loss flat or rising → overfitting. Fixes: more or more varied data, stronger augmentation, a smaller model, or stopping earlier.
- Both stuck high → **underfitting**: the model is too small, the learning rate is wrong, or there's a bug (check the labels first).

---

## 6. Training vs inference

- **Training** (PyTorch, needs a GPU, done by us): adjusts the weights. Slow and expensive.
- **Inference** (ONNX Runtime, CPU is fine, done by users): runs the finished model with the weights frozen. Fast and cheap.

After training we **export** the model to an `.onnx` file. The shipped app only needs ONNX Runtime, so users never have to install PyTorch. This is why the `arabic_ocr` package must never import `torch`; a test in `tests/test_smoke.py` checks that.

---

## Quick glossary

| Term | Meaning |
|---|---|
| Tensor | A grid of numbers with a shape |
| Weights / parameters | The adjustable numbers inside a model |
| Forward pass | Running input through the model to get output |
| Loss | One number measuring how wrong the output is |
| Gradient | Which way to change each weight to lower the loss |
| Optimizer | The rule that applies gradient updates |
| Learning rate | Size of each update |
| Batch / epoch | A group of examples / one pass over all training data |
| Overfitting | Memorizing training data instead of learning the skill |
| Checkpoint | Saved weights, used to resume or ship |
| Inference | Using a trained model to make predictions |
