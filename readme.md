# Math Solver AI

Using handwritten images, we extract spatial features of expressions and recognize sequences to solve a basic mathematical expression.

| Component | Purpose |
| --- | --- |
| CNN | Extract spatial features from image|
| BiLSTM | Capture sequence relationships left-to-right |
| CTC | Predict characters without needing aligned labels |

---

The task breakdown includes the **architecture**, **labelling strategy**, and a plan to train it from our dataset.

## Architecutre Breakdown: CRNN for Handwritten Math Expression Recognition
### Input
* Full expression image (e.g. `120 + 324` rendered as one horizontal image).
* Size: Say $32 * 32$ grayscale. (height x varable width)
### CNN Feature Extractor
We extract spatial features from the image. After this, flatten the height dimension and treat the width as a time sequence.
### Sequence Model (BiLSTM)
After feature extraction, we process the feature sequence and capture temporal relationships.
* **Input**: T x B x F (time steps, batch size, features)
* **Output**: For each time step, we get a prediction for a character.
### Linear Projection
Mapping LSTM output to our character set.
* `num_classes = len(vocab) + 1` (for CTC blank)
* Vocab: `['0'-'9', '+', '-', '×', '÷', '=']` → 14 tokens
### CTC Loss
We have used CTC loss because we don't know exact position of each character in the image. CTC allows the model to align predicted character sequences with actual text labels like "$3+4$" without needing character bounding boxes.
### Labelling Strategy
We need to encode text labels into integer sequences (training, test, validation labels).
```bash
{
    'add': 0,         → '+'
    'divide': 1,      → '÷'
    'eight': 2,       → '8'
    'five': 3,        → '5'
    'four': 4,        → '4'
    'multiply': 5,    → '×'
    'nine': 6,        → '9'
    'one': 7,         → '1'
    'seven': 8,       → '7'
    'six': 9,         → '6'
    'subtract': 10,   → '-'
    'three': 11,      → '3'
    'two': 12,        → '2'
    'zero': 13        → '0'
}

```
**Character decoding map** from your numeric labels to output expression
```python
label_to_char = {
    0: '+',   1: '÷',  2: '8',  3: '5',
    4: '4',   5: '×',  6: '9',  7: '1',
    8: '7',   9: '6', 10: '-', 11: '3',
   12: '2',  13: '0'
}
```
## Training Steps
1. **Expression Image Generator**: Generate a dataset of synthetic expression images and labels.
2. Build the CRNN model using PyTorch (as above).
3. **Training loop** (data loading, label encoding, CTC loss)
4. **Inference logic** to decode expression back to text. During inference, use CTC greedy decode or beam search to get best character sequence.
### CRNN Output Class
For CTC, we need:
* 1 class for each label (14 total)
* 1 additional "blank" class (for CTC) → total of 15 classes
### Label Sequences for CTC Training
If our expression is 3 + 5, we’ll encode it like:
```python
label_sequence = [11, 0, 3]  # 'three', 'add', 'five'
```
During inference, we’ll decode back using `label_to_char` and remove duplicate/blank predictions from the CTC output.
### Expression Generator (Pseudocode)
```python
def create_expression_image(digit_imgs, operator_imgs):
    # Randomly pick: '4 + 7' → [img4, img+, img7]
    imgs = [get_digit(4), get_operator('+'), get_digit(7)]
    # Concatenate horizontally
    expr_img = concatenate_images(imgs)
    # Label = '4+7'
    return expr_img, "4+7"
```
We can add random spacing, jitter, and noise to mimic real handwriting variation.
### 