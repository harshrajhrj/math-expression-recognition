import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset

REL_DATASET_PATH = '../maths-dataset/'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, REL_DATASET_PATH)

LABEL_TO_ID = {
    'add': 0, 'divide': 1, 'eight': 2, 'five': 3, 'four': 4,
    'multiply': 5, 'nine': 6, 'one': 7, 'seven': 8, 'six': 9,
    'subtract': 10, 'three': 11, 'two': 12, 'zero': 13
}

ID_TO_CHAR = {
    0: '+', 1: '/', 2: '8', 3: '5', 4: '4', 5: '*', 6: '9',
    7: '1', 8: '7', 9: '6', 10: '-', 11: '3', 12: '2', 13: '0'
}

NUM_CLASSES = len(LABEL_TO_ID) + 1  # 14 + 1 for CTC blank

class GenerateExpressionDataset(Dataset):
    def __init__(self, label_dir: str, num_samples: int = 1000, transform=None):
        self.label_dir = label_dir
        self.num_samples = num_samples
        self.transform = transform
        self.digits = [k for k in LABEL_TO_ID if k not in ['add', 'subtract', 'multiply', 'divide']]
        self.operators = ['add', 'subtract', 'multiply', 'divide']

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        left = random.choice(self.digits)
        op = random.choice(self.operators)
        right = random.choice(self.digits)

        target_height = 32
        imgs = [self._load_image(label) for label in [left, op, right]]
        imgs = [img.resize((int(target_height * img.width / img.height), target_height), Image.Resampling.LANCZOS) for img in imgs]
        total_width = sum(img.width for img in imgs) + 4 * (len(imgs) - 1)

        new_img = Image.new('L', (total_width, target_height), color=255)
        x_offset = 0
        for img in imgs:
            new_img.paste(img, (x_offset, 0))
            x_offset += img.width + 4

        if self.transform:
            new_img = self.transform(new_img)

        label_seq = [LABEL_TO_ID[left], LABEL_TO_ID[op], LABEL_TO_ID[right]]
        return new_img, torch.tensor(label_seq, dtype=torch.long)

    def _load_image(self, label):
        folder = os.path.join(self.label_dir, label)
        files = os.listdir(folder)
        file = random.choice(files)
        return Image.open(os.path.join(folder, file)).convert("L")
    
# Example usage
# if __name__ == "__main__":
#     train_path = os.path.normpath(DATASET_PATH)
#     label_dir = train_path + "/train/"
#     expr_generator = GenerateExpressionDataset(label_dir, 1)
#     img, label_seq = expr_generator.__getitem__()
#     img.show()
#     print("Label Sequence:", label_seq)