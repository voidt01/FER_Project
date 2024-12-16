import os
import cv2

class LoadData:
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.images = []
        self.image_labels = []

        if not os.path.exists(self.root_dir):
            raise FileNotFoundError("File not Found")
        else:
            for idx, dir in enumerate(os.listdir(self.root_dir)):
                img_folder = os.path.join(self.root_dir, dir)
                for img in os.listdir(img_folder):
                    img_path = os.path.join(img_folder, img)
                    img = cv2.imread(img_path)
                    if img is not None:
                        self.images.append(img)
                        self.image_labels.append(idx)
                    else:
                        print(f'{img} is none')

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.image_labels[idx]

    def __iter__(self):
        for img, label in zip(self.images, self.image_labels):
            yield img, label