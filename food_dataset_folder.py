import os, glob
from torch.utils.data import Dataset
import cv2

class FoodDataset(Dataset):
    def __init__(self, root_dir, mode, transform=None):

        self.image_path = glob.glob(os.path.join(root_dir, mode, '*','*.jpg'))
        self.label_list = os.listdir(os.path.join(root_dir, mode))
        self.label_dict = {}
        self.transform = transform

        for i, label in enumerate(self.label_list):
            self.label_dict[label] = i      
        
    def __getitem__(self, item):
        name = self.label_list[item]
        image_path = self.image_path[item]
        
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        label = self.label_dict[name]

        if self.transform is not None:
            image = self.transform(image=image)['image']
            
        return image, label


    def __len__(self):
        return len(self.image_path)

# test = FoodDataset('./food_final', mode='val')
# for t in test:
#     _, label = t
#     print(label)
# print(len(test))
