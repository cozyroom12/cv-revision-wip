import pandas as pd
import os, glob, shutil
from tqdm import tqdm

"""
food dataset
    train
        class_name(0~250)
    val
        class_name
"""

# root_dir = './food_dataset_final'

class FoodDataPrep:
    def __init__(self, root_dir, mode='train'):
        self.root_dir = root_dir
        self.mode = mode

        text_file = pd.read_csv('./food_dataset/class_list.txt', sep=' ', header=None, names=['label', 'name'])
        self.food_name = text_file.name.to_list()
        self.food_label = text_file.label.to_list()

        self.food_dict = {}
        for i, name in enumerate(self.food_name):
            self.food_dict[i] = name

        self.file_csv = pd.read_csv(f'./food_dataset/{self.mode}_labels.csv')

        for name in self.food_name:
            name_path = os.path.join(self.root_dir, self.mode, name)
            os.makedirs(name_path, exist_ok=True)            

    def move_files(self):
        # dir = './food_dataset/train_set/train_set/*.jpg'
        for label in self.food_label:
            mask = (self.file_csv.label == label)
            #print(label)
            file_masked = self.file_csv.loc[mask].img_name.to_list()
            file_name = self.food_dict[label]  # folder name 이될것임
            
            for image in tqdm(file_masked, desc=f'Moving {file_name} images'):
            #for image in file_masked:
                src_dir = os.path.join(f'./food_dataset/{self.mode}_set/{self.mode}_set', image)
                dst_dir = os.path.join(self.root_dir, self.mode, file_name, image)

                shutil.copyfile(src_dir, dst_dir)
        print("moving has finished.")

t = FoodDataPrep('./food_final/', mode='train')
t.move_files()
#v = FoodDataPrep('./food_final/', mode='val')
#v.move_files()
