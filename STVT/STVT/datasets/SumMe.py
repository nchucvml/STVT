import torch
import h5py
from torch.utils.data import Dataset, DataLoader
import numpy as np

def SumMe(args, distributed=False):
    class SumMeDataset(Dataset):
        global In_target
        In_target = 0
        def __init__(self, file_dir, video_amount, F_In_target=False):
            self.image_label_list = self.read_file(file_dir, F_In_target)
            self.video_amount = video_amount
            self.len = len(self.image_label_list)
            self.F_In_target = F_In_target

        def __getitem__(self, i):
            index = i % self.len
            img, label, video_number, imagenumber = self.image_label_list[index]
            return img, label, video_number, imagenumber

        def __len__(self):
            data_len = self.len
            return data_len

        def read_file(self, file_dir, F_In_target):
            global In_target
            with h5py.File(file_dir, "r") as f:
                patch_number = args.sequence
                dim = 512
                image_label_list = []
                for key in f.keys():
                    video_number = int(key[6:])
                    if video_number in video_amount:
                        video = f[key]
                        features = video['feature'][:]
                        gtsummary = video['label'][:]

                        downsample_image_number = len(features)
                        gonumber = int(downsample_image_number / patch_number)
                        for ds_image_index in range(gonumber):
                            for index_column in range(int(patch_number ** 0.5)):
                                image_row = np.reshape(
                                    features[ds_image_index * patch_number + index_column * int(patch_number ** 0.5)],
                                    (dim, 1, 1))
                                for index_row in range(1, int(patch_number ** 0.5)):
                                    image = np.reshape(features[ds_image_index * patch_number + index_column * (
                                        int(patch_number ** 0.5)) + index_row], (dim, 1, 1))
                                    image_row = np.concatenate(([image_row, image]), axis=2)
                                if index_column == 0:
                                    cat_image = image_row
                                else:
                                    cat_image = np.concatenate(([cat_image, image_row]), axis=1)

                            cat_image = cat_image.tolist()
                            cat_image = torch.FloatTensor(cat_image)
                            f_gtsummary = gtsummary[ds_image_index * patch_number:(ds_image_index + 1) * patch_number]
                            if F_In_target:
                                In_target += sum(f_gtsummary)
                            f_gtsummary = torch.tensor(f_gtsummary, dtype=torch.long)
                            f_video_number = [video_number for x in
                                              range(ds_image_index * patch_number, (ds_image_index + 1) * patch_number)]
                            f_video_number = torch.tensor(f_video_number, dtype=torch.long)
                            f_image_number = [x for x in
                                              range(ds_image_index * patch_number+1, (ds_image_index + 1) * patch_number + 1)]
                            f_image_number = torch.tensor(f_image_number, dtype=torch.long)
                            image_label_list.append((cat_image, f_gtsummary, f_video_number, f_image_number))
            return image_label_list

    all_arr = []
    for i in range(25):
        all_arr.append(i+1)
    test_arr = list(map(int, args.test_dataset.split(',')))
    train_arr = [i for i in all_arr if i not in test_arr]

    file_dir = './STVT/datasets/datasets/SumMe.h5'

    video_amount = train_arr
    train_data = SumMeDataset(file_dir=file_dir, video_amount=video_amount, F_In_target=True)
    train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True, drop_last=True)
    video_amount = test_arr
    test_data = SumMeDataset(file_dir=file_dir, video_amount=video_amount, F_In_target=False)
    test_loader = DataLoader(dataset=test_data, batch_size=args.val_batch_size, shuffle=False, drop_last=True)

    return train_loader, test_loader, In_target