import torch
import h5py
from tarp import get_tarp_coverage
def preprocess_probes_z_channel(img):  # channel 2
    img = torch.clamp(img, 0, 7.35)
    img = 2 * img / 7.35 - 1.
    return img


def preprocess_probes_r_channel(img):  # channel 1
    img = torch.clamp(img, 0, 3.47)
    img = 2 * img / 3.47 - 1.
    return img


def preprocess_probes_g_channel(img):  # channel 0
    img = torch.clamp(img, 0, 1.48)
    img = 2 * img / 1.48 - 1.
    return img


def probes_preprocessing(colors):
    prep_map = {0: preprocess_probes_g_channel, 1: preprocess_probes_r_channel, 2: preprocess_probes_z_channel}
    def prep(img):
        B, C, *_ = img.shape
        for c in range(C):
            img[:, c] = prep_map[sorted(colors)[c]](img[:, c])
        return img
    return prep


class AstroDataset(torch.utils.data.Dataset):
    def __init__(self, path_to_h5, key, channels, start_idx, end_idx, channels_last=False):
        self.filepath = path_to_h5
        # self.device = device
        self.key = key
        self.channels = channels
        self.channels_last = channels_last
        self.hf = h5py.File(self.filepath, "r")
        if end_idx == -1:
            end_idx = self.hf[self.key].shape[0]
        self.size = end_idx - start_idx# self.hf[self.key].shape[0]
        self.start_idx = start_idx
        self.preprocessing = probes_preprocessing(self.channels)

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        index = self.start_idx + index
        example = dict()
        if self.channels_last:
            im = torch.tensor(self.hf[self.key][index, :, :, self.channels])#.to(self.device)
            # put channels first for Conv2D score model
            im = torch.permute(im, (2, 0, 1))
            im = self.preprocessing(im.reshape(1, 3, 256, 256)).squeeze(0)
            example["image"] = im.permute(1, 2, 0).numpy()
            return example
        else:
            return torch.tensor(self.hf[self.key][index, self.channels])#.to(self.device)



if __name__ == "__main__":
    train_dataset = AstroDataset("/network/scratch/m/moksh.jain/astro/inverse_problem/probes.h5", "galaxies", [0,1,2], 300, -1, channels_last=True)
    val_dataset = AstroDataset("/network/scratch/m/moksh.jain/astro/inverse_problem/probes.h5", "galaxies", [0,1,2], 200, 300, channels_last=True)
    print(len(train_dataset))
    print(len(val_dataset))
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=100, shuffle=False)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1000, shuffle=False)
    
    gt_samples = next(iter(val_loader))["image"].reshape(100, -1).cpu().numpy()
    print(gt_samples.shape)
    train_samples = next(iter(train_loader))["image"].reshape(10, 100, -1).cpu().numpy()
    print(train_samples.shape)
    coverage = get_tarp_coverage(train_samples, gt_samples, bootstrap=True)
    print(coverage)
    import numpy as np
    means = coverage[0].mean(axis=0)
    conf = 1.96 * coverage[0].std(axis=0) / np.sqrt(coverage[0].shape[0])
    # import pdb; pdb.set_trace();
    import matplotlib.pyplot as plt
    plt.plot(coverage[1])
    plt.plot(means)
    plt.fill_between(range(len(means)), means - conf, means + conf, alpha=0.6)
    plt.savefig("coverage4.png")
    # import pdb; pdb.set_trace();
    # print(val_dataset[0].shape)
    # print(dataset[0].dtype)
    # from PIL import Image
    # import numpy as np

    # img = train_dataset[0].reshape(1, 3, 256, 256)
    # img = probes_preprocessing([0, 1, 2])(img)
    # img = img[0].permute(1, 2, 0)
    # img = img.numpy()
    # # img = (img + 1) / 2
    # # img = np.clip(img, 0, 1)
    # img = (img * 255).astype(np.uint8)
    # img = Image.fromarray(img)
    # img.save("test_process.png")
