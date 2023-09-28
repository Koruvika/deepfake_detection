from torch.utils.data import Dataset


class FaceForencisDataset(Dataset):
    def __init__(self):
        super(Dataset, self).__init__()

    def __len__(self):
        return

    def __getitem__(self, index):
        return


if __name__ == "__main__":
    dataset = FaceForencisDataset()
    