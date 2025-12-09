import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# PCA values calculated from the provided notebook for the POC dataset.
# These are used in the Lighting transform for color augmentation.
POC_EIGVALS_224 = torch.tensor([3290.864258, 1239.206299, 538.111511])
POC_EIGVECS_224 = torch.tensor([
    [0.4459, 0.7635, 0.4671],
    [0.8872, -0.4461, -0.1177],
    [-0.1185, -0.4669, 0.8763],
])

POC_EIGVALS_256 = torch.tensor([3126.742920, 1230.723022, 530.931519])
POC_EIGVECS_256 = torch.tensor([
    [0.4332, 0.7737, 0.4623],
    [0.8945, -0.4318, -0.1156],
    [-0.1102, -0.4637, 0.8791],
])

# Standard ImageNet normalization values
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class Lighting(object):
    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = eigval
        self.eigvec = eigvec

    def __call__(self, img):
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        
        rgb = self.eigvec.type_as(img).clone() \
            .mul(alpha.view(1, 3).expand(3, 3)) \
            .mul(self.eigval.view(1, 3).expand(3, 3)) \
            .sum(1).squeeze()

        img = img.add(rgb.div(255.0).view(3, 1, 1).expand_as(img))
        return img


class TenCropAndNormalize(object):
    def __init__(self, normalize_transform):
        self.normalize_transform = normalize_transform
        self.to_tensor = transforms.ToTensor()

    def __call__(self, crops):
        return torch.stack([self.normalize_transform(self.to_tensor(crop)) for crop in crops])

def create_dataloader(config: dict):
    data_path = config['path']
    preprocessing_config = config['preprocessing']
    batch_size = config['batch_size']
    
    train_dir = f"{data_path}/train"
    test_dir = f"{data_path}/test"
    
    normalize_transform = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    
    if preprocessing_config['name'].lower() == 'resize':
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            Lighting(0.1, POC_EIGVALS_224, POC_EIGVECS_224),
            normalize_transform,
        ])
        test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize_transform,
        ])
    elif preprocessing_config['name'].lower() == '10crop':
        train_transform = transforms.Compose([
            transforms.Resize(preprocessing_config['resize_size']),
            transforms.RandomCrop(preprocessing_config['crop_size']),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            Lighting(0.1, POC_EIGVALS_256, POC_EIGVECS_256),
            normalize_transform,
        ])
        test_transform = transforms.Compose([
            transforms.Resize(preprocessing_config['resize_size']),
            transforms.TenCrop(preprocessing_config['crop_size']),
            TenCropAndNormalize(normalize_transform)
        ])
    else:
        raise ValueError(f"Preprocessing method '{preprocessing_config['name']}' not recognized.")
        
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    
    print(f"DataLoaders created successfully.")
    print(f"  - Preprocessing: {preprocessing_config['name']}")
    print(f"  - Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")
    
    return train_loader, test_loader