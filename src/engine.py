import torch
import torch.nn as nn

def train_one_epoch(model: nn.Module, 
                    dataloader: torch.utils.data.DataLoader, 
                    optimizer: torch.optim.Optimizer, 
                    loss_fn: nn.Module, 
                    device: torch.device):
    model.train()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = loss_fn(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total_samples += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()

    epoch_loss = total_loss / total_samples
    epoch_acc = correct_predictions / total_samples
    return epoch_loss, epoch_acc


def evaluate(model: nn.Module, 
             dataloader: torch.utils.data.DataLoader, 
             loss_fn: nn.Module, 
             device: torch.device, 
             preprocessing_name: str,
             return_preds: bool = False):
    model.eval()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, local_labels = inputs.to(device), labels.to(device)
            
            # 10-Crop
            # [batch, 10, C, H, W]
            is_10_crop = (preprocessing_name.lower() == '10crop' and len(inputs.shape) == 5)

            if is_10_crop:
                batch_size, n_crops, C, H, W = inputs.size()
                inputs = inputs.view(-1, C, H, W) # -> [batch*10, C, H, W]
                
                outputs = model(inputs)
                
                outputs = outputs.view(batch_size, n_crops, -1).mean(1) # -> [batch, num_classes]
            else:
                outputs = model(inputs)

            loss = loss_fn(outputs, local_labels)

            total_loss += loss.item() * local_labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_samples += local_labels.size(0)
            correct_predictions += (predicted == local_labels).sum().item()
            
            if return_preds:
                all_preds.append(predicted.cpu())
                all_labels.append(labels.cpu())

    epoch_loss = total_loss / total_samples
    epoch_acc = correct_predictions / total_samples
    
    if return_preds:
        return epoch_loss, epoch_acc, torch.cat(all_preds), torch.cat(all_labels)
    else:
        return epoch_loss, epoch_acc
