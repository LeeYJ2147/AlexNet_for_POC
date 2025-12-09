import yaml
import argparse
import torch
import torch.nn as nn
import os
from tqdm import tqdm

from src.data_loader import create_dataloader
from src.models import create_model
from src.engine import train_one_epoch, evaluate

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main(config_path):
    # 1. Load Configuration
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {config_path}")
        return
        
    print("--- Configuration ---")
    print(yaml.dump(config, allow_unicode=True, default_flow_style=False))
    print("---------------------")

    # 2. Setup Environment
    device_str = config['training'].get('device', 'cpu')
    if device_str == 'cuda' and not torch.cuda.is_available():
        device_str = 'cpu'
        
    device = torch.device(device_str)
    print(f"Using device: {device}")
    set_seed(config['training']['seed'])

    output_dir = f"saved_models/{config['run_name']}"
    os.makedirs(output_dir, exist_ok=True)
    
    # 3. Create DataLoaders and Model
    train_loader, test_loader = create_dataloader(config['data'])
    model = create_model(
        model_name=config['model']['name'],
        num_classes=config['data']['num_classes'],
        pretrained=config['model']['pretrained'],
        **config['model'].get('params', {})
    ).to(device)

    # 4. Optimizer, Scheduler, and Early Stopping Setup
    training_config = config['training']
    attention_lr = training_config.get('attention_lr', training_config['learning_rate'])
    weight_decay = training_config.get('weight_decay', 0.0)
    
    # Separate parameters for differential learning rate
    attention_params = [p for n, p in model.named_parameters() if 'attention' in n and p.requires_grad]
    backbone_params = [p for n, p in model.named_parameters() if 'attention' not in n and p.requires_grad]

    optimizer = torch.optim.Adam([
        {'params': backbone_params},
        {'params': attention_params, 'lr': attention_lr}
    ], lr=training_config['learning_rate'], weight_decay=weight_decay)
    
    scheduler_config = training_config.get('lr_scheduler', {})
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=scheduler_config.get('step_size', 10), 
        gamma=scheduler_config.get('gamma', 0.1)
    )

    loss_fn = nn.CrossEntropyLoss()
    
    patience = training_config.get('early_stopping_patience', 20)
    epochs_no_improve = 0

    # 5. Training Loop
    best_val_acc = 0.0
    best_epoch = 0
    epochs = config['training']['epochs']
    
    train_loss_history, val_loss_history = [], []
    best_epoch_logs = []

    epoch_pbar = tqdm(range(epochs), desc="Total Epochs")
    for epoch in epoch_pbar:
        
        train_loss, train_acc = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device
        )
        
        val_loss, val_acc = evaluate(
            model=model,
            dataloader=test_loader,
            loss_fn=loss_fn,
            device=device,
            preprocessing_name=config['data']['preprocessing']['name']
        )

        scheduler.step()
        
        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        
        epoch_pbar.set_postfix({
            'Train Acc': f"{train_acc:.4f}",
            'Val Acc': f"{val_acc:.4f}",
            'Best Acc': f"{best_val_acc:.4f} @ ep {best_epoch}"
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            epochs_no_improve = 0

            save_path = os.path.join(output_dir, "best_model.pth")
            torch.save(model.state_dict(), save_path)

            _, _, test_preds, test_labels = evaluate(model, test_loader, loss_fn, device, config['data']['preprocessing']['name'], return_preds=True)
            preds_save_path = os.path.join(output_dir, "best_model_predictions.pt")
            torch.save({'preds': test_preds, 'labels': test_labels}, preds_save_path)

            log_msg = f"Epoch {epoch + 1}: New best model saved to {save_path} (Accuracy: {best_val_acc:.4f})"
            best_epoch_logs.append(log_msg)
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve >= patience:
            print(f"\nEarly stopping triggered after {patience} epochs with no improvement.")
            break

    for log in best_epoch_logs:
        print(log)
        
    print("\n--- Training Finished ---")
    print(f"Best validation accuracy: {best_val_acc:.4f}")

    return (train_loss_history), (val_loss_history)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a model based on a YAML config file.")
    parser.add_argument('--config', type=str, required=True, 
                        help="Path to the configuration file (e.g., 'configs/exp1b.yaml')")
    args = parser.parse_args()
    main(args.config)
