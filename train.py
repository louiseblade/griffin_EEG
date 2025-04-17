import os.path

from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import random
import copy  # for deep copying best model state

from logging.handlers import RotatingFileHandler
import logging

from data_utils import *
from sklearn.model_selection import train_test_split
from proposed_method.novaNet import novaNet
from adjacency_matrix import *
from maxnorm_constraint import *
from weight_init import *
from weight_decay import *
from other_models.EEGNEt import *
from ATCNet_torch.ATCNet import ATCNet
from sklearn.metrics import cohen_kappa_score
from Griffin_module.griffin import Griffin
if __name__ == '__main__':

    torch.backends.cudnn.benchmark = True

    # ---------------------------
    # Configure logging with rotation
    # ---------------------------
    log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    log_file = 'training.log'
    rotating_handler = RotatingFileHandler(log_file, maxBytes=5 * 1024 * 1024, backupCount=3)
    rotating_handler.setFormatter(log_formatter)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)

    logging.basicConfig(level=logging.INFO, handlers=[rotating_handler, console_handler])
    logger = logging.getLogger(__name__)

    # ---------------------------
    # User-defined parameters
    # ---------------------------
    seeds = list(range(20))  # For demonstration; can revert to more seeds as needed.
    subjects = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    num_epochs = 1
    batch_size = 64
    save_model = True  # <-- Change this to False if you do NOT want to save model checkpoints.
    model_name = "ATCNet"  # <-- Change this to "ATCNet" or "EEGNet" as needed.
    # ---------------------------
    # System setup
    # ---------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    logger.info(f"Subjects: {subjects}")

    # ---------------------------
    # Data setup
    # ---------------------------
    label = "true_labels"
    data = "data_npz"

    # We'll store results here
    results = {}
    best_models_info = {}  # NEW: Track best seed/model per subject

    for seed in seeds:
        logger.info(f"\nRunning seed {seed}")
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # Store results
        results[seed] = {'accuracy': {}, 'kappa': {}}

        for subject in subjects:
            logger.info(f"\nProcessing Subject {subject}")

            train_list = [subject]
            test_list = [subject]

            # ---------------------------
            # Load training data
            # ---------------------------
            X_full, y_full = load_data(data, label, train_list, "T", transpose=True)
            X_test, y_test = load_data(data, label, test_list, "E", transpose=True)
            logger.info("Data loaded successfully.")

            # ---------------------------
            # Split into train / val
            # ---------------------------
            # X_train, X_val, y_train_np, y_val_np = train_test_split(
            #     X_full, y_full, test_size=0.2, random_state=seed, stratify=y_full
            # )
            logger.info("Data split into training and validation sets.")

            # #adjacency plv
            # adjacency = [compute_plv_adjacency(X_train, 250, 1, 51),
            #                 compute_plv_adjacency(X_train, 250, 1, 4),
            #                 compute_plv_adjacency(X_train, 250, 4, 8),
            #       ]
            # adjacency = [torch.tensor(adj, dtype=torch.float).to(device) for adj in adjacency]

            # ---------------------------
            # Convert to torch Tensors
            # ---------------------------
            X_train = torch.tensor(X_full).float().to(device)
            y_train = torch.tensor(y_full).long().to(device)
            # X_val   = torch.tensor(X_val).float().to(device)
            # y_val   = torch.tensor(y_val_np).long().to(device)
            X_test  = torch.tensor(X_test).float().to(device)
            y_test  = torch.tensor(y_test).long().to(device)

            # ---------------------------
            # Create DataLoaders
            # ---------------------------
            train_dataset = TensorDataset(X_train, y_train)
            # val_dataset   = TensorDataset(X_val, y_val)
            train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
            # val_loader    = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)

            # ---------------------------
            # Initialize Model
            # ---------------------------
            model = Griffin(dim=1125, depth=depth, lru_layer=lru_layer, mlp_mult=mlp_mult, dropout=dropout).to(device)
            model.apply(initialize_weights_keras_style)
            logger.info("Model initialized successfully.")

            # ---------------------------
            # Criterion and Optimizer
            # ---------------------------
            criterion = nn.CrossEntropyLoss()
            optimizer = get_optimizer(model, lr=0.001, weight_decay=True, verbose=False)

            scaler = GradScaler()
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", patience=20, factor=0.9, min_lr=0.0001
            )

            max_norm_value = 0.6

            # ---------------------------
            # For logging / tracking
            # ---------------------------
            train_losses, val_losses = [], []
            train_accs,   val_accs   = [], []

            # We'll track best model by highest val accuracy:
            best_val_acc = 0.0
            best_epoch = 0
            best_model_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0

            # ---------------------------
            # Training loop
            # ---------------------------
            for epoch in range(num_epochs):
                model.train()
                epoch_train_loss = 0.0
                correct_train = 0
                total_train = 0

                for (X_batch, y_batch) in train_loader:
                    optimizer.zero_grad()

                    with autocast():
                        output = model(X_batch)
                        loss = criterion(output, y_batch)

                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                    # Apply max-norm constraint
                    apply_max_norm_refined(
                        max_norm_val=max_norm_value,
                        modules_to_apply=[model],
                        layers=(nn.Conv1d, nn.Conv2d, nn.Linear),
                    )

                    epoch_train_loss += loss.item() * X_batch.size(0)
                    preds = torch.argmax(output, dim=1)
                    correct_train += (preds == y_batch).sum().item()
                    total_train += y_batch.size(0)

                epoch_train_loss /= total_train
                epoch_train_acc = correct_train / total_train

                # Validation step
                model.eval()
                with torch.no_grad():
                    val_output = model(X_test)
                    epoch_val_loss = criterion(val_output, y_test).item()
                    preds = torch.argmax(val_output, dim=1)
                    correct_val = (preds == y_test).sum().item()
                    total_val = y_test.size(0)
                    epoch_val_acc = correct_val / total_val

                # Step the scheduler with validation loss (optional - you can keep it or adapt to use accuracy)
                scheduler.step(epoch_val_loss)

                # Store training/val stats for logging
                train_losses.append(epoch_train_loss)
                val_losses.append(epoch_val_loss)
                train_accs.append(epoch_train_acc)
                val_accs.append(epoch_val_acc)

                # Check if this is the best (highest) val acc so far
                if epoch_val_acc > best_val_acc:
                    best_val_acc = epoch_val_acc
                    best_epoch = epoch
                    best_model_state = copy.deepcopy(model.state_dict())
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

                # Logging as desired
                if epoch % 100 == 0:
                    logger.info(
                        f"[Epoch {epoch}] "
                        f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc * 100:.2f}% | "
                        f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc * 100:.2f}% | "
                        f"LR: {optimizer.param_groups[0]['lr']:.6f}"
                    )

            # ---------------------------
            # Load the best model weights (by lowest val loss)
            # then evaluate on test set
            # ---------------------------
            model.load_state_dict(best_model_state)
            model.eval()

            with torch.no_grad():
                test_output = model(X_test)
                test_loss = criterion(test_output, y_test).item()
                _, predicted = torch.max(test_output, 1)
                correct = (predicted == y_test).sum().item()
                total = y_test.size(0)
                test_acc = correct / total
                kappa = cohen_kappa_score(y_test.cpu().numpy(), predicted.cpu().numpy())

            # Store results
            results[seed]['accuracy'][subject] = test_acc * 100
            results[seed]['kappa'][subject] = kappa

            PATH = f"{model_name}/subject_{subject}"
            if not os.path.exists(PATH):
                os.makedirs(PATH)

            path = f"{model_name}/subject_{subject}/sj_{subject}_seed_{seed}.pth"

            # NEW: Track best seed per subject
            if subject not in best_models_info or test_acc * 100 > best_models_info[subject]['best_acc']:
                best_models_info[subject] = {
                    'best_acc': test_acc * 100,
                    'best_kappa': kappa,
                    'best_seed': seed,
                    'model_path': path if save_model else None
                }

            logger.info(f"Seed {seed}, Subject {subject} - "
                        f"Best Val ACC: {best_val_acc:.4f} (Epoch {best_epoch}), "
                        f"Test Acc: {test_acc * 100:.2f}%, Kappa: {kappa:.4f}")

            # ---------------------------
            # Optionally save the best model to file
            # ---------------------------
            if save_model:
                save_path = path
                torch.save(best_model_state, save_path)
                logger.info(f"Best model saved to: {save_path}")

            # Cleanup
            del model
            torch.cuda.empty_cache()
            logger.info(f"Cleared model and cache for subject {subject}")

    # --------------------------------------------
    # Print Aggregated Results
    # --------------------------------------------

    print("\n---------------------------------")
    print("Best Performance Per Subject:")
    print("---------------------------------")
    print(f"{'Subject':<10} {'Best Seed':<10} {'Accuracy':<10} {'Kappa':<10}")
    for subject in subjects:
        info = best_models_info.get(subject, {'best_acc': 0, 'best_kappa': 0, 'best_seed': -1})
        print(f"{subject:<10} {info['best_seed']:<10} {info['best_acc']:.2f}%     {info['best_kappa']:.4f}")

    # NEW: Calculate averages of best performances
    best_accs = [v['best_acc'] for v in best_models_info.values()]
    best_kappas = [v['best_kappa'] for v in best_models_info.values()]
    print(f"\nAverage Best Accuracy: {np.mean(best_accs):.2f}% ± {np.std(best_accs):.2f}%")
    print(f"Average Best Kappa: {np.mean(best_kappas):.4f} ± {np.std(best_kappas):.4f}")

