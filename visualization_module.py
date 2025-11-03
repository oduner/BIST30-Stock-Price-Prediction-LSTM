import os
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')


def plot_training_history(history: dict, fold_num: int, company_code):

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'{company_code} OpenChange of Training History - Fold {fold_num}', fontsize=16)

    axes[0, 0].plot(history.get("loss", []), label="Train Total Loss", color='blue')
    axes[0, 0].plot(history.get("val_loss", []), label="Val Total Loss", color='red')
    axes[0, 0].set_title("Total Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    axes[0, 1].plot(history.get("price_output_loss", []), label="Train Price Loss", color='blue')
    axes[0, 1].plot(history.get("val_price_output_loss", []), label="Val Price Loss", color='red')
    axes[0, 1].set_title("Price Prediction Loss")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Loss")
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    axes[0, 2].plot(history.get("direction_output_loss", []), label="Train Direction Loss", color='blue')
    axes[0, 2].plot(history.get("val_direction_output_loss", []), label="Val Direction Loss", color='red')
    axes[0, 2].set_title("Direction Prediction Loss")
    axes[0, 2].set_xlabel("Epoch")
    axes[0, 2].set_ylabel("Loss")
    axes[0, 2].legend()
    axes[0, 2].grid(True)

    axes[1, 0].plot(history.get("price_output_mae_price", []), label="Train MAE", color='blue')
    axes[1, 0].plot(history.get("val_price_output_mae_price", []), label="Val MAE", color='red')
    axes[1, 0].set_title("Mean Absolute Error")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("MAE")
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    axes[1, 1].plot(history.get("direction_output_binary_accuracy", []), label="Train Accuracy", color='blue')
    axes[1, 1].plot(history.get("val_direction_output_binary_accuracy", []), label="Val Accuracy", color='red')
    axes[1, 1].set_title("Direction Accuracy")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Accuracy")
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    if "lr" in history:
        axes[1, 2].plot(history["lr"], label="Learning Rate", color='green')
        axes[1, 2].set_title("Learning Rate Schedule")
        axes[1, 2].set_xlabel("Epoch")
        axes[1, 2].set_ylabel("Learning Rate")
        axes[1, 2].set_yscale('log')
        axes[1, 2].legend()
        axes[1, 2].grid(True)
    else:
        axes[1, 2].text(0.5, 0.5, "Learning Rate\nNot Available",
                       ha='center', va='center', transform=axes[1, 2].transAxes)
        axes[1, 2].set_title("Learning Rate Schedule")


    save_path = "Results/OpenChange/Plotting"

    company_folder = os.path.join(save_path, company_code)
    os.makedirs(company_folder, exist_ok=True)
    file_name = f"{company_code} OpenChange of training_history_fold{fold_num}.png"
    full_path = os.path.join(company_folder, file_name)
    plt.savefig(full_path)
    print(f"Plot kaydedildi: {full_path}")

    plt.close(fig)