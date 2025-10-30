# Ensure in correct directory (recognition/Siamese_network_s4786177)

from train import train_model
from predict import evaluate_on_test
from dataset import set_seed

# Configuration constants
SEED = 42 
CKPT_PATH = "siamese_ce.pt"

if __name__ == "__main__":
    set_seed(SEED)
    
    # 1. Run Training (saves the best model to siamese_ce.pt)
    train_model(epochs=10, batch_size=32, save_path=CKPT_PATH)

    # 2. Run Evaluation (loads the saved model and generates all final reports)
    evaluate_on_test(
        checkpoint_path=CKPT_PATH, 
        plot_tsne=True
    )