# train.py
import torch
from easyocr import Reader
from easyocr import utils
from easyocr.training import ModelTrainer
from pathlib import Path

# -------------------------
# Settings
# -------------------------
train_lmdb_path = "/content/drive/MyDrive/training_lmdb/train"
valid_lmdb_path = "/content/drive/MyDrive/training_lmdb/val"
output_dir = "/path/to/output_model"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
num_iter = 50000
batch_size = 64

# Language list for training
lang_list = ['en']  # modify as needed

# -------------------------
# Initialize ModelTrainer
# -------------------------
trainer = ModelTrainer(
    lang_list=lang_list,
    gpu=torch.cuda.is_available(),
    batch_size=batch_size,
    saved_model=None  # None = start from scratch
)

# -------------------------
# Start Training
# -------------------------
trainer.train(
    train_data=train_lmdb_path,
    valid_data=valid_lmdb_path,
    num_iter=num_iter,
    output_dir=output_dir
)

# -------------------------
# Save final model
# -------------------------
final_model_path = Path(output_dir) / "custom_model.pth"
trainer.save_model(final_model_path)
print(f"Training complete. Model saved at: {final_model_path}")
