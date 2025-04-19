# import os
# import pandas as pd
# import torch
# import torch.nn as nn
# from torchvision import models, transforms
# from torch.utils.data import Dataset, DataLoader
# from PIL import Image

# IMG_DIR = "data/images"  
# FINAL_HEIGHT_FT = 270
# START_DATE = "2025-04-01"
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# EPOCHS = 20
# BATCH_SIZE = 2

# # ====== Load and preprocess metadata ======
# data = pd.read_csv("metadata.csv")
# data["date"] = pd.to_datetime(data["file_name"].str.replace(".png", ""), format="%Y-%m-%d")
# data["days_since_start"] = (data["date"] - pd.to_datetime(START_DATE)).dt.days
# min_px, max_px = data["height_px"].min(), data["height_px"].max()
# data["height_ft"] = ((data["height_px"] - min_px) / (max_px - min_px)) * FINAL_HEIGHT_FT
# data["rate_ft_per_day"] = data["height_ft"] / data["days_since_start"].replace(0, 1)
# data["percent_complete"] = data["height_ft"] / FINAL_HEIGHT_FT

# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
# ])

# class BuildingDataset(Dataset):
#     def __init__(self, df, img_dir, transform):
#         self.df = df
#         self.img_dir = img_dir
#         self.transform = transform

#     def __len__(self):
#         return len(self.df)

#     def __getitem__(self, idx):
#         row = self.df.iloc[idx]
#         img_path = os.path.join(self.img_dir, row["file_name"])
#         img = Image.open(img_path).convert("RGB")
#         img = self.transform(img)
#         days = torch.tensor([row["days_since_start"]], dtype=torch.float32)
#         target = torch.tensor([row["rate_ft_per_day"], row["percent_complete"]], dtype=torch.float32)
#         return img, days, target

# dataset = BuildingDataset(data, IMG_DIR, transform)
# loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# class ConstructionModel(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.cnn = models.resnet18(weights="IMAGENET1K_V1")
#         self.cnn.fc = nn.Identity()
#         self.fc1 = nn.Linear(512 + 1, 128)
#         self.fc2 = nn.Linear(128, 2)

#     def forward(self, x_img, x_days):
#         x = self.cnn(x_img)
#         x = torch.cat([x, x_days], dim=1)
#         x = self.fc1(x)
#         return self.fc2(x)

# model = ConstructionModel().to(DEVICE)
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
# loss_fn = nn.MSELoss()

# print("Training...")
# model.train()
# for epoch in range(EPOCHS):
#     total_loss = 0
#     for imgs, days, targets in loader:
#         imgs, days, targets = imgs.to(DEVICE), days.to(DEVICE), targets.to(DEVICE)
#         preds = model(imgs, days)
#         loss = loss_fn(preds, targets)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item()
#     print(f"Epoch {epoch+1}: Loss = {total_loss:.4f}")

# # Save model
# torch.save(model.state_dict(), "model.pth")
# print("Model saved as model.pth")