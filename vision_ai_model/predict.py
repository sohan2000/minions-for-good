import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from datetime import datetime, timedelta

FINAL_HEIGHT_FT = 270
START_DATE = "2025-04-01"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

class ConstructionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = models.resnet18(weights="IMAGENET1K_V1")
        self.cnn.fc = nn.Identity()
        self.fc1 = nn.Linear(512 + 1, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x_img, x_days):
        x = self.cnn(x_img)
        x = torch.cat([x, x_days], dim=1)
        x = self.fc1(x)
        return self.fc2(x)

# Load model
model = ConstructionModel().to(DEVICE)
model.load_state_dict(torch.load("model.pth", map_location=DEVICE))
model.eval()

def predict(img_path, img_date):
    img = Image.open(img_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)
    days_since_start = (datetime.strptime(img_date, "%Y-%m-%d") - datetime.strptime(START_DATE, "%Y-%m-%d")).days
    days_tensor = torch.tensor([[days_since_start]], dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        output = model(img_tensor, days_tensor)[0].cpu().numpy()

    rate_ft_per_day = max(output[0], 1e-3)  # avoid division by zero
    percent_complete = output[1]
    feet_built = percent_complete * FINAL_HEIGHT_FT
    days_remaining = (1 - percent_complete) * FINAL_HEIGHT_FT / rate_ft_per_day
    projected_completion = datetime.strptime(img_date, "%Y-%m-%d") + timedelta(days=round(days_remaining))

    print(f"Date: {img_date}")
    print(f"Predicted rate: {rate_ft_per_day:.2f} ft/day")
    print(f"Completion: {percent_complete * 100:.2f}%")
    print(f"Feet built so far: {feet_built:.2f} ft")
    print(f"Estimated days to complete: {days_remaining:.1f}")
    print(f"Projected completion date: {projected_completion.strftime('%Y-%m-%d')}")

    return {
        "date": img_date,
        "rate_ft_per_day": rate_ft_per_day,
        "percent_complete": percent_complete * 100,
        "feet_built": feet_built,
        "days_remaining": days_remaining,
        "projected_completion_date": projected_completion.strftime("%Y-%m-%d")
    }

#Example usage
# predict("data/images/2025-05-14.png", "2025-05-14")


