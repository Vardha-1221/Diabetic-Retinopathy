import torch

# Load the model from file
model = torch.load('full_model.pth', map_location=torch.device('cpu'), weights_only=False)

# Print model architecture
print(model)
