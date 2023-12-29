import torch
import torch.nn as nn
from torchvision.transforms import Compose, ToTensor
from PIL import Image
import matplotlib.pyplot as plt
import os

NUM_CLASSES = 100

MODEL_PATH_BIGCNN = r"C:\Users\Win\Desktop\GaitCO\Gait_Application\CNN_Models\BigCNN-100-3.pth"
MODEL_PATH_FOURSPLIT = r"C:\Users\Win\Desktop\GaitCO\Gait_Application\CNN_Models\FourSplitCNN-100-3.pth"
MODEL_PATH_TRANSFORMER = r"C:\Users\Win\Desktop\GaitCO\Gait_Application\CNN_Models\TransformerCNN-100-3.pth"

ROOT_PATH = r"C:\Users\Win\Desktop\GaitCO\processed_images"

class CNNClassifier(nn.Module):
    def __init__(self, num_classes):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.fc1 = nn.Linear(64 * 32 * 32, 1024)
        self.fc2 = nn.Linear(1024, num_classes)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        x = self.maxpool(self.relu(self.conv1(x)))
        x = self.maxpool(self.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)  
        return x
    
class FourSplitCNNClassifier(nn.Module):
    def __init__(self, num_classes):
        super(FourSplitCNNClassifier, self).__init__()

        # Separate convolution layers for each strip
        self.conv1_1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        self.conv1_2 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        self.conv1_3 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2_3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        self.conv1_4 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2_4 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        # Adjust the input size for fc1
        self.fc1 = nn.Linear(64 * 4 * 8 * 32, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        batch_size = x.size(0)
        strips = torch.chunk(x, 4, dim=2)
        
        # Process each strip independently
        processed_strips = []
        for i, strip in enumerate(strips):
            strip = strip.squeeze(dim=1)  # Adjust dimension for conv1
            if i == 0:
                out = self.maxpool(self.relu(self.conv1_1(strip)))
                out = self.maxpool(self.relu(self.conv2_1(out)))
            elif i == 1:
                out = self.maxpool(self.relu(self.conv1_2(strip)))
                out = self.maxpool(self.relu(self.conv2_2(out)))
            elif i == 2:
                out = self.maxpool(self.relu(self.conv1_3(strip)))
                out = self.maxpool(self.relu(self.conv2_3(out)))
            else:
                out = self.maxpool(self.relu(self.conv1_4(strip)))
                out = self.maxpool(self.relu(self.conv2_4(out)))

            processed_strips.append(out)

        # Concatenate the processed strips
        concatenated = torch.cat(processed_strips, dim=1)
        concatenated = torch.flatten(concatenated, start_dim=1)

        x = self.relu(self.fc1(concatenated))
        x = self.fc2(x)
        return x

class CNNTransformerClassifier(nn.Module):
    def __init__(self, num_classes, d_model=512, nhead=8, num_transformer_layers=3):
        super(CNNTransformerClassifier, self).__init__()
        
        self.d_model = d_model

        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2)

        # Flattening layer
        self.flatten = nn.Flatten()

        # Linear layer to transform the feature dimension
        self.feature_transform = nn.Linear(64 * 32 * 32, d_model)

        # Transformer encoder layers
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_transformer_layers)

        # Fully connected layer for classification
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.maxpool(self.relu(self.conv1(x)))
        x = self.maxpool(self.relu(self.conv2(x)))

        # Flatten and transform the feature dimension
        x = self.flatten(x)
        x = self.feature_transform(x)

        # Reshape for transformer encoder: [batch size, sequence length, features]
        x = x.view(x.size(0), -1, self.d_model)

        # Transformer encoder
        x = self.transformer_encoder(x)

        # Classification head
        x = self.fc(x[:, 0, :])  # Use the output corresponding to the first token
        return x

def load_bigcnn_model(model_path, num_classes=100):
    model = CNNClassifier(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path))
    model = model.to("cpu")
    return model

def load_foursplit_model(model_path, num_classes=100):
    model = FourSplitCNNClassifier(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path))
    model = model.to("cpu")
    return model

def load_transformer_model(model_path, num_classes=100):
    model = CNNTransformerClassifier(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path))
    model = model.to("cpu")
    return model

def predict(model, image_paths, transform):
    images = [Image.open(path) for path in image_paths]
    images = [transform(img) for img in images]
    stacked_images = torch.cat(images, dim=0).unsqueeze(0)  # Add batch [1,3,128,128]
    stacked_images = stacked_images.to(next(model.parameters()).device)

    model.eval()  
    with torch.no_grad():
        outputs = model(stacked_images)

    probabilities = torch.softmax(outputs, dim=1)
    top_three_probs, top_three_indices = torch.topk(probabilities, 3, dim=1)
    top_three_user_ids = (top_three_indices + 1)

    return top_three_probs.squeeze(0).tolist(), top_three_indices.squeeze(0).tolist(), top_three_user_ids.squeeze(0).tolist()

def visualize_predictions(
    input_image_path, 
    predicted_user_ids, 
    probabilities, 
    root_dir
):
    input_image = Image.open(input_image_path)

    _, axes = plt.subplots(1, 4, figsize=(20,5))
    axes[0].imshow(input_image, cmap='gray')
    axes[0].set_title("Query Image")
    axes[0].axis('off')

    for i, user_id in enumerate(predicted_user_ids):
        formatted_user_id = f"{user_id:05d}"
        user_image_path = os.path.join(root_dir, '00001-04000', formatted_user_id ,'A1_silhouette', '0001.png')
        user_image = Image.open(user_image_path)

        axes[i + 1].imshow(user_image, cmap='gray')
        axes[i + 1].set_title(f'User ID: {formatted_user_id}\nProb: {round(probabilities[i] * 100, 4)}')
        axes[i + 1].axis('off')
    
    plt.show()

def improve_visualize_predictions(input_image_path, predicted_user_ids, probabilities, root_dir):
    # Display the input image
    input_image = Image.open(input_image_path)
    
    _, axes = plt.subplots(3, 4, figsize=(20, 20))
    
    axes[0, 0].imshow(input_image, cmap='gray')
    axes[0, 0].set_title("Input Image")
    axes[0, 0].axis('off')
    for ax in axes[0, 1:]:
        ax.axis('off')
    
    axes[1, 0].axis('off')
    axes[2, 0].axis('off')

    for user_index, user_id in enumerate(predicted_user_ids):
        formatted_user_id = f"{user_id:05d}"
        
        for silhouette_index, silhouette in enumerate(['A1_silhouette', 'A2_silhouette', 'A3_silhouette']):
            user_image_path = os.path.join(root_dir, '00001-04000', formatted_user_id ,silhouette, '0001.png')
            
            if os.path.exists(user_image_path):
                user_image = Image.open(user_image_path)
                axes[silhouette_index, user_index + 1].imshow(user_image, cmap='gray')
                if silhouette_index == 0:
                    axes[silhouette_index, user_index + 1].set_title(f'User ID: {formatted_user_id}\nProb: {round(probabilities[user_index] * 100, 4)}')
                axes[silhouette_index, user_index + 1].axis('off')
            else:
                axes[silhouette_index, user_index + 1].axis('off')

    plt.show()

if __name__ == "__main__":

    print("Testing the Three Models:\n")

    bigcnn_model = load_bigcnn_model(model_path=MODEL_PATH_BIGCNN, num_classes=NUM_CLASSES)
    foursplit_model = load_foursplit_model(model_path=MODEL_PATH_FOURSPLIT, num_classes=NUM_CLASSES)
    trasnformer_model = load_transformer_model(model_path=MODEL_PATH_TRANSFORMER, num_classes=NUM_CLASSES)

    # Testing Image path - exist in the database
    id_folder_range = "/00001-04000/00002/"
    id_folder_range = "/60001-62528/60672/"
    image_paths = [
        f'C:/Users/Win/Desktop/GaitCO/processed_images{id_folder_range}A1_silhouette/0001.png', 
        f'C:/Users/Win/Desktop/GaitCO/processed_images{id_folder_range}A1_silhouette/0005.png', 
        f'C:/Users/Win/Desktop/GaitCO/processed_images{id_folder_range}A1_silhouette/0010.png'
    ]
    transform = Compose([ToTensor()])

    print('BigCNN:')
    top_probs, top_indices, top_user_id = predict(bigcnn_model, image_paths, transform)

    print(f"Top 3 probabilities: {top_probs}")
    print(f"Top 3 class (array) indices: {top_indices}")
    print(f"Top 3 User Id: {top_user_id}")

    print("\n")

    # Visualization for the First Model prediction
    visualize_predictions(
        input_image_path=image_paths[0],
        predicted_user_ids=top_user_id,
        probabilities=top_probs,
        root_dir=ROOT_PATH
    )

    improve_visualize_predictions(
        input_image_path=image_paths[0],
        predicted_user_ids=top_user_id,
        probabilities=top_probs,
        root_dir=ROOT_PATH
    )


    #######################################################################################

    print('FourSplitCNN:')
    top_probs, top_indices, top_user_id = predict(foursplit_model, image_paths, transform)

    print(f"Top 3 probabilities: {top_probs}")
    print(f"Top 3 class (array) indices: {top_indices}")
    print(f"Top 3 User Id: {top_user_id}")

    print("\n")

    #######################################################################################

    print('TransformerCNN:')
    top_probs, top_indices, top_user_id = predict(trasnformer_model, image_paths, transform)

    print(f"Top 3 probabilities: {top_probs}")
    print(f"Top 3 class (array) indices: {top_indices}")
    print(f"Top 3 User Id: {top_user_id}")

    print("\n")