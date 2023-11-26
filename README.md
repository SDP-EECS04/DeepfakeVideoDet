# Deepfake Video Detection
#Frame Extraction Code:
# Define paths
dataset_dir = '/home/abdulla/sdp/train_sample_videos'
output_dir = '/home/abdulla/sdp/OUTPUT2'
# Load metadata
with open(os.path.join(dataset_dir, 'metadata.json'), 'r') as f:  
    metadata = json.load(f)

# Loop through each video in the metadata
for filename,entry in metadata.items():
    label = entry['label']    
    original = entry['original'] if 'original' in entry else None   
   
    video_path = os.path.join(dataset_dir, filename)    
    cap = cv2.VideoCapture(video_path)    

    #    Create output directories    
    output_subdir = 'real' if label == 'REAL' else 'fake'   
    output_subdir = os.path.join(output_dir, output_subdir)    
    os.makedirs(output_subdir, exist_ok=True)
    # Extract frames from video and save them    
    frame_count = 0   
    while cap.isOpened():
        ret, frame = cap.read()        
        if not ret:            
            break
        # Save the frame as an image in the appropriate subdirectory        
        frame_filename = f"{os.path.splitext(filename)[0]}_{frame_count:04d}.jpg"       
        frame_path = os.path.join(output_subdir, frame_filename)        
        cv2.imwrite(frame_path, frame)
        frame_count += 1   
    cap.release()
print("Data preprocessing complete.")
Reading Metadata.json Code:
train_sample_metadata.groupby('label')['label'].count().plot(figsize=(15, 5), kind='bar', title='Distribution of Labels in the Training Set')
plt.show()
train_sample_metadata = pd.read_json('/home/abdulla/sdp/train_sample_videos/metadata.json').T 
train_sample_metadata.head()
Main Code:
import os
import json
import random
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.nn.utils.rnn import pad_sequence, pack_sequence, pad_packed_sequence 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score , confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Define the CNN model architecture
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1).to(device)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
       
        # Calculate the output size after convolution and pooling
        conv_out_size = self._get_conv_out((3, 128, 128))
       
        # Adjust the fully connected layer to consider sequence length and convolution output size
        self.fc1 = nn.Linear(conv_out_size* 10, num_classes) #* 100

    def _get_conv_out(self, shape):
        x = torch.zeros(1, *shape).to(torch.float32).to(device)
        x = self.conv1(x)
        x = self.pool(x)
        return int(torch.prod(torch.tensor(x.size())))

    def forward(self, x):
        batch_size, sequence_length, channels, height, width = x.size()
       
        # Reshape the input tensor to merge batch_size and sequence_length dimensions
        x = x.view(-1, channels, height, width)
       
        x = self.pool(torch.relu(self.conv1(x)))
       
        # Calculate the conv_out_size and reshape the tensor
        conv_out_size = self._get_conv_out((channels, height, width))
        x = x.view(batch_size, sequence_length, conv_out_size)
       
        x = self.fc1(x.view(batch_size, -1))  # Flatten sequence and batch dimensions
        return x

# Define training parameters
batch_size = 16
learning_rate = 0.0001
num_epochs = 25

# Define paths
dataset_dir = '/home/abdulla/sdp/OUTPUT2'
metadata_path = os.path.join(dataset_dir, 'metadata.json')
#dataset_dir = 'output'
#metadata_path = os.path.join(dataset_dir, 'metadata.json')


# Define image transformation
transform = transforms.Compose([transforms.ToTensor(),])

# Create the dataset and data loader
class DeepfakeDataset(Dataset):
    def __init__(self, dataset_dir, metadata, transform=None):
        self.dataset_dir = dataset_dir
        self.metadata = metadata
        self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        video_filename = list(self.metadata.keys())[idx]
        entry = self.metadata[video_filename]

        label = 1 if entry['label'] == 'FAKE' else 0
        image_folder = 'fake' if label == 1 else 'real'

        target_height = 128  # Specify your desired height
        target_width = 128   # Specify your desired width

        frames = []

        for frame_number in range(10):
            image_filename = f"{video_filename.replace('.mp4', '')}_{frame_number:04d}.jpg"
            image_path = os.path.join(self.dataset_dir, image_folder, image_filename)

            if os.path.exists(image_path):  # Check if the image file exists
                    image = cv2.imread(image_path)
                    if image is not None:
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        if self.transform:
                            image = self.transform(image)
                            # Resize the image to the target dimensions
                            resized_image = transforms.Resize((target_height, target_width))(image)
                            frames.append(resized_image)

        if frames:  # Check if any frames were added
            frames = pad_sequence(frames, batch_first=True, padding_value=0)
            return frames, label
        else:
            # Return default tensors or handle the case where frames are missing
            default_frames = torch.zeros(10, 3, target_height, target_width)  # Adjust dimensions            
            return default_frames, label
   

def load_metadata(metadata_path):
    with open(metadata_path, 'r') as f:  
        metadata = json.load(f)
    return metadata

def validate_model(model, data_loader, device):
    model.eval()
    true_labels = []
    predicted_labels = []

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(torch.argmax(outputs, dim=1).cpu().numpy())

    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels)
    recall = recall_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels)

    return accuracy, precision, recall, f1

def plot_confusion_matrix(confusion_matrix, labels, title):
    plt.figure(figsize=(8, 6))
    sns.set(font_scale=1.2)
    sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title(title)
    plt.show()


def train_model_with_validation(model, train_loader, val_loader, num_epochs, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        true_labels = []
        predicted_labels = []

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)        
            optimizer.zero_grad()        
            outputs = model(images)        
            loss = criterion(outputs.view(-1,2), labels)        
            loss.backward()        
            optimizer.step()        
            running_loss += loss.item()        
            true_labels.extend(labels.cpu().numpy())        
            predicted_labels.extend(torch.argmax(outputs, dim=1).cpu().numpy())
           
            del loss,outputs,images,labels
            torch.cuda.empty_cache()
           
        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = accuracy_score(true_labels, predicted_labels)
        train_confusion = confusion_matrix(true_labels, predicted_labels)

         # Validation
        val_accuracy, val_precision, val_recall, val_f1 = validate_model(model, val_loader, device)
        val_true_labels = []        
        val_predicted_labels = []
        with torch.no_grad():            
            for val_images, val_labels in val_loader:                
                val_images, val_labels = val_images.to(device), val_labels.to(device)                
                val_outputs = model(val_images)
                val_true_labels.extend(val_labels.cpu().numpy())
                val_predicted_labels.extend(torch.argmax(val_outputs, dim=1).cpu().numpy())                
       
        val_confusion = confusion_matrix(val_true_labels, val_predicted_labels)



        print(f'Epoch [{epoch+1}/{num_epochs}] - '
              f'Training Loss: {epoch_loss:.4f} - Training Accuracy: {epoch_accuracy:.4f} - '
              f'Validation Accuracy: {val_accuracy:.4f} - Validation Precision: {val_precision:.4f} - '
              f'Validation Recall: {val_recall:.4f} - Validation F1: {val_f1:.4f}')
        print()
        print(f'Confusion Matrix (Training):\n', train_confusion)
        plot_confusion_matrix(train_confusion, labels=["Real", "Fake"], title="Training Confusion Matrix")
        print()
        print(f'Confusion Matrix (Validation):\n', val_confusion)
        plot_confusion_matrix(val_confusion, labels=["Real", "Fake"], title="Validation Confusion Matrix")

    print('Training/Validation complete.')


def collate_fn(batch):    
    frames, labels = zip(*batch)    
    padded_frames = pad_sequence(frames, batch_first=True).to(torch.float32).to(device)    
    return padded_frames, torch.tensor(labels)

if __name__ == '__main__':
    metadata = load_metadata(metadata_path)
   
    # Filter videos for fake and real labels
    fake_videos = [filename for filename, entry in metadata.items() if entry['label'] == 'FAKE']
    real_videos = [filename for filename, entry in metadata.items() if entry['label'] == 'REAL']
   
    # Randomly shuffle the lists
    random.shuffle(fake_videos)
    random.shuffle(real_videos)
   
    # Select 40 fake and 40 real videos for training, and 10 fake and 10 real videos for validation
    train_fake_videos = fake_videos[:40]
    train_real_videos = real_videos[:40]
    val_fake_videos = fake_videos[40:50]
    val_real_videos = real_videos[40:50]
   
    # Combine the selected videos for training and validation
    train_videos = train_fake_videos + train_real_videos
    print('No.Training videos: ',{len(train_videos)}, 'which consist of  ',{len(train_fake_videos)},'fake and ',{len(train_real_videos)},'real video.')
    val_videos = val_fake_videos + val_real_videos
    print('No.Validation videos: ',{len(val_videos)}, 'which consist of  ',{len(val_fake_videos)},'fake and ',{len(val_real_videos)},'real video.')
   

    # Create training and validation metadata dictionaries
    train_metadata = {video_filename: metadata[video_filename] for video_filename in train_videos}
    val_metadata = {video_filename: metadata[video_filename] for video_filename in val_videos}
   
    # Create the dataset and data loaders for training and validation
    train_dataset = DeepfakeDataset(dataset_dir, train_metadata, transform=transform)
    val_dataset = DeepfakeDataset(dataset_dir, val_metadata, transform=transform)
   
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn)
    val_data_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)

    # ... (creating and training the model)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleCNN(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
   
    model.to(device)
    # Train the model with validation
    train_model_with_validation(model, train_data_loader, val_data_loader, num_epochs, device)
