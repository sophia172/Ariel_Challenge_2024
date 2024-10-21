# 1. Model exploration
- ## **MLP**
	- code
		``` python
		# Define the MLP model
	
		class MLPModel(nn.Module):
		
			def __init__(self, input_shape=(187, 283), output_size=283):
				super(MLPModel, self).__init__()
				# Flatten the input size
				
				input_size = input_shape[0] * input_shape[1]
				# Define the layers
				
				self.fc1 = nn.Linear(input_size, 1024) # First fully connected layer	
				self.fc2 = nn.Linear(1024, 512) # Second fully connected layer	
				self.fc3 = nn.Linear(512, output_size) # Output layer
				# Activation function (ReLU)
				
				self.relu = nn.ReLU()
				
			def forward(self, x):
			# Flatten the input from (187, 283) to (52921)
				x = x.view(x.size(0), -1) # Ensure the input is flattened
			
				# Forward 
				pass through the layers with ReLU activation
				
				x = self.relu(self.fc1(x))
				x = self.relu(self.fc2(x))
				x = self.fc3(x) # No activation on the final output layer
				return x
		```
	- Result Analysis
		- Input
			![[Pasted image 20241018002607.png]]
		- output gt
			![[Pasted image 20241018002648.png]]
		- output prediction
			![[Pasted image 20241018131534.png]]
		- loss train 3.44e-5, val 2.34e-4				
- ### Fine-tune MLP V24
	- normalise data and add activation
		- prediction
			![[Pasted image 20241018131756.png]]
		- loss  train 2.7e-4, val 6.2 e-4
- ## ViT
	- Loss train 3.2e-4, 3.1e-4
	- Prediction
		![[Pasted image 20241021124635.png]]
	- Debug: model callapse
		- Imbalanced dataset - ignore
		- Learning rate is too high >> Do not think this is the problem
		- Model is too simple 
		- Wrong Loss function >> MSE and crossEntropy both caused this problem
		- Model overfitting
		- Gradient Vanishing/Exploding >> **gradient vanished**
		- data preprocessing >> make sure data is normalised
		- Saturated output layer >> sigmoid or tanh might cause large/small output, use Relu to avoid this problem
- ## LSTM
- ## Denoising Autoencoder
	-  code template ```
		``` python
		import torch
		import torch.nn as nn
		import torch.optim as optim
		from torch.utils.data import DataLoader
		from torchvision import datasets, transforms
		import matplotlib.pyplot as plt
		import math
		
		
		class PatchEmbedding(nn.Module):
		    def __init__(self, img_size, patch_size, hidden_size):
		        super(PatchEmbedding, self).__init__()
		        self.img_height, self.img_width = img_size  # Non-square height and width
		        self.patch_height, self.patch_width = patch_size
		        self.num_patches = (self.img_height // self.patch_height) * (self.img_width // self.patch_width)
		        self.patch_dim = self.patch_height * self.patch_width  # Grayscale image: 1 channel
		        self.linear_embedding = nn.Linear(self.patch_dim, hidden_size)
		        
		    def forward(self, x):
		        # Break the image into patches
		        patches = x.unfold(2, self.patch_height, self.patch_height).unfold(3, self.patch_width, self.patch_width)
		        patches = patches.contiguous().view(x.size(0), -1, self.patch_height * self.patch_width)  # (batch, num_patches, patch_dim)
		        embeddings = self.linear_embedding(patches)
		        return embeddings
		
		
		class ViTEncoder(nn.Module):
		    def __init__(self, hidden_size, num_layers, num_heads):
		        super(ViTEncoder, self).__init__()
		        self.transformer_encoder = nn.TransformerEncoder(
		            nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads),
		            num_layers=num_layers
		        )
		    
		    def forward(self, x):
		        return self.transformer_encoder(x)
		
		
		class ViTDecoder(nn.Module):
		    def __init__(self, hidden_size, patch_size, img_size):
		        super(ViTDecoder, self).__init__()
		        self.patch_height, self.patch_width = patch_size
		        self.img_height, self.img_width = img_size
		        self.num_patches = (self.img_height // self.patch_height) * (self.img_width // self.patch_width)
		        self.linear_embedding = nn.Linear(hidden_size, self.patch_height * self.patch_width)
		    
		    def forward(self, x):
		        # Decode the patch embeddings back into image patches
		        patches = self.linear_embedding(x)
		        patches = patches.view(x.size(0), self.img_height // self.patch_height, self.img_width // self.patch_width, self.patch_height, self.patch_width)
		        # Reassemble the patches into the original image
		        reassembled_image = patches.permute(0, 1, 3, 2, 4).contiguous().view(x.size(0), 1, self.img_height, self.img_width)
		        return reassembled_image
		
		
		
		class ViTAutoencoder(nn.Module):
		    def __init__(self, img_size, patch_size, hidden_size, num_layers, num_heads):
		        super(ViTAutoencoder, self).__init__()
		        self.img_size = img_size
		        self.patch_size = patch_size
		        self.hidden_size = hidden_size
		        
		        # Patch embedding for non-square grayscale images
		        self.patch_embedding = PatchEmbedding(img_size, patch_size, hidden_size)
		        
		        # Positional encoding
		        self.positional_encoding = nn.Parameter(torch.randn(1, self.patch_embedding.num_patches, hidden_size))
		        
		        # Encoder and decoder
		        self.encoder = ViTEncoder(hidden_size, num_layers, num_heads)
		        self.decoder = ViTDecoder(hidden_size, patch_size, img_size)
		        
		    def forward(self, x):
		        # Embed the image into patches
		        patches = self.patch_embedding(x)
		        
		        # Add positional encoding
		        patches += self.positional_encoding
		        
		        # Encoder
		        encoded = self.encoder(patches)
		        
		        # Decoder
		        decoded = self.decoder(encoded)
		        
		        return decoded
		
		
		
		# Data loading for grayscale images (e.g., MNIST or custom dataset)
		transform = transforms.Compose([
		    transforms.Grayscale(),  # Ensure it's grayscale
		    transforms.ToTensor(),
		    transforms.Normalize((0.5,), (0.5,))
		])
		
		# Use a simple dataset like MNIST for demonstration
		train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
		train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
		
		# Initialize the model, loss function, and optimizer
		model = ViTAutoencoder(img_size=(28, 32), patch_size=(4, 4), hidden_size=64, num_layers=6, num_heads=8)
		criterion = nn.MSELoss()
		optimizer = optim.Adam(model.parameters(), lr=0.001)
		
		# Training loop
		num_epochs = 10
		for epoch in range(num_epochs):
		    running_loss = 0.0
		    for images, _ in train_loader:
		        optimizer.zero_grad()
		        outputs = model(images)
		        loss = criterion(outputs, images)
		        loss.backward()
		        optimizer.step()
		        running_loss += loss.item()
		    
		    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')
		
		
		# Function to display images
		def imshow(img, title):
		    img = img.squeeze().numpy()
		    plt.imshow(img, cmap='gray')
		    plt.title(title)
		    plt.show()
		
		# Get a batch of test data
		dataiter = iter(train_loader)
		images, _ = dataiter.next()
		
		# Pass the images through the model
		with torch.no_grad():
		    reconstructed_images = model(images)
		
		# Display the original and reconstructed images
		imshow(images[0], title='Original Image')
		imshow(reconstructed_images[0], title='Reconstructed Image')
		
		
		```
# 2. Data exploration
- ## Use transition time only V24
	- 40 - 140
		compared with fine tuned MLP V23
		- loss train 5.8e-4 valid 8.8e-4
		- plot
			![[Pasted image 20241018132137.png]]
- ## AIRS data flux intensity has nothing to do with the target amplitude or where it starts
	- AIRS data plot
		- ![[Pasted image 20241021004509.png]]
	- With Linear correlation plot
		- ![[Pasted image 20241021004630.png]]
- 