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
	- Loss train 4.5e-4, 3.2e-4
	- Prediction
		![[Pasted image 20241018151223.png]]
- ## LSTM
	- 
# 2. Data exploration
- ## Use transition time only V24
	- 40 - 140
		compared with fine tuned MLP V23
		- loss train 5.8e-4 valid 8.8e-4
		- plot
			![[Pasted image 20241018132137.png]]