import torch
import torch.nn as nn

def get_checkpoint_path():
    """Return the path to save the best performing model checkpoint.
    
    Returns:
        checkpoint_path (str)
            The path to save the best performing model checkpoint
    """
    return 'best_model_checkpoint.pth'

class LinearRegression(nn.Module):
    def __init__(self, input_dim=1, output_dim=1):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)

def create_loss_and_optimizer(model):
    """Create and return a loss function and optimizer.
    
    Parameters:
        model (torch.nn.Module)
            A neural network
    
    Returns:
        loss_fn (function)
            The loss function for the model
        optimizer (torch.optim.Optimizer)
            The optimizer for the model
    """
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    return loss_fn, optimizer

def train(x, y, model, loss_fn, optimizer, checkpoint_path, num_epochs=1000):
    """Train a model.
    
    Parameters:
        x (torch.Tensor)
            The input data
        y (torch.Tensor)
            The expected output data
        model (torch.nn.Module)
            A neural network
        loss_fn (function)
            The loss function
        optimizer (torch.optim.Optimizer)
            The optimizer for the model
        checkpoint_path (str)
            The path to save the best performing checkpoint
        num_epochs (int)
            The number of epochs to train for
    
    Side Effects:
        - Save the best performing model checkpoint to `checkpoint_path`
    """
    best_loss = float('inf')
    model.train()
    for _ in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(x)
        loss = loss_fn(outputs, y)
        
        loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            current_outputs = model(x)
            current_loss = loss_fn(current_outputs, y).item()
        
        # Save the best model
        if current_loss < best_loss:
            best_loss = current_loss
            torch.save(model.state_dict(), checkpoint_path)

def load_model_checkpoint(checkpoint_path):
    """Load a model checkpoint from disk.

    Parameters:
        checkpoint_path (str)
            The path to load the checkpoint from
    
    Returns:
        model (torch.nn.Module)
            The model loaded from the checkpoint
    """
    model = LinearRegression()
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()
    return model