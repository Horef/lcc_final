"""
This module defines the Encoder class and its subclasses for learning encoders
from semantic vectors to voxel activations.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sklearn.linear_model
import scipy.stats
from tqdm.auto import tqdm


class Encoder():
    ''' Base class for encoders. '''
    def __init__(self, input_dim, output_dim):
        pass
    
    def learn_encoder(self, voxels, vectors):
        """ Learn the encoder from voxel activations and semantic vectors. """
        raise NotImplementedError("This method should be overridden by subclasses.")

    def encode(self, vectors):
        """ Encode vectors into voxel activations. """
        raise NotImplementedError("This method should be overridden by subclasses.")

def init_and_train_encoder(enc, voxels, vectors, **kwargs):
    """
    Used to initialize and train an encoder.
    Args:
        enc (Encoder): The encoder class or instance to be initialized and trained.
        voxels (np.ndarray): Voxel activations, shape (C, V) where C is the number of concepts and V is the number of voxels.
        vectors (np.ndarray): Semantic vectors, shape (C, D) where D is the number of semantic dimensions.
        **kwargs: Additional keyword arguments for encoder initialization (Will be passed to the encoder's constructor).
    Returns:
        model (Encoder): The trained encoder instance.
    """

    input_dim = vectors.shape[1]
    output_dim = voxels.shape[1]
    
    # If 'enc' is a class, instantiate it; if it's already an instance, use it directly
    if isinstance(enc, type):
        model = enc(input_dim, output_dim, **kwargs)
    else:
        model = enc
    model.learn_encoder(voxels, vectors)

    return model

def test_encoder(enc, voxels, vectors, fold_n, seed=3, ext_test=False, ext_test_voxels=None, ext_test_vectors=None, **kwargs):
    """
    Used to test an encoder on a dataset using k-fold cross-validation.
    Args:
        enc (Encoder): The encoder class or instance to be tested.
        voxels (np.ndarray): Voxel activations, shape (C, V) where C is the number of concepts and V is the number of voxels.
        vectors (np.ndarray): Semantic vectors, shape (C, D) where D is the number of semantic dimensions.
        fold_n (int): Number of folds for cross-validation.
        seed (int): Random seed for reproducibility.
        ext_test (bool): If True, will use external test set, which should be passed in `ext_test_voxels` and `ext_test_vectors`.
        ext_test_voxels (np.ndarray, optional): External test voxel activations.
        ext_test_vectors (np.ndarray, optional): External test semantic vectors.
        **kwargs: Additional keyword arguments for encoder initialization.  (Will be passed to the encoder's constructor.)
    Returns:
        entity_accuracies (dict): A dictionary mapping entity IDs to their accuracy scores.
        fold_accuracies (list): A list of average accuracy scores for each fold.
    """

    # For the sake of reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)

    n_samples = voxels.shape[0]
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    folds = np.array_split(indices, fold_n)

    entity_accuracies = {}
    fold_accuracies = []

    for test_id, test_fold in tqdm(enumerate(folds), total=fold_n, desc=f"Testing {enc.__name__} on Folds", unit="fold"):
        train_folds = [fold for i, fold in enumerate(folds) if i != test_id]
        train_indices = np.concatenate(train_folds)
        test_indices = test_fold

        train_voxels = voxels[train_indices]
        test_voxels = voxels[test_indices]

        train_vectors = vectors[train_indices]
        test_vectors = vectors[test_indices]

        # If using external test set, override test_voxels and test_vectors
        if ext_test:
            test_voxels = ext_test_voxels
            test_vectors = ext_test_vectors

        # Train the encoder
        encoder = init_and_train_encoder(enc, train_voxels, train_vectors, **kwargs)

        # Get predictions
        predictions = encoder.encode(test_vectors)

        # Compute accuracy scores
        accuracy_scores = []
        # for each prediction, sorting all vectors by pearson correlation to the prediction
        for i in range(predictions.shape[0]):
            pred = predictions[i]
            target = test_voxels[i]
            # Compute pearson correlation with all vectors
            correlations = np.array([scipy.stats.pearsonr(pred, vox)[0] for vox in test_voxels])
            # Sort by similarity
            sorted_indices = np.argsort(correlations)[::-1]
            # Get the index of the target voxel vector in the sorted list
            target_index = np.where(np.all(test_voxels[sorted_indices] == target, axis=1))[0][0] + 1
            accuracy_scores.append(target_index / len(test_voxels))
            # Store the accuracy score for the concept
            entity_id = test_indices[i]
            entity_accuracies[entity_id] = target_index / len(test_voxels)
        # Compute the average accuracy score
        avg_accuracy = np.mean(accuracy_scores)
        fold_accuracies.append(avg_accuracy)

    return entity_accuracies, fold_accuracies


def calculate_optimal_num_heads(input_dim, max_heads=16, preferred_heads=None):
    """
    Calculate the optimal number of attention heads for a given input dimension.
    
    Args:
        input_dim (int): The input dimension
        max_heads (int): Maximum number of heads to consider
        preferred_heads (int, optional): Preferred number of heads (will find closest valid)
    
    Returns:
        int: Optimal number of heads that divides input_dim
    """
    if preferred_heads is not None:
        # If a specific number is preferred, find the closest valid one
        if input_dim % preferred_heads == 0:
            return preferred_heads
        
        # Find the closest valid number of heads
        possible_heads = [h for h in range(1, input_dim + 1) if input_dim % h == 0]
        return min(possible_heads, key=lambda x: abs(x - preferred_heads))
    
    # Find the largest divisor of input_dim that's <= max_heads
    possible_heads = [h for h in range(1, min(max_heads + 1, input_dim + 1)) if input_dim % h == 0]
    return max(possible_heads) if possible_heads else 1


class RidgeEncoder(Encoder):
    """ Encoder using ridge regression. """
    def __init__(self, input_dim, output_dim, seed=3):
        super(RidgeEncoder, self).__init__(input_dim, output_dim)
        self.ridge = sklearn.linear_model.RidgeCV(
            alphas=np.array([1, 10, .01, 100, .001, 1000, .0001, 10000, .00001, 100000, .000001, 1000000]),
            fit_intercept=True,
            alpha_per_target=True,
            gcv_mode='auto'
        )

    def learn_encoder(self, voxels, vectors):
        """ Learn the encoder using ridge regression. """
        self.ridge.fit(vectors, voxels)
        self.coef_ = self.ridge.coef_.T

    def encode(self, vectors):
        """ Encode vectors into voxel activations using the learned encoder. """
        return self.ridge.predict(vectors)

class CorrelationLoss(nn.Module):
    """
    Loss function that computes 1 - Pearson correlation coefficient as the loss.
    """
    def __init__(self, epsilon=1e-8):
        super(CorrelationLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, pred, target):
        pred = pred - pred.mean(dim=1, keepdim=True)
        target = target - target.mean(dim=1, keepdim=True)
        pred_norm = pred / (pred.norm(dim=1, keepdim=True) + self.epsilon)
        target_norm = target / (target.norm(dim=1, keepdim=True) + self.epsilon)
        corr = (pred_norm * target_norm).sum(dim=1)
        return 1 - corr.mean()

class FCEncoder(nn.Module, Encoder):
    def __init__(self, input_dim, output_dim, seed=3, metric='mse', epsilon=1e-8):
        """ 
        A simple fully connected encoder. 
        
        Args:
            input_dim (int): The dimension of the input semantic vectors.
            output_dim (int): The dimension of the output voxel activations.
            seed (int): Random seed for reproducibility.
            metric (str): Loss function to use ('mse' or 'corr' for pearson correlation). Default is 'mse'.
            epsilon (float): Small value to avoid division by zero. Default is 1e-8.
        Raises:
            ValueError: If the metric is not 'mse' or 'corr'.
        """
        if metric not in ['mse', 'corr']:
            raise ValueError("Metric must be 'mse' or 'corr'.")

        super(FCEncoder, self).__init__()
        self.seed = seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.metric = metric
        self.epsilon = epsilon
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc(x)
    
    def learn_encoder(self, voxels, vectors, epochs=1000, learning_rate=0.01):
        """ Given voxels (a CxV matrix of V voxel activations per C concepts)
        and vectors (a CxD matrix of D semantic dimensions per C concepts)
        find a matrix M such that the dot product of a D-dimensional semantic vector
        and M gives a V-dimensional encoded voxel activations.
        
        The matrix M is learned using a simple neural network.
        """
        
        # Define the loss function based on the metric
        if self.metric == 'mse':
            criterion = nn.MSELoss()
        elif self.metric == 'corr':
            criterion = CorrelationLoss(self.epsilon)
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        for epoch in range(epochs):
            self.train()
            optimizer.zero_grad()
            outputs = self(torch.tensor(vectors, dtype=torch.float32))
            loss = criterion(outputs, torch.tensor(voxels, dtype=torch.float32))
            loss.backward()
            optimizer.step()

    def encode(self, vectors):
        """ Encode vectors into voxel activations using the learned encoder. """
        self.eval()
        with torch.no_grad():
            return self(torch.tensor(vectors, dtype=torch.float32)).numpy()
        
class FC2Encoder(nn.Module, Encoder):
    """ A more complicated encoder using fully connected layers. """
    def __init__(self, input_dim, output_dim, seed=3, metric='mse', epsilon=1e-8):
        """
        Initialize the FC2Encoder.

        Args:
            input_dim (int): The dimension of the input semantic vectors.
            output_dim (int): The dimension of the output voxel activations.
            seed (int): Random seed for reproducibility.
            metric (str): Loss function to use ('mse' or 'corr' for pearson correlation). Default is 'mse'.
            epsilon (float): Small value to avoid division by zero. Default is 1e-8.
        
        Raises:
            ValueError: If the metric is not 'mse' or 'corr'.
        """
        if metric not in ['mse', 'corr']:
            raise ValueError("Metric must be 'mse' or 'corr'.")

        super(FC2Encoder, self).__init__()
        self.seed = seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.metric = metric
        self.epsilon = epsilon
        self.fc1 = nn.Linear(input_dim, (input_dim + output_dim) // 2)
        self.fc2 = nn.Linear((input_dim + output_dim) // 2, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.fc2(x)
        return x

    def learn_encoder(self, voxels, vectors, epochs=1000, learning_rate=0.01):
        """ Learn the encoder using fully connected layers. """

        # Define the loss function based on the metric
        if self.metric == 'mse':
            criterion = nn.MSELoss()
        elif self.metric == 'corr':
            criterion = CorrelationLoss(self.epsilon)
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        for epoch in range(epochs):
            self.train()
            optimizer.zero_grad()
            outputs = self(torch.tensor(vectors, dtype=torch.float32))
            loss = criterion(outputs, torch.tensor(voxels, dtype=torch.float32))
            loss.backward()
            optimizer.step()

    def encode(self, vectors):
        """ Encode vectors into voxel activations using the learned encoder. """
        self.eval()
        with torch.no_grad():
            return self(torch.tensor(vectors, dtype=torch.float32)).numpy()


class ResidualEncoder(nn.Module, Encoder):
    """ Encoder with residual connections for better gradient flow. """
    def __init__(self, input_dim, output_dim, hidden_dim=None, seed=3):
        super(ResidualEncoder, self).__init__()
        self.seed = seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        if hidden_dim is None:
            hidden_dim = max(input_dim, output_dim)
        
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.hidden1 = nn.Linear(hidden_dim, hidden_dim)
        self.hidden2 = nn.Linear(hidden_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.2)
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        # Initial projection
        x = self.input_proj(x)
        x = nn.ReLU()(x)
        
        # First residual block
        residual = x
        x = self.layer_norm1(x)
        x = self.hidden1(x)
        x = nn.ReLU()(x)
        x = self.dropout(x)
        x = x + residual  # Residual connection
        
        # Second residual block
        residual = x
        x = self.layer_norm2(x)
        x = self.hidden2(x)
        x = nn.ReLU()(x)
        x = self.dropout(x)
        x = x + residual  # Residual connection
        
        # Output projection
        x = self.output_proj(x)
        return x

    def learn_encoder(self, voxels, vectors, epochs=2000, learning_rate=0.001):
        """ Learn the encoder with advanced training techniques. """
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(self.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=100)

        for epoch in range(epochs):
            self.train()
            optimizer.zero_grad()
            outputs = self(torch.tensor(vectors, dtype=torch.float32))
            loss = criterion(outputs, torch.tensor(voxels, dtype=torch.float32))
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step(loss)

    def encode(self, vectors):
        """ Encode vectors into voxel activations using the learned encoder. """
        self.eval()
        with torch.no_grad():
            return self(torch.tensor(vectors, dtype=torch.float32)).numpy()


class AttentionEncoder(nn.Module, Encoder):
    """ Encoder with attention mechanism to focus on important semantic dimensions. """
    def __init__(self, input_dim, output_dim, num_heads=None, seed=3, metric='mse', epsilon=1e-8):
        """
        Initialize the AttentionEncoder.

        Args:
            input_dim (int): The dimension of the input semantic vectors.
            output_dim (int): The dimension of the output voxel activations.
            num_heads (int, optional): Number of attention heads. If None, will be calculated dynamically based on input_dim.
            seed (int): Random seed for reproducibility.
            metric (str): Loss function to use ('mse' or 'corr' for pearson correlation). Default is 'mse'.
            epsilon (float): Small value to avoid division by zero. Default is 1e-8.
        
        Raises:
            ValueError: If the metric is not 'mse' or 'corr'.
        """
        if metric not in ['mse', 'corr']:
            raise ValueError("Metric must be 'mse' or 'corr'.")

        super(AttentionEncoder, self).__init__()
        self.seed = seed
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.metric = metric
        self.epsilon = epsilon
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Dynamically calculate num_heads to ensure input_dim is divisible by num_heads
        self.num_heads = calculate_optimal_num_heads(input_dim, preferred_heads=num_heads)
        
        # Multi-head attention
        self.multihead_attn = nn.MultiheadAttention(input_dim, self.num_heads, batch_first=True)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(input_dim * 2, input_dim)
        )
        
        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.layer_norm2 = nn.LayerNorm(input_dim)
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(input_dim, output_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(output_dim * 2, output_dim)
        )

    def forward(self, x):
        # Add sequence dimension for attention (batch_size, seq_len=1, features)
        x = x.unsqueeze(1)
        
        # Self-attention
        attn_output, _ = self.multihead_attn(x, x, x)
        x = self.layer_norm1(x + attn_output)
        
        # Feed-forward network
        ffn_output = self.ffn(x)
        x = self.layer_norm2(x + ffn_output)
        
        # Remove sequence dimension
        x = x.squeeze(1)
        
        # Output projection
        x = self.output_proj(x)
        return x

    def learn_encoder(self, voxels, vectors, epochs=2000, learning_rate=0.0005):
        """ Learn the encoder with attention mechanism. """

        # Define the loss function based on the metric
        if self.metric == 'mse':
            criterion = nn.MSELoss()
        elif self.metric == 'corr':
            criterion = CorrelationLoss(epsilon=self.epsilon)

        optimizer = optim.AdamW(self.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        for epoch in range(epochs):
            self.train()
            optimizer.zero_grad()
            outputs = self(torch.tensor(vectors, dtype=torch.float32))
            loss = criterion(outputs, torch.tensor(voxels, dtype=torch.float32))
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()

    def encode(self, vectors):
        """ Encode vectors into voxel activations using the learned encoder. """
        self.eval()
        with torch.no_grad():
            return self(torch.tensor(vectors, dtype=torch.float32)).numpy()


class ImprovedAttentionEncoder(nn.Module, Encoder):
    """ Enhanced attention encoder with multiple improvements """
    def __init__(self, input_dim, output_dim, num_heads=None, num_layers=2, seed=3):
        super(ImprovedAttentionEncoder, self).__init__()
        self.seed = seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        
        # Dynamically calculate num_heads to ensure input_dim is divisible by num_heads
        self.num_heads = calculate_optimal_num_heads(input_dim, preferred_heads=num_heads)
        
        # Multiple attention layers
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(input_dim, self.num_heads, batch_first=True, dropout=0.1)
            for _ in range(num_layers)
        ])
        
        # Layer normalization for each attention layer
        self.layer_norms1 = nn.ModuleList([
            nn.LayerNorm(input_dim) for _ in range(num_layers)
        ])
        self.layer_norms2 = nn.ModuleList([
            nn.LayerNorm(input_dim) for _ in range(num_layers)
        ])
        
        # Feed-forward networks for each layer
        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, input_dim * 4),
                nn.GELU(),  # GELU often works better than ReLU for attention
                nn.Dropout(0.1),
                nn.Linear(input_dim * 4, input_dim),
                nn.Dropout(0.1)
            ) for _ in range(num_layers)
        ])
        
        # Output projection with intermediate layer
        self.output_proj = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.LayerNorm(input_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(input_dim, output_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(output_dim * 2, output_dim)
        )
        
        # Positional encoding (even though we have only one "token", it helps)
        self.positional_encoding = nn.Parameter(torch.randn(1, 1, input_dim) * 0.02)

    def forward(self, x):
        # Add sequence dimension and positional encoding
        x = x.unsqueeze(1) + self.positional_encoding
        
        # Apply multiple attention layers
        for i in range(self.num_layers):
            # Self-attention with residual connection
            attn_output, _ = self.attention_layers[i](x, x, x)
            x = self.layer_norms1[i](x + attn_output)
            
            # Feed-forward with residual connection
            ffn_output = self.ffns[i](x)
            x = self.layer_norms2[i](x + ffn_output)
        
        # Remove sequence dimension
        x = x.squeeze(1)
        
        # Output projection
        x = self.output_proj(x)
        return x

    def learn_encoder(self, voxels, vectors, epochs=3000, learning_rate=0.0003):
        """ Enhanced training with better techniques """
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(self.parameters(), lr=learning_rate, weight_decay=1e-4)
        
        # Warm-up + cosine annealing schedule
        warm_up_steps = epochs // 10
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=learning_rate, steps_per_epoch=1, epochs=epochs,
            pct_start=warm_up_steps/epochs
        )

        for epoch in range(epochs):
            self.train()
            optimizer.zero_grad()
            outputs = self(torch.tensor(vectors, dtype=torch.float32))
            loss = criterion(outputs, torch.tensor(voxels, dtype=torch.float32))
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.5)
            
            optimizer.step()
            scheduler.step()

    def encode(self, vectors):
        """ Encode vectors into voxel activations """
        self.eval()
        with torch.no_grad():
            return self(torch.tensor(vectors, dtype=torch.float32)).numpy()


class VAEEncoder(nn.Module, Encoder):
    """ Variational Autoencoder-style encoder for more robust representations. """
    def __init__(self, input_dim, output_dim, latent_dim=None, seed=3, metric='mse', epsilon=1e-8):
        """
        Initialize the VAEEncoder.
        Args:
            input_dim (int): The dimension of the input semantic vectors.
            output_dim (int): The dimension of the output voxel activations.
            latent_dim (int, optional): The dimension of the latent space. If None, defaults to half the min(input_dim, output_dim).
            seed (int): Random seed for reproducibility.
            metric (str): Loss function to use ('mse' or 'corr' for pearson correlation). Default is 'mse'.
            epsilon (float): Small value to avoid division by zero. Default is 1e-8.
        Raises:
            ValueError: If the metric is not 'mse' or 'corr'.
        """
        if metric not in ['mse', 'corr']:
            raise ValueError("Metric must be 'mse' or 'corr'.")
        
        super(VAEEncoder, self).__init__()
        self.seed = seed
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.metric = metric
        self.epsilon = epsilon
        
        if latent_dim is None:
            latent_dim = min(input_dim, output_dim) // 2
        
        self.latent_dim = latent_dim
        
        # Encoder to latent space
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(input_dim // 2, latent_dim * 2)  # *2 for mean and logvar
        )
        
        # Decoder from latent space
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, output_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(output_dim // 2, output_dim)
        )

    def reparameterize(self, mu, logvar):
        """ Reparameterization trick for VAE. """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # Encode to latent space
        encoded = self.encoder(x)
        mu, logvar = encoded[:, :self.latent_dim], encoded[:, self.latent_dim:]
        
        # Reparameterize
        z = self.reparameterize(mu, logvar)
        
        # Decode to output space
        output = self.decoder(z)
        
        return output, mu, logvar

    def learn_encoder(self, voxels, vectors, epochs=2000, learning_rate=0.001, beta=0.1):
        """ Learn the VAE encoder with KL divergence regularization. """
        optimizer = optim.AdamW(self.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=150)

        for epoch in range(epochs):
            self.train()
            optimizer.zero_grad()
            
            outputs, mu, logvar = self(torch.tensor(vectors, dtype=torch.float32))
            
            # Reconstruction loss
            if self.metric == 'mse':
                recon_loss = nn.MSELoss()(outputs, torch.tensor(voxels, dtype=torch.float32))
            elif self.metric == 'corr':
                recon_loss = CorrelationLoss(self.epsilon)(outputs, torch.tensor(voxels, dtype=torch.float32))
            
            # KL divergence loss
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            kl_loss = kl_loss / mu.size(0)  # Normalize by batch size
            
            # Total loss
            loss = recon_loss + beta * kl_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step(loss)

    def encode(self, vectors):
        """ Encode vectors into voxel activations using the learned encoder. """
        self.eval()
        with torch.no_grad():
            outputs, _, _ = self(torch.tensor(vectors, dtype=torch.float32))            
            return outputs.numpy()