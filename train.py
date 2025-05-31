import torch
import numpy as np
from neural_network import LearningAI
from utils import prepare_data, evaluate_model, create_batches
import matplotlib.pyplot as plt

def train_model(model, X_train, y_train, X_test, y_test, 
                epochs=100, batch_size=32, learning_rate=0.001):
    """
    Train the model and return training history.
    """
    history = {
        'train_loss': [],
        'test_loss': [],
        'metrics': []
    }
    
    for epoch in range(epochs):
        model.train()
        epoch_losses = []
        
        # Training
        for X_batch, y_batch in create_batches(X_train, y_train, batch_size):
            loss = model.train_step(X_batch, y_batch)
            epoch_losses.append(loss)
        
        # Calculate average training loss
        avg_train_loss = np.mean(epoch_losses)
        history['train_loss'].append(avg_train_loss)
        
        # Evaluation
        metrics = evaluate_model(model, X_test, y_test)
        history['test_loss'].append(metrics['mse'])
        history['metrics'].append(metrics)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch + 1}/{epochs}')
            print(f'Training Loss: {avg_train_loss:.4f}')
            print(f'Test Loss: {metrics["mse"]:.4f}')
            print(f'RMSE: {metrics["rmse"]:.4f}\n')
    
    return history

def plot_training_history(history):
    """
    Plot the training history.
    """
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['test_loss'], label='Test Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot([m['rmse'] for m in history['metrics']], label='RMSE')
    plt.title('Model RMSE')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Example usage with synthetic data
    np.random.seed(42)
    n_samples = 1000
    n_features = 5
    
    # Generate synthetic data
    X = np.random.randn(n_samples, n_features)
    y = np.sum(X, axis=1) + np.random.randn(n_samples) * 0.1
    
    # Prepare data
    X_train, X_test, y_train, y_test, scaler = prepare_data(
        np.column_stack([X, y])
    )
    
    # Create and train model
    model = LearningAI(
        input_size=n_features,
        hidden_size=64,
        output_size=1
    )
    
    history = train_model(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        epochs=100,
        batch_size=32
    )
    
    # Plot training history
    plot_training_history(history)
    
    # Save the trained model
    model.save_model('trained_model.pth') 