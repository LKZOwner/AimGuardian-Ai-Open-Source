import torch
import numpy as np
from neural_network import LearningAI
from utils import prepare_data
import argparse

def main():
    parser = argparse.ArgumentParser(description='Learning AI System')
    parser.add_argument('--mode', choices=['train', 'predict'], required=True,
                      help='Mode to run the AI system in')
    parser.add_argument('--model_path', default='trained_model.pth',
                      help='Path to save/load the model')
    parser.add_argument('--data_path', help='Path to the data file (CSV)')
    parser.add_argument('--target_column', help='Target column name for training')
    parser.add_argument('--num_hidden_layers', type=int, default=50,
                      help='Number of hidden layers in the neural network (default: 50)')
    args = parser.parse_args()

    if args.mode == 'train':
        if not args.data_path or not args.target_column:
            print("Error: data_path and target_column are required for training mode")
            return

        # Load and prepare data
        import pandas as pd
        data = pd.read_csv(args.data_path)
        
        # Prepare data
        X_train, X_test, y_train, y_test, scaler = prepare_data(
            data, 
            target_column=args.target_column
        )
        
        # Create and train model
        model = LearningAI(
            input_size=X_train.shape[1],
            hidden_size=64,
            output_size=1,
            num_hidden_layers=args.num_hidden_layers
        )
        
        from train import train_model, plot_training_history
        history = train_model(
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test
        )
        
        # Plot training history
        plot_training_history(history)
        
        # Save the model
        model.save_model(args.model_path)
        print(f"Model saved to {args.model_path}")

    elif args.mode == 'predict':
        if not args.data_path:
            print("Error: data_path is required for predict mode")
            return

        # Load the model
        model = LearningAI(
            input_size=5,  # This should match your trained model
            hidden_size=64,
            output_size=1,
            num_hidden_layers=args.num_hidden_layers
        )
        model.load_model(args.model_path)
        
        # Load and prepare data
        import pandas as pd
        data = pd.read_csv(args.data_path)
        
        # Convert to tensor and normalize
        X = torch.FloatTensor(data.values)
        
        # Make predictions
        model.eval()
        with torch.no_grad():
            predictions = model(X)
        
        # Save predictions
        results = pd.DataFrame({
            'predictions': predictions.numpy().flatten()
        })
        results.to_csv('predictions.csv', index=False)
        print("Predictions saved to predictions.csv")

if __name__ == "__main__":
    main() 