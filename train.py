import os
from data_loader import VideoDataset
from content_moderator import ContentModerator
import numpy as np
from sklearn.model_selection import train_test_split

def main():
    # Set paths to your video directories
    violence_dir = os.path.join("dataset", "violence")
    nonviolence_dir = os.path.join("dataset", "nonviolence")
    
    # Check if directories exist
    if not os.path.exists(violence_dir):
        print(f"Error: Directory {violence_dir} does not exist!")
        return
    if not os.path.exists(nonviolence_dir):
        print(f"Error: Directory {nonviolence_dir} does not exist!")
        return
    
    # Initialize dataset
    dataset = VideoDataset(violence_dir, nonviolence_dir)
    
    # Process videos and get frames with labels
    print("Processing videos...")
    frames, labels = dataset.process_videos()
    
    # Split into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        frames, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    print(f"Training set size: {len(X_train)}")
    print(f"Validation set size: {len(X_val)}")
    
    # Initialize content moderator in training mode
    moderator = ContentModerator(train_mode=True)
    
    # Train the model
    print("\nTraining model...")
    moderator.train(X_train, y_train, batch_size=32, epochs=5)
    
    # Evaluate the model
    print("\nEvaluating model...")
    metrics = moderator.evaluate(X_val, y_val)
    
    print("\nValidation Metrics:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    
    # Save the trained model
    print("\nSaving model...")
    os.makedirs("models", exist_ok=True)
    moderator.model.save_pretrained("models/best_model")
    print("Model saved successfully!")

if __name__ == "__main__":
    main() 