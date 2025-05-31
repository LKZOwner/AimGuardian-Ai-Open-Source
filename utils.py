import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
import pandas as pd
from torchvision import transforms
from PIL import Image
import os
from sklearn.model_selection import train_test_split
from concurrent.futures import ThreadPoolExecutor
import queue
import threading
import time
from collections import deque
import torch.cuda.amp as amp
import cv2

def prepare_data(data, target_column=None, test_size=0.2):
    """
    Prepare data for training by splitting into features and targets,
    and normalizing the data.
    """
    if isinstance(data, pd.DataFrame):
        if target_column is None:
            raise ValueError("target_column must be specified when using DataFrame")
        X = data.drop(columns=[target_column])
        y = data[target_column]
    else:
        X = data[:, :-1]
        y = data[:, -1]
    
    # Convert to torch tensors
    X = torch.FloatTensor(X)
    y = torch.FloatTensor(y).reshape(-1, 1)
    
    # Normalize the data
    scaler = StandardScaler()
    X = torch.FloatTensor(scaler.fit_transform(X))
    
    # Split into train and test sets
    train_size = int(len(X) * (1 - test_size))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    return X_train, X_test, y_train, y_test, scaler

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model's performance on test data.
    """
    model.eval()
    correct = 0
    total = 0
    predictions = []
    probabilities = []
    
    with torch.no_grad():
        for X_batch, y_batch in create_batches(X_test, y_test, batch_size=32):
            outputs = model(X_batch)
            _, predicted = torch.max(outputs.data, 1)
            probs = torch.softmax(outputs, dim=1)
            
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
            predictions.extend(predicted.cpu().numpy())
            probabilities.extend(probs.cpu().numpy())
    
    accuracy = 100 * correct / total
    return {
        'accuracy': accuracy,
        'predictions': np.array(predictions),
        'probabilities': np.array(probabilities)
    }

def create_batches(X, y, batch_size=32):
    """
    Create batches for training.
    """
    n_samples = len(X)
    indices = np.random.permutation(n_samples)
    
    for i in range(0, n_samples, batch_size):
        batch_indices = indices[i:i + batch_size]
        yield X[batch_indices], y[batch_indices]

class FastImageLoader:
    def __init__(self, batch_size=32, num_workers=4, max_queue_size=100):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_queue = queue.Queue(maxsize=max_queue_size)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])
        self.running = False
        self.workers = []
        
    def _load_image(self, image_path):
        try:
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image)
            return image_tensor
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None
            
    def _worker(self, image_paths):
        for path in image_paths:
            if not self.running:
                break
            tensor = self._load_image(path)
            if tensor is not None:
                self.image_queue.put(tensor)
                
    def start_loading(self, image_paths):
        self.running = True
        # Split paths among workers
        chunk_size = len(image_paths) // self.num_workers
        chunks = [image_paths[i:i + chunk_size] for i in range(0, len(image_paths), chunk_size)]
        
        # Start worker threads
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [executor.submit(self._worker, chunk) for chunk in chunks]
            
    def stop_loading(self):
        self.running = False
        for worker in self.workers:
            worker.join()
            
    def get_batch(self):
        batch = []
        try:
            for _ in range(self.batch_size):
                tensor = self.image_queue.get(timeout=0.1)
                batch.append(tensor)
            return torch.stack(batch)
        except queue.Empty:
            return None

class FastImageProcessor:
    def __init__(self, model, batch_size=32, fps_target=100):
        self.model = model
        self.batch_size = batch_size
        self.fps_target = fps_target
        self.frame_time = 1.0 / fps_target
        self.image_loader = FastImageLoader(batch_size=batch_size)
        self.last_process_time = time.time()
        self.processed_count = 0
        self.fps_history = deque(maxlen=10)
        
    def process_images(self, image_paths):
        self.image_loader.start_loading(image_paths)
        results = []
        
        while True:
            current_time = time.time()
            elapsed = current_time - self.last_process_time
            
            # Maintain target FPS
            if elapsed < self.frame_time:
                time.sleep(self.frame_time - elapsed)
                
            batch = self.image_loader.get_batch()
            if batch is None:
                break
                
            # Process batch
            with torch.no_grad():
                predictions, probabilities = self.model.predict(batch)
                results.extend(predictions.cpu().numpy())
                
            self.processed_count += len(batch)
            self.last_process_time = time.time()
            
            # Calculate and update FPS
            if len(self.fps_history) > 0:
                current_fps = 1.0 / (current_time - self.last_process_time)
                self.fps_history.append(current_fps)
                
        self.image_loader.stop_loading()
        return np.array(results)
    
    def get_current_fps(self):
        if len(self.fps_history) > 0:
            return np.mean(self.fps_history)
        return 0

def prepare_dataset(data_dir, test_size=0.2, batch_size=32):
    """
    Prepare dataset from a directory of images with optimized loading.
    """
    normal_dir = os.path.join(data_dir, 'normal')
    aimbot_dir = os.path.join(data_dir, 'aimbot')
    
    # Get all image paths
    normal_paths = [os.path.join(normal_dir, f) for f in os.listdir(normal_dir)]
    aimbot_paths = [os.path.join(aimbot_dir, f) for f in os.listdir(aimbot_dir)]
    
    # Create labels
    normal_labels = [0] * len(normal_paths)
    aimbot_labels = [1] * len(aimbot_paths)
    
    # Combine paths and labels
    all_paths = normal_paths + aimbot_paths
    all_labels = normal_labels + aimbot_labels
    
    # Split into train and test
    train_paths, test_paths, train_labels, test_labels = train_test_split(
        all_paths, all_labels, test_size=test_size, random_state=42, stratify=all_labels
    )
    
    return train_paths, test_paths, train_labels, test_labels

def evaluate_model(model, test_paths, test_labels, batch_size=32):
    """
    Evaluate model performance with optimized processing.
    """
    processor = FastImageProcessor(model, batch_size=batch_size)
    predictions = processor.process_images(test_paths)
    
    accuracy = np.mean(predictions == test_labels)
    return {
        'accuracy': accuracy * 100,
        'fps': processor.get_current_fps(),
        'predictions': predictions
    }

def get_gpu_memory():
    """Get available GPU memory"""
    if torch.cuda.is_available():
        return torch.cuda.get_device_properties(0).total_memory
    return 0

def optimize_batch_size():
    """Calculate optimal batch size based on GPU memory"""
    gpu_mem = get_gpu_memory()
    if gpu_mem >= 12 * 1024 * 1024 * 1024:  # 12GB or more
        return 128
    elif gpu_mem >= 8 * 1024 * 1024 * 1024:  # 8GB or more
        return 64
    else:
        return 32

def preprocess_aimmy_screenshot(image):
    """
    Preprocess aimmy v2 screenshot with specific optimizations
    """
    # Convert to numpy for faster processing
    img_np = np.array(image)
    
    # 1. Crop to game view (assuming 16:9 aspect ratio)
    height, width = img_np.shape[:2]
    target_ratio = 16/9
    current_ratio = width/height
    
    if current_ratio > target_ratio:
        # Image is too wide
        new_width = int(height * target_ratio)
        start_x = (width - new_width) // 2
        img_np = img_np[:, start_x:start_x + new_width]
    elif current_ratio < target_ratio:
        # Image is too tall
        new_height = int(width / target_ratio)
        start_y = (height - new_height) // 2
        img_np = img_np[start_y:start_y + new_height, :]
    
    # 2. Optimize for aimmy's color space
    # Convert to YUV color space for better aim detection
    img_yuv = cv2.cvtColor(img_np, cv2.COLOR_RGB2YUV)
    
    # Enhance contrast in Y channel
    y_channel = img_yuv[:,:,0]
    y_channel = cv2.equalizeHist(y_channel)
    img_yuv[:,:,0] = y_channel
    
    # Convert back to RGB
    img_np = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
    
    return Image.fromarray(img_np)

def load_aimmy_screenshot(image_path, target_size=(224, 224), device='cuda'):
    """
    Load and preprocess an aimmy v2 screenshot with GPU acceleration
    """
    try:
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Apply aimmy-specific preprocessing
        image = preprocess_aimmy_screenshot(image)
        
        # GPU-accelerated transforms
        transform = transforms.Compose([
            transforms.Resize(target_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])
        
        # Convert to tensor and move to GPU
        image_tensor = transform(image).to(device, non_blocking=True)
        return image_tensor
    except Exception as e:
        print(f"Error loading aimmy screenshot {image_path}: {e}")
        return None

def prepare_aimmy_dataset(data_dir, test_size=0.2, device='cuda'):
    """
    Prepare dataset from aimmy v2 screenshots with GPU optimization
    """
    normal_dir = os.path.join(data_dir, 'normal_aim')
    aimbot_dir = os.path.join(data_dir, 'aimbot')
    
    # Get optimal batch size for GPU
    batch_size = optimize_batch_size()
    print(f"Using batch size: {batch_size} for GPU")
    
    # Enable automatic mixed precision
    scaler = amp.GradScaler()
    
    # Load normal aim screenshots
    normal_images = []
    normal_labels = []
    for img_name in os.listdir(normal_dir):
        if img_name.endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(normal_dir, img_name)
            img_tensor = load_aimmy_screenshot(img_path, device=device)
            if img_tensor is not None:
                normal_images.append(img_tensor)
                normal_labels.append(0)
    
    # Load aimbot screenshots
    aimbot_images = []
    aimbot_labels = []
    for img_name in os.listdir(aimbot_dir):
        if img_name.endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(aimbot_dir, img_name)
            img_tensor = load_aimmy_screenshot(img_path, device=device)
            if img_tensor is not None:
                aimbot_images.append(img_tensor)
                aimbot_labels.append(1)
    
    # Combine datasets
    all_images = normal_images + aimbot_images
    all_labels = normal_labels + aimbot_labels
    
    # Convert to tensors (already on GPU)
    X = torch.stack(all_images)
    y = torch.tensor(all_labels, dtype=torch.long, device=device)
    
    # Split into train and test sets
    indices = torch.randperm(len(X), device=device)
    test_size = int(len(X) * test_size)
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]
    
    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]
    
    print(f"Loaded {len(normal_images)} normal aim screenshots")
    print(f"Loaded {len(aimbot_images)} aimbot screenshots")
    print(f"Using device: {device}")
    
    return X_train, X_test, y_train, y_test, scaler

def create_batches(X, y, batch_size=None, device='cuda'):
    """
    Create optimized batches for GPU training
    """
    if batch_size is None:
        batch_size = optimize_batch_size()
    
    n_samples = len(X)
    indices = torch.randperm(n_samples, device=device)
    
    for i in range(0, n_samples, batch_size):
        batch_indices = indices[i:i + batch_size]
        yield X[batch_indices], y[batch_indices]

def evaluate_model(model, X_test, y_test, batch_size=None, device='cuda'):
    """
    Evaluate model performance with GPU optimization
    """
    model.eval()
    correct = 0
    total = 0
    predictions = []
    probabilities = []
    
    if batch_size is None:
        batch_size = optimize_batch_size()
    
    with torch.no_grad(), amp.autocast():
        for X_batch, y_batch in create_batches(X_test, y_test, batch_size, device):
            outputs = model(X_batch)
            _, predicted = torch.max(outputs.data, 1)
            probs = torch.softmax(outputs, dim=1)
            
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
            predictions.extend(predicted.cpu().numpy())
            probabilities.extend(probs.cpu().numpy())
    
    accuracy = 100 * correct / total
    return {
        'accuracy': accuracy,
        'predictions': np.array(predictions),
        'probabilities': np.array(probabilities)
    } 