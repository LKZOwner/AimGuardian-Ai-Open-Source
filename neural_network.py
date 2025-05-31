import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda.amp as amp

class LearningAI(nn.Module):
    def __init__(self, input_channels=3, num_classes=2):  # 2 classes: normal aim vs aimbot
        super(LearningAI, self).__init__()
        
        # Enable automatic mixed precision
        self.scaler = amp.GradScaler()
        
        # Feature extraction layers with GPU optimization
        self.features = nn.Sequential(
            # First conv block - detect basic patterns
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),  # inplace for memory efficiency
            nn.MaxPool2d(2),
            
            # Second conv block - detect aim patterns
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Third conv block - detect complex patterns
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Fourth conv block - detect aimbot signatures
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Fifth conv block - final feature extraction
            nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))  # Global average pooling
        )
        
        # Classification layers
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
        # Initialize weights for better GPU performance
        self._init_weights()
        
        # Training setup with modern optimizations
        self.learning_rate = 0.001
        self.optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=0.01,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Learning rate scheduler with warmup
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.learning_rate,
            epochs=100,
            steps_per_epoch=100,
            pct_start=0.1,
            anneal_strategy='cos'
        )
        
        # Move model to GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    @torch.cuda.amp.autocast()  # Enable automatic mixed precision
    def forward(self, x):
        # Feature extraction
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        
        # Classification
        x = self.classifier(x)
        return x
    
    def train_step(self, inputs, targets):
        self.optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
        
        # Forward pass with mixed precision
        with amp.autocast():
            outputs = self(inputs)
            ce_loss = F.cross_entropy(outputs, targets)
            probs = F.softmax(outputs, dim=1)
            entropy_loss = -torch.mean(torch.sum(probs * torch.log(probs + 1e-10), dim=1))
            loss = ce_loss + 0.1 * entropy_loss
        
        # Backward pass with gradient scaling
        self.scaler.scale(loss).backward()
        
        # Gradient clipping
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        
        # Optimizer step with gradient scaling
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        # Update learning rate
        self.scheduler.step()
        
        return loss.item()
    
    @torch.no_grad()  # Disable gradient calculation for inference
    @torch.cuda.amp.autocast()  # Enable automatic mixed precision
    def predict(self, inputs):
        self.eval()
        outputs = self(inputs)
        probabilities = F.softmax(outputs, dim=1)
        predictions = torch.argmax(probabilities, dim=1)
        return predictions, probabilities
    
    def save_model(self, path):
        torch.save({
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict()
        }, path)
    
    def load_model(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        self.eval() 