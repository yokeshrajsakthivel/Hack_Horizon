import torch
import numpy as np
from cryptography.fernet import Fernet
import hashlib
import hmac
import pickle
from sklearn.ensemble import IsolationForest

class ModelSecurity:
    def __init__(self, model):
        self.model = model
        self.key = Fernet.generate_key()
        self.model_hash = self._compute_model_hash()
        self.data_validator = DataValidator()
        self.adversarial_defender = AdversarialDefender()
        
    def _compute_model_hash(self):
        model_bytes = pickle.dumps(self.model.state_dict())
        return hashlib.sha256(model_bytes).hexdigest()
    
    def verify_model_integrity(self):
        current_hash = self._compute_model_hash()
        return hmac.compare_digest(current_hash, self.model_hash)
    
    def secure_predict(self, input_data):
        if not self.verify_model_integrity():
            raise SecurityException("Model integrity compromised")
        
        if not self.data_validator.validate(input_data):
            raise SecurityException("Invalid input data detected")
        
        protected_input = self.adversarial_defender.defend(input_data)
        
        with torch.no_grad():
            output = self.model(protected_input)
        
        return output

class DataValidator:
    def __init__(self):
        self.detector = IsolationForest(contamination=0.01)
        self.reference_data = None
    
    def validate(self, input_tensor):
        input_tensor = input_tensor.cpu() if input_tensor.is_cuda else input_tensor
        input_np = input_tensor.numpy()
        
        if not torch.isfinite(input_tensor).all():
            return False
        
        if (input_tensor < -1).any() or (input_tensor > 1).any():
            return False
            
        if self.reference_data is not None:
            if self.detector.predict(input_np.reshape(1, -1))[0] == -1:
                return False
                
        return True

class AdversarialDefender:
    def __init__(self):
        self.noise_std = 0.01
        self.bit_depth = 5
    
    def defend(self, input_tensor):
        protected = input_tensor + torch.randn_like(input_tensor) * self.noise_std
        
        if self.bit_depth < 8:
            scale = (2 ** self.bit_depth - 1)
            protected = (protected * scale).round() / scale
            
        protected = torch.clamp(protected, -1, 1)
        
        return protected

class SecurityException(Exception):
    pass