# secure_layer.py  (modified from your original)

import torch
import numpy as np
from cryptography.fernet import Fernet
import hashlib
import hmac
import pickle
from sklearn.ensemble import IsolationForest

class SecurityException(Exception):
    pass

class DataValidator:
    def __init__(self):
        self.detector = None
        self.reference_data = None

    def fit_reference(self, reference_tensors):
        """
        Call this at setup time with a list/array of reference samples (unbatched tensors).
        Example: reference_tensors = [t1.cpu().numpy().ravel(), t2.cpu().numpy().ravel(), ...]
        """
        import numpy as np
        X = np.stack(reference_tensors, axis=0)
        self.detector = IsolationForest(contamination=0.01)
        self.detector.fit(X)
        self.reference_data = True

    def validate(self, input_tensor):
        # move to CPU numpy for checks
        t = input_tensor.detach().cpu()
        if not torch.isfinite(t).all():
            return False, "Non-finite values in input"

        # value-range check assuming your inputs are normalized in [-1,1]
        if (t < -1.0).any() or (t > 1.0).any():
            return False, "Input values out of expected range [-1,1]"

        # optional anomaly detection (requires fit_reference called earlier)
        if self.detector is not None:
            x = t.numpy().ravel().reshape(1, -1)
            pred = self.detector.predict(x)[0]  # 1 = normal, -1 = outlier
            if pred == -1:
                return False, "IsolationForest flagged input as outlier"

        return True, "OK"


class AdversarialDefender:
    def __init__(self, noise_std=0.01, bit_depth=8):
        self.noise_std = noise_std
        self.bit_depth = bit_depth

    def defend(self, input_tensor):
        protected = input_tensor + torch.randn_like(input_tensor) * self.noise_std

        if self.bit_depth < 8:
            scale = (2 ** self.bit_depth - 1)
            protected = (protected * scale).round() / scale

        protected = torch.clamp(protected, -1.0, 1.0)
        return protected


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
        ok = hmac.compare_digest(current_hash, self.model_hash)
        return ok

    def secure_predict(self, input_data):
        """
        Returns a tuple: (output_tensor, secure_bool, reason_str).
        It never raises except for unexpected internal errors.
        """
        # 1) integrity
        if not self.verify_model_integrity():
            return None, False, "Model integrity compromised"

        # 2) validate input
        ok, reason = self.data_validator.validate(input_data)
        if not ok:
            return None, False, f"Invalid input: {reason}"

        # 3) defend (optional adversarial noise / quantize)
        protected_input = self.adversarial_defender.defend(input_data)

        # 4) forward
        with torch.no_grad():
            output = self.model(protected_input.unsqueeze(0))  # keep batch dim
        return output, True, "OK"
