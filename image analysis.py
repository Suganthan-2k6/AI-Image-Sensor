import cv2
import numpy as np
from skimage import exposure, filters, feature
from scipy.fft import fft2, fftshift

def analyze_frequency(image_path):
    """Analyze frequency domain for AI artifacts"""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return {"error": "Could not read image"}
    
    # FFT Analysis
    f = fft2(img)
    fshift = fftshift(f)
    magnitude_spectrum = np.abs(fshift)
    
    # AI-generated images often have unnatural high-frequency patterns
    mean_freq = np.mean(magnitude_spectrum)
    std_freq = np.std(magnitude_spectrum)
    high_freq_ratio = np.sum(magnitude_spectrum > (mean_freq + 2 * std_freq)) / magnitude_spectrum.size
    
    return {
        "mean_frequency": float(mean_freq),
        "std_frequency": float(std_freq),
        "high_frequency_ratio": float(high_freq_ratio),
        "ai_likelihood": float(high_freq_ratio * 100)  # Higher = more likely AI
    }

def analyze_noise_consistency(image_path):
    """Check if noise pattern is consistent (AI images often have uniform noise)"""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return {"error": "Could not read image"}
    
    # Apply edge detection
    edges = cv2.Canny(img, 100, 200)
    
    # Calculate noise
    noise = cv2.GaussianBlur(img, (5, 5), 0) - img
    noise_std = np.std(noise)
    
    # Check if noise is too uniform
    hist = np.histogram(noise, bins=20)[0]
    hist_std = np.std(hist)
    noise_consistency = hist_std / np.mean(hist) if np.mean(hist) > 0 else 0
    
    return {
        "noise_std": float(noise_std),
        "noise_consistency": float(noise_consistency),  # Lower = more consistent = more likely AI
        "edge_density": float(np.sum(edges > 0) / edges.size),
        "ai_score": float(noise_consistency * 50)  # Higher consistency = higher AI score
    }

def detect_face_anomalies(image_path):
    """Detect facial anomalies common in AI images"""
    img = cv2.imread(image_path)
    if img is None:
        return {"error": "Could not read image"}
    
    # Face detection
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    anomalies = {
        "face_count": len(faces),
        "eyes_mismatch": False,
        "asymmetry": 0,
        "skin_artifacts": 0
    }
    
    if len(faces) > 0:
        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]
            
            # Check for asymmetry (AI often has subtle asymmetry)
            left_half = face_roi[:, :w//2]
            right_half = face_roi[:, w//2:]
            
            if left_half.size > 0 and right_half.size > 0:
                left_features = feature.hog(left_half, visualize=False)
                right_features = feature.hog(cv2.flip(right_half, 1), visualize=False)
                
                if len(left_features) == len(right_features):
                    similarity = np.corrcoef(left_features, right_features)[0, 1]
                    anomalies["asymmetry"] = float(1 - similarity) * 100
    
    return anomalies