import piexif
from PIL import Image
import os
import subprocess
import json

def analyze_metadata(image_path):
    """Extract and analyze EXIF metadata for AI fingerprints"""
    results = {
        "has_metadata": False,
        "exif_data": {},
        "ai_indicators": [],
        "score": 0
    }
    
    try:
        img = Image.open(image_path)
        exif_dict = piexif.load(img.info.get('exif', b''))
        
        if exif_dict:
            results["has_metadata"] = True
            
            # Extract useful EXIF fields
            for ifd in exif_dict:
                for tag, value in exif_dict[ifd].items():
                    tag_name = piexif.TAGS.get(ifd, {}).get(tag, {}).get('name', str(tag))
                    if isinstance(value, bytes):
                        try:
                            value = value.decode('utf-8', errors='ignore')
                        except:
                            value = str(value)
                    results["exif_data"][tag_name] = str(value)
            
            # Check for AI generation indicators
            # 1. Software field might reveal AI tools
            if 'Software' in results["exif_data"]:
                software = results["exif_data"]["Software"].lower()
                ai_tools = ['midjourney', 'stable diffusion', 'dall-e', 'firefly', 'stability.ai']
                if any(tool in software for tool in ai_tools):
                    results["ai_indicators"].append(f"Software: {software}")
                    results["score"] += 30
            
            # 2. Check for missing camera data
            if 'Make' not in results["exif_data"] and 'Model' not in results["exif_data"]:
                results["ai_indicators"].append("No camera make/model found")
                results["score"] += 20
            
            # 3. Check for unusual color space
            if 'ColorSpace' in results["exif_data"]:
                if results["exif_data"]["ColorSpace"] == 'Uncalibrated':
                    results["ai_indicators"].append("Uncalibrated color space")
                    results["score"] += 15
                    
    except Exception as e:
        results["error"] = str(e)
    
    # Normalize score
    results["score"] = min(results["score"], 100)
    return results