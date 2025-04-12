from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import numpy as np
from PIL import Image
import cv2
from scipy.signal import find_peaks
import io

app = FastAPI()

def calculate_threads(gray_image, dpi):
    """Core thread counting logic with DPI-scaled parameters"""
    # Scale kernel sizes based on DPI (originally sized for 300 DPI)
    scale_factor = dpi / 300.0
    
    # CLAHE enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray_image)
    
    # Sobel edge detection
    sobelx = cv2.Sobel(enhanced, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(enhanced, cv2.CV_64F, 0, 1, ksize=3)
    abs_sobelx = np.uint8(np.absolute(sobelx))
    abs_sobely = np.uint8(np.absolute(sobely))
    
    # Thresholding
    _, thresh_x = cv2.threshold(abs_sobelx, 30, 255, cv2.THRESH_BINARY)
    _, thresh_y = cv2.threshold(abs_sobely, 30, 255, cv2.THRESH_BINARY)
    
    # Scale morphological kernels
    kernel_h_width = max(1, int(15 * scale_factor))
    kernel_v_height = max(1, int(15 * scale_factor))
    
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_h_width, 1))
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_v_height))
    
    horizontal = cv2.morphologyEx(thresh_x, cv2.MORPH_OPEN, kernel_h)
    vertical = cv2.morphologyEx(thresh_y, cv2.MORPH_OPEN, kernel_v)
    
    # Projections and peaks
    h_proj = np.sum(horizontal, axis=1)
    v_proj = np.sum(vertical, axis=0)
    
    h_peaks, _ = find_peaks(h_proj, height=np.max(h_proj)*0.4, distance=max(1, int(5*scale_factor)))
    v_peaks, _ = find_peaks(v_proj, height=np.max(v_proj)*0.4, distance=max(1, int(5*scale_factor)))
    
    return len(h_peaks), len(v_peaks)

@app.post("/analyze")
async def analyze_thread_count(file: UploadFile = File(...)):
    try:
        # Read and validate image
        contents = await file.read()
        pil_image = Image.open(io.BytesIO(contents)).convert('RGB')
        img = np.array(pil_image)
        
        # Validate image is square (1x1 inch)
        if img.shape[0] != img.shape[1]:
            raise HTTPException(
                status_code=400,
                detail="Image must be square (1×1 inch)"
            )
            
        dpi = img.shape[0]  # DPI = pixels per inch
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        h_count, v_count = calculate_threads(gray, dpi)
        
        return JSONResponse(
            content={
                "horizontal_thread_count": h_count,
                "vertical_thread_count": v_count,
                "calculated_dpi": dpi
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))