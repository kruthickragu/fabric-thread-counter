from flask import Flask, request, jsonify
import cv2
import numpy as np
from PIL import Image
from scipy.signal import find_peaks
from io import BytesIO
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

dpi = 300
inch_size = 1
px_size = int(dpi * inch_size)  # 1 inch in pixels

@app.route('/')
def health_check():
    return jsonify({"status": "ok"})

@app.route('/api/thread-count', methods=['POST'])
def thread_count():
    if 'image' not in request.files:
        return jsonify({'error': 'Image file is missing'}), 400

    file = request.files['image']
    
    try:
        # Log request received
        logger.info(f"Processing image: {file.filename}")
        
        # Load image from bytes
        pil_image = Image.open(file.stream).convert('RGB')
        roi = np.array(pil_image)
        
        # Log image dimensions
        logger.info(f"Image dimensions: {roi.shape}")

        # Step 1: Grayscale conversion
        gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)

        # Step 2: CLAHE for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # Step 3: Sobel edge detection
        sobelx = cv2.Sobel(enhanced, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(enhanced, cv2.CV_64F, 0, 1, ksize=3)
        abs_sobelx = np.uint8(np.absolute(sobelx))
        abs_sobely = np.uint8(np.absolute(sobely))

        # Step 4: Thresholding
        _, thresh_x = cv2.threshold(abs_sobelx, 30, 255, cv2.THRESH_BINARY)
        _, thresh_y = cv2.threshold(abs_sobely, 30, 255, cv2.THRESH_BINARY)

        # Step 5: Morphological filtering
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
        horizontal = cv2.morphologyEx(thresh_x, cv2.MORPH_OPEN, kernel_h)
        vertical = cv2.morphologyEx(thresh_y, cv2.MORPH_OPEN, kernel_v)

        # Step 6: Projection and peak detection
        h_proj = np.sum(horizontal, axis=1)
        v_proj = np.sum(vertical, axis=0)

        h_peaks, _ = find_peaks(h_proj, height=np.max(h_proj) * 0.4, distance=5)
        v_peaks, _ = find_peaks(v_proj, height=np.max(v_proj) * 0.4, distance=5)
        
        # Log results
        logger.info(f"Detected threads - horizontal: {len(h_peaks)}, vertical: {len(v_peaks)}")

        return jsonify({
            'dpi': dpi,
            'box_size': '1 inch × 1 inch',
            'horizontal_threads': int(len(h_peaks)),
            'vertical_threads': int(len(v_peaks))
        })

    except Exception as e:
        logger.error(f"Error processing image: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)  # Set debug=False for production