from flask import Flask, request, jsonify
import numpy as np
import cv2
from PIL import Image
from scipy.signal import find_peaks
from io import BytesIO

app = Flask(__name__)

# Constants
dpi = 300  # Dots per inch
inch_size = 1
px_size = int(dpi * inch_size)  # 300 pixels

@app.route('/analyze-thread-count', methods=['POST'])
def analyze_thread_count():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']

    try:
        # Load image from bytes
        pil_image = Image.open(file).convert('RGB')
        img = np.array(pil_image)

        # Validate image size (should be 300x300 pixels for 1x1 inch)
        if img.shape[0] != px_size or img.shape[1] != px_size:
            return jsonify({'error': f'Image must be {px_size}x{px_size} pixels for 1 inch at 300 DPI'}), 400

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # CLAHE for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # Sobel edges
        sobelx = cv2.Sobel(enhanced, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(enhanced, cv2.CV_64F, 0, 1, ksize=3)
        abs_sobelx = np.uint8(np.absolute(sobelx))
        abs_sobely = np.uint8(np.absolute(sobely))

        # Threshold
        _, thresh_x = cv2.threshold(abs_sobelx, 30, 255, cv2.THRESH_BINARY)
        _, thresh_y = cv2.threshold(abs_sobely, 30, 255, cv2.THRESH_BINARY)

        # Morphological filtering
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
        horizontal = cv2.morphologyEx(thresh_x, cv2.MORPH_OPEN, kernel_h)
        vertical = cv2.morphologyEx(thresh_y, cv2.MORPH_OPEN, kernel_v)

        # Projections
        h_proj = np.sum(horizontal, axis=1)
        v_proj = np.sum(vertical, axis=0)

        # Peak detection
        h_peaks, _ = find_peaks(h_proj, height=np.max(h_proj) * 0.4, distance=5)
        v_peaks, _ = find_peaks(v_proj, height=np.max(v_proj) * 0.4, distance=5)

        # Count threads
        num_h_threads = len(h_peaks)
        num_v_threads = len(v_peaks)

        return jsonify({
            'dpi': dpi,
            'horizontal_thread_count': num_h_threads,
            'vertical_thread_count': num_v_threads
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500



import os

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Railway sets PORT as an env variable
    app.run(host="0.0.0.0", port=port, debug=True)