# thread_count_api.py
from flask import Flask, request, jsonify
import numpy as np
import cv2
from PIL import Image
from io import BytesIO
from scipy.signal import find_peaks

app = Flask(__name__)

dpi = 300
inch_size = 1
px_size = int(dpi * inch_size)

def process_image(image):
    img = np.array(Image.open(BytesIO(image)).convert('RGB'))
    clone = img.copy()

    # Set static position or dynamically adjust
    start_x, start_y = 50, 50
    roi = clone[start_y:start_y + px_size, start_x:start_x + px_size]

    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    sobelx = cv2.Sobel(enhanced, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(enhanced, cv2.CV_64F, 0, 1, ksize=3)
    abs_sobelx = np.uint8(np.absolute(sobelx))
    abs_sobely = np.uint8(np.absolute(sobely))

    _, thresh_x = cv2.threshold(abs_sobelx, 30, 255, cv2.THRESH_BINARY)
    _, thresh_y = cv2.threshold(abs_sobely, 30, 255, cv2.THRESH_BINARY)

    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
    horizontal = cv2.morphologyEx(thresh_x, cv2.MORPH_OPEN, kernel_h)
    vertical = cv2.morphologyEx(thresh_y, cv2.MORPH_OPEN, kernel_v)

    h_proj = np.sum(horizontal, axis=1)
    v_proj = np.sum(vertical, axis=0)

    h_peaks, _ = find_peaks(h_proj, height=np.max(h_proj) * 0.4, distance=5)
    v_peaks, _ = find_peaks(v_proj, height=np.max(v_proj) * 0.4, distance=5)

    return len(h_peaks), len(v_peaks)

@app.route('/analyze-threads', methods=['POST'])
def analyze_threads():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    image_data = file.read()
    num_h, num_v = process_image(image_data)

    return jsonify({
        'horizontal_threads': num_h,
        'vertical_threads': num_v,
        'dpi': dpi
    })

if __name__ == '__main__':
    app.run(debug=True)
