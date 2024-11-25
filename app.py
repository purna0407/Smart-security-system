from flask import Flask, render_template, Response, send_file, jsonify
import cv2
import os
from ultralytics import YOLO  

# Initialize the YOLO model globally
model = YOLO('best (3).pt')

app = Flask(__name__)

# Video capture object
video_capture = cv2.VideoCapture(0)

IMAGE_DIRS = {
    'original': 'original_images',
    'median': 'median_filtered_images',
    'highpass': 'highpass_images',
    'histogram': 'histogram_images',
    'edgedetection': 'edgedetection_images',
    'sobel': 'sobel_filtered_images',
    'unsharpmask': 'unsharp_masked_images',
    'logo': 'logo'
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_recent_image')
def get_recent_image():
    intruder_dir = 'runs/detect/predict/crops/intruder'
    latest_image = get_latest_image(intruder_dir)
    if latest_image:
        return send_file(latest_image, mimetype='image/jpeg')
    else:
        return jsonify({'error': 'No recent image found'}), 404

def serve_image(image_dir, image_name):
    image_path = os.path.join(image_dir, image_name)
    if os.path.exists(image_path):
        return send_file(image_path, mimetype='image/jpeg')
    else:
        return jsonify({'error': f'Image {image_name} not found'}), 404

@app.route('/image/original')
def get_original_image():
    return serve_image(IMAGE_DIRS['original'], "Original_image.jpg")

@app.route('/image/medianfiltered')
def get_median_filtered_image():
    return serve_image(IMAGE_DIRS['median'], "median_filtered_image.jpg")

@app.route('/image/highpass')
def get_highpass_image():
    return serve_image(IMAGE_DIRS['highpass'], "hpf_filtered_image.jpg")

@app.route('/image/histogram')
def get_histogram_image():
    return serve_image(IMAGE_DIRS['histogram'], "histogram_equalized_image.jpg")

@app.route('/image/edgedetection')
def get_edgedetection_image():
    return serve_image(IMAGE_DIRS['edgedetection'], "edge_detected_image.jpg")

@app.route('/image/sobelfiltered')
def get_sobel_filtered_image():
    return serve_image(IMAGE_DIRS['sobel'], "sobel_filtered_image.jpg")

@app.route('/image/unsharpmask')
def get_unsharp_masked_image():
    return serve_image(IMAGE_DIRS['unsharpmask'], "unsharp_masked_image.jpg")

@app.route('/image/logo')
def get_logo():
    image_path = os.path.join(IMAGE_DIRS['logo'], "logo.jpeg")
    if os.path.exists(image_path):
        return send_file(image_path, mimetype='image/png')
    else:
        return jsonify({'error': 'Logo not found'}), 404

def generate_frames():
    while True:
        success, frame = video_capture.read()
        if not success:
            break

        results = model(frame, conf=0.6, save=True, save_crop=True, iou=0.2)
        annotated_frame = results[0].plot()

        for result in results:
            for box in result.boxes:
                class_name = box.cls  # Assuming this gives class name
                confidence = box.conf  # Assuming this gives confidence score

                if (class_name == 'purna' and confidence < 0.9) or class_name == 'intruder':
                    DIP_Techniques.process(frame)
                    x1, y1, x2, y2 = map(int, box.xyxy)
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red for intruders

                elif class_name == 'user':
                    x1, y1, x2, y2 = map(int, box.xyxy)
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green for users

        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def get_latest_image(folder_path):
    if os.path.exists(folder_path):
        files = os.listdir(folder_path)
        image_files = [file for file in files if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
        if image_files:
            image_files.sort(key=lambda x: os.path.getmtime(os.path.join(folder_path, x)), reverse=True)
            return os.path.join(folder_path, image_files[0])
    return None

if __name__ == '__main__':
    app.run()
