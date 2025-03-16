from flask import Flask, render_template, request, send_from_directory, jsonify
import os
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps, ImageDraw, ImageFont
import base64
import re
import io
import uuid
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configure folders
UPLOAD_FOLDER = 'static/uploads'
PROCESSED_FOLDER = 'static/processed'
FILTER_PREVIEW_FOLDER = 'static/img'

# Create directories if they don't exist
for folder in [UPLOAD_FOLDER, PROCESSED_FOLDER, FILTER_PREVIEW_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# Helper function to convert base64 to image
def base64_to_image(base64_str):
    # Extract the base64 encoded binary data from the input string
    image_data = re.sub('^data:image/.+;base64,', '', base64_str)
    image_bytes = base64.b64decode(image_data)
    image = Image.open(io.BytesIO(image_bytes))
    return image

# Helper function to convert image to base64
def image_to_base64(image, format='PNG'):
    buffered = io.BytesIO()
    image.save(buffered, format=format)
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return f"data:image/{format.lower()};base64,{img_str}"

# Helper function to convert PIL Image to OpenCV format
def pil_to_cv2(pil_image):
    # Convert PIL Image to numpy array
    cv_image = np.array(pil_image)
    # Convert RGB to BGR (OpenCV format)
    if len(cv_image.shape) == 3 and cv_image.shape[2] == 3:
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
    return cv_image

# Helper function to convert OpenCV image to PIL format
def cv2_to_pil(cv_image):
    # Convert BGR to RGB (PIL format)
    if len(cv_image.shape) == 3 and cv_image.shape[2] == 3:
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    # Convert numpy array to PIL Image
    pil_image = Image.fromarray(cv_image)
    return pil_image

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_image():
    try:
        data = request.json
        image_data = data.get('image')
        adjustments = data.get('adjustments', {})
        
        if not image_data:
            return jsonify({'error': 'No image data provided'})
        
        # Convert base64 to PIL Image
        image = base64_to_image(image_data)
        
        # Process image based on adjustments
        processed_image = apply_adjustments(image, adjustments)
        
        # Convert back to base64
        processed_base64 = image_to_base64(processed_image)
        
        return jsonify({'processed_image': processed_base64})
    
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}", exc_info=True)
        return jsonify({'error': f'Failed to process image: {str(e)}'})

def apply_adjustments(image, adjustments):
    # Make a copy of the image to avoid modifying the original
    img = image.copy()
    
    # Handle crop if specified
    if adjustments.get('crop'):
        crop_data = adjustments.get('cropData', {})
        x = crop_data.get('x', 0)
        y = crop_data.get('y', 0)
        width = crop_data.get('width', img.width)
        height = crop_data.get('height', img.height)
        img = img.crop((x, y, x + width, y + height))
    
    # Handle resize if specified
    if adjustments.get('resize'):
        width = adjustments.get('width', img.width)
        height = adjustments.get('height', img.height)
        img = img.resize((width, height), Image.LANCZOS)
    
    # Apply rotation
    rotation = adjustments.get('rotation', 0)
    if rotation != 0:
        img = img.rotate(rotation, expand=True, resample=Image.BICUBIC)
    
    # Apply flips
    if adjustments.get('flipHorizontal'):
        img = ImageOps.mirror(img)
    
    if adjustments.get('flipVertical'):
        img = ImageOps.flip(img)
    
    # Apply basic adjustments using PIL
    brightness = adjustments.get('brightness', 0)
    if brightness != 0:
        factor = 1.0 + (brightness / 100.0)
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(factor)
    
    contrast = adjustments.get('contrast', 0)
    if contrast != 0:
        factor = 1.0 + (contrast / 100.0)
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(factor)
    
    saturation = adjustments.get('saturation', 0)
    if saturation != 0:
        factor = 1.0 + (saturation / 100.0)
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(factor)
    
    sharpness = adjustments.get('sharpness', 0)
    if sharpness > 0:
        factor = 1.0 + (sharpness / 50.0)
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(factor)
    
    blur_amount = adjustments.get('blur', 0)
    if blur_amount > 0:
        img = img.filter(ImageFilter.GaussianBlur(radius=blur_amount / 10))
    
    # Apply noise reduction (requires OpenCV)
    noise_reduction = adjustments.get('noise', 0)
    if noise_reduction > 0:
        cv_img = pil_to_cv2(img)
        strength = int(noise_reduction / 10) * 2 + 1  # Must be odd number
        cv_img = cv2.fastNlMeansDenoisingColored(cv_img, None, strength, strength, 7, 21)
        img = cv2_to_pil(cv_img)
    
    # Apply filters using OpenCV
    filter_name = adjustments.get('filter')
    if filter_name and filter_name != 'none':
        img = apply_filter(img, filter_name)
    
    # Apply effects
    effect_name = adjustments.get('effect')
    if effect_name:
        img = apply_effect(img, effect_name)
    
    # Add text if specified
    if adjustments.get('addText'):
        text = adjustments.get('text', '')
        font_name = adjustments.get('font', 'Arial')
        font_size = int(adjustments.get('fontSize', 24))
        color = adjustments.get('color', '#ffffff')
        position = adjustments.get('position', 'center')
        
        img = add_text_to_image(img, text, font_name, font_size, color, position)
    
    return img

def apply_filter(image, filter_name):
    # Convert PIL to OpenCV for some filters
    cv_img = pil_to_cv2(image)
    
    if filter_name == 'grayscale':
        # Use PIL for grayscale
        return ImageOps.grayscale(image).convert('RGB')
    
    elif filter_name == 'sepia':
        # Sepia filter
        sepia_kernel = np.array([
            [0.393, 0.769, 0.189],
            [0.349, 0.686, 0.168],
            [0.272, 0.534, 0.131]
        ])
        sepia_img = cv2.transform(cv_img, sepia_kernel)
        sepia_img = np.clip(sepia_img, 0, 255).astype(np.uint8)
        return cv2_to_pil(sepia_img)
    
    elif filter_name == 'invert':
        return ImageOps.invert(image)
    
    elif filter_name == 'blur':
        return image.filter(ImageFilter.GaussianBlur(radius=10))
    
    elif filter_name == 'edge_detection':
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        return cv2_to_pil(edges_rgb)
    
    elif filter_name == 'sharpen':
        return image.filter(ImageFilter.SHARPEN)
    
    elif filter_name == 'emboss':
        return image.filter(ImageFilter.EMBOSS)
    
    elif filter_name == 'vintage':
        # Vintage effect (sepia + vignette)
        sepia_kernel = np.array([
            [0.393, 0.769, 0.189],
            [0.349, 0.686, 0.168],
            [0.272, 0.534, 0.131]
        ])
        sepia_img = cv2.transform(cv_img, sepia_kernel)
        sepia_img = np.clip(sepia_img, 0, 255).astype(np.uint8)
        
        # Add vignette
        rows, cols = sepia_img.shape[:2]
        
        # Generate vignette mask
        x = int(cols/2)
        y = int(rows/2)
        radius = min(x, y)
        
        mask = np.zeros((rows, cols), dtype=np.uint8)
        cv2.circle(mask, (x, y), radius, 255, -1)
        mask = cv2.GaussianBlur(mask, (51, 51), 0)
        
        # Normalize mask
        mask = mask / 255.0
        
        # Apply mask to each channel
        for c in range(3):
            sepia_img[:,:,c] = sepia_img[:,:,c] * mask
        
        return cv2_to_pil(sepia_img)
    
    elif filter_name == 'cool':
        # Cool tone filter (blue tint)
        b, g, r = cv2.split(cv_img)
        b = np.clip(b * 1.1, 0, 255).astype(np.uint8)
        cool_img = cv2.merge([b, g, r])
        return cv2_to_pil(cool_img)
    
    elif filter_name == 'warm':
        # Warm tone filter (orange tint)
        b, g, r = cv2.split(cv_img)
        r = np.clip(r * 1.1, 0, 255).astype(np.uint8)
        warm_img = cv2.merge([b, g, r])
        return cv2_to_pil(warm_img)
    
    elif filter_name == 'pencil':
        # Pencil sketch effect
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        inv_gray = 255 - gray
        blur = cv2.GaussianBlur(inv_gray, (21, 21), 0)
        sketch = cv2.divide(gray, 255 - blur, scale=256)
        sketch_rgb = cv2.cvtColor(sketch, cv2.COLOR_GRAY2RGB)
        return cv2_to_pil(sketch_rgb)
    
    # Default: return original image
    return image

def apply_effect(image, effect_name):
    # Convert PIL to OpenCV for some effects
    cv_img = pil_to_cv2(image)
    
    if effect_name == 'vignette':
        rows, cols = cv_img.shape[:2]
        
        # Generate vignette mask
        x = int(cols/2)
        y = int(rows/2)
        radius = min(x, y)
        
        mask = np.zeros((rows, cols), dtype=np.uint8)
        cv2.circle(mask, (x, y), radius, 255, -1)
        mask = cv2.GaussianBlur(mask, (51, 51), 0)
        
        # Normalize mask
        mask = mask / 255.0
        
        # Apply mask to each channel
        for c in range(3):
            cv_img[:,:,c] = cv_img[:,:,c] * mask
        
        return cv2_to_pil(cv_img)
    
    elif effect_name == 'pixelate':
        # Pixelate effect
        h, w = cv_img.shape[:2]
        
        # Downscale
        temp = cv2.resize(cv_img, (w//20, h//20), interpolation=cv2.INTER_LINEAR)
        
        # Upscale
        pixelated = cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)
        
        return cv2_to_pil(pixelated)
    
    elif effect_name == 'oil_painting':
        # Oil painting effect
        oil = cv2.xphoto.oilPainting(cv_img, 7, 1)
        return cv2_to_pil(oil)
    
    elif effect_name == 'cartoon':
        # Cartoon effect
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)
        edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
        
        # Color quantization
        data = np.float32(cv_img).reshape((-1, 3))
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.001)
        ret, label, center = cv2.kmeans(data, 8, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        center = np.uint8(center)
        result = center[label.flatten()]
        result = result.reshape(cv_img.shape)
        
        # Combine
        cartoon = cv2.bitwise_and(result, result, mask=edges)
        return cv2_to_pil(cartoon)
    
    elif effect_name == 'watercolor':
        # Watercolor effect
        # Apply bilateral filter multiple times
        for _ in range(2):
            cv_img = cv2.bilateralFilter(cv_img, 9, 75, 75)
        
        # Increase saturation
        hsv = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        s = np.clip(s * 1.5, 0, 255).astype(np.uint8)
        hsv = cv2.merge([h, s, v])
        watercolor = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        return cv2_to_pil(watercolor)
    
    elif effect_name == 'posterize':
        # Posterize effect (color quantization)
        data = np.float32(cv_img).reshape((-1, 3))
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.001)
        ret, label, center = cv2.kmeans(data, 5, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        center = np.uint8(center)
        result = center[label.flatten()]
        posterized = result.reshape(cv_img.shape)
        return cv2_to_pil(posterized)
    
    elif effect_name == 'halftone':
        # Halftone effect
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # Create a blank white image
        halftone = np.ones((h, w, 3), dtype=np.uint8) * 255
        
        # Define dot size and spacing
        dot_size = 5
        spacing = 10
        
        # Draw dots
        for y in range(0, h, spacing):
            for x in range(0, w, spacing):
                if y + spacing < h and x + spacing < w:
                    # Get average intensity in this cell
                    intensity = np.mean(gray[y:y+spacing, x:x+spacing])
                    
                    # Calculate dot radius based on intensity (darker = larger dot)
                    radius = int((255 - intensity) * dot_size / 255)
                    
                    if radius > 0:
                        cv2.circle(halftone, (x + spacing//2, y + spacing//2), radius, (0, 0, 0), -1)
        
        return cv2_to_pil(halftone)
    
    elif effect_name == 'glitch':
        # Glitch effect
        h, w = cv_img.shape[:2]
        
        # Split into channels
        b, g, r = cv2.split(cv_img)
        
        # Shift channels randomly
        for _ in range(10):
            # Random position to apply glitch
            y = np.random.randint(0, h-20)
            h_shift = np.random.randint(5, 15)
            w_shift = np.random.randint(5, 15)
            
            # Apply shifts to different channels
            if np.random.random() > 0.5:
                r[y:y+h_shift, w_shift:] = r[y:y+h_shift, :-w_shift]
            else:
                b[y:y+h_shift, :-w_shift] = b[y:y+h_shift, w_shift:]
        
        # Merge channels back
        glitched = cv2.merge([b, g, r])
        
        return cv2_to_pil(glitched)
    
    # Default: return original image
    return image

def add_text_to_image(image, text, font_name, font_size, color, position):
    # Create a drawing context
    draw = ImageDraw.Draw(image)
    
    # Try to use the specified font, fall back to default if not available
    try:
        font = ImageFont.truetype(font_name, font_size)
    except IOError:
        # Use default font
        font = ImageFont.load_default()
    
    # Convert hex color to RGB
    if color.startswith('#'):
        r = int(color[1:3], 16)
        g = int(color[3:5], 16)
        b = int(color[5:7], 16)
        rgb_color = (r, g, b)
    else:
        rgb_color = (255, 255, 255)  # Default to white
    
    # Calculate text size
    text_width, text_height = draw.textsize(text, font=font)
    
    # Calculate position
    width, height = image.size
    
    if position == 'top-left':
        x, y = 10, 10
    elif position == 'top':
        x, y = (width - text_width) // 2, 10
    elif position == 'top-right':
        x, y = width - text_width - 10, 10
    elif position == 'left':
        x, y = 10, (height - text_height) // 2
    elif position == 'center':
        x, y = (width - text_width) // 2, (height - text_height) // 2
    elif position == 'right':
        x, y = width - text_width - 10, (height - text_height) // 2
    elif position == 'bottom-left':
        x, y = 10, height - text_height - 10
    elif position == 'bottom':
        x, y = (width - text_width) // 2, height - text_height - 10
    elif position == 'bottom-right':
        x, y = width - text_width - 10, height - text_height - 10
    else:
        x, y = (width - text_width) // 2, (height - text_height) // 2
    
    # Add text shadow for better visibility
    draw.text((x+2, y+2), text, font=font, fill=(0, 0, 0))
    
    # Draw the text
    draw.text((x, y), text, font=font, fill=rgb_color)
    
    return image

@app.route('/static/img/<path:filename>')
def serve_filter_preview(filename):
    return send_from_directory(FILTER_PREVIEW_FOLDER, filename)

@app.route('/static/uploads/<path:filename>')
def serve_upload(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/static/processed/<path:filename>')
def serve_processed(filename):
    return send_from_directory(PROCESSED_FOLDER, filename)

# Generate filter preview images if they don't exist
def generate_filter_previews():
    # Sample image for previews
    sample_img_path = os.path.join(FILTER_PREVIEW_FOLDER, 'sample.jpg')
    
    # Check if sample image exists, if not create one
    if not os.path.exists(sample_img_path):
        # Create a gradient image
        width, height = 300, 200
        image = Image.new('RGB', (width, height), color='white')
        draw = ImageDraw.Draw(image)
        
        # Draw a simple scene
        # Sky
        for y in range(height//2):
            color = (135 - y//2, 206 - y//2, 235)
            draw.line([(0, y), (width, y)], fill=color)
        
        # Ground
        for y in range(height//2, height):
            color = (34, 139 - (y-height//2)//2, 34)
            draw.line([(0, y), (width, y)], fill=color)
        
        # Sun
        draw.ellipse([(width-80, 20), (width-30, 70)], fill=(255, 255, 0))
        
        # Mountains
        draw.polygon([(0, height//2), (width//3, height//4), (width//2, height//2)], fill=(120, 120, 120))
        draw.polygon([(width//3, height//2), (2*width//3, height//3), (width, height//2)], fill=(100, 100, 100))
        
        # Save the sample image
        image.save(sample_img_path)
    
    # Load the sample image
    sample_img = Image.open(sample_img_path)
    
    # Generate previews for each filter
    filters = {
        'none': lambda img: img,
        'grayscale': lambda img: ImageOps.grayscale(img).convert('RGB'),
        'sepia': lambda img: apply_filter(img, 'sepia'),
        'invert': lambda img: ImageOps.invert(img),
        'blur': lambda img: img.filter(ImageFilter.GaussianBlur(radius=10)),
        'edge_detection': lambda img: apply_filter(img, 'edge_detection'),
        'sharpen': lambda img: img.filter(ImageFilter.SHARPEN),
        'emboss': lambda img: img.filter(ImageFilter.EMBOSS),
        'vintage': lambda img: apply_filter(img, 'vintage'),
        'cool': lambda img: apply_filter(img, 'cool'),
        'warm': lambda img: apply_filter(img, 'warm'),
        'pencil': lambda img: apply_filter(img, 'pencil')
    }
    
    # Generate and save filter previews
    for filter_name, filter_func in filters.items():
        preview_path = os.path.join(FILTER_PREVIEW_FOLDER, f'filter-{filter_name}.jpg')
        if not os.path.exists(preview_path):
            filtered_img = filter_func(sample_img.copy())
            filtered_img.save(preview_path)
    
    # Generate effect previews
    effects = {
        'vignette': lambda img: apply_effect(img, 'vignette'),
        'pixelate': lambda img: apply_effect(img, 'pixelate'),
        'oil_painting': lambda img: apply_effect(img, 'oil_painting'),
        'cartoon': lambda img: apply_effect(img, 'cartoon'),
        'watercolor': lambda img: apply_effect(img, 'watercolor'),
        'posterize': lambda img: apply_effect(img, 'posterize'),
        'halftone': lambda img: apply_effect(img, 'halftone'),
        'glitch': lambda img: apply_effect(img, 'glitch')
    }
    
    # Generate and save effect previews
    for effect_name, effect_func in effects.items():
        preview_path = os.path.join(FILTER_PREVIEW_FOLDER, f'effect-{effect_name}.jpg')
        if not os.path.exists(preview_path):
            effected_img = effect_func(sample_img.copy())
            effected_img.save(preview_path)

# Generate filter previews on startup
generate_filter_previews()

if __name__ == '__main__':
    app.run(debug=True)

