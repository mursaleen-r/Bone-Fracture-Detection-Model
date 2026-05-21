import gradio as gr
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps, ImageDraw
import scipy.ndimage as ndimage
import os

# ==========================================
# 1. CONFIGURATION
# ==========================================
MODEL_PATH = 'fracture_model.h5'
IMG_SIZE = (224, 224)

# Load Model
try:
    model = load_model(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# ==========================================
# 2. REFINED DIAGNOSTIC LOGIC
# ==========================================
def get_medical_label(confidence_score):
    if confidence_score < 0.70: return "Grade I (Minor)"
    elif confidence_score < 0.90: return "Grade II (Moderate)"
    else: return "Grade III (Severe)"

def compute_saliency_map(model, img_array):
    """
    Computes the raw attention map of the AI.
    """
    img_tensor = tf.convert_to_tensor(img_array)
    with tf.GradientTape() as tape:
        tape.watch(img_tensor)
        predictions = model(img_tensor)
        loss = predictions[0][0]
    
    gradients = tape.gradient(loss, img_tensor)
    gradients = tf.abs(gradients)
    saliency = tf.reduce_max(gradients, axis=-1)[0]
    return saliency.numpy()

def find_lesion_boxes(saliency_map, original_img_array, max_lesions=3):
    """
    Refined Clustering Logic with Morphological Closing.
    """
    # 1. Gaussian Smoothing (Reduced sigma for sharper edges)
    saliency_map = ndimage.gaussian_filter(saliency_map, sigma=1.0)
    
    # 2. Normalize
    denom = np.max(saliency_map) - np.min(saliency_map)
    if denom == 0: return []
    saliency_map = (saliency_map - np.min(saliency_map)) / denom
    
    # 3. Adaptive Thresholding
    # We use a slightly lower threshold to catch faint fracture lines
    threshold = np.percentile(saliency_map, 96)
    binary_map = saliency_map > threshold
    
    # 4. Morphological Closing (NEW FEATURE)
    # This connects nearby hot pixels to form a solid line.
    # It fixes the issue where the box would sometimes be split in two.
    binary_map = ndimage.binary_closing(binary_map, structure=np.ones((3,3)))
    
    # 5. Anatomy Check
    # Ensure we are looking at bone (bright), not background (dark)
    mean_intensity = np.mean(original_img_array)
    bone_threshold = mean_intensity * 0.55 
    img_intensity = np.mean(original_img_array[0], axis=-1)
    clean_map = np.logical_and(binary_map, img_intensity > bone_threshold)
    
    # 6. Cluster Analysis
    labeled_array, num_features = ndimage.label(clean_map)
    
    if num_features == 0: return []
        
    valid_clusters = []
    for i in range(1, num_features + 1):
        size = np.sum(labeled_array == i)
        # Filter out tiny noise specks
        if size > 20: 
            valid_clusters.append((i, size))
            
    valid_clusters.sort(key=lambda x: x[1], reverse=True)
    top_clusters = valid_clusters[:max_lesions]
    
    boxes = []
    slices = ndimage.find_objects(labeled_array)
    for cluster_id, size in top_clusters:
        y_slice, x_slice = slices[cluster_id - 1]
        boxes.append((x_slice.start, y_slice.start, x_slice.stop, y_slice.stop))
        
    return boxes

# ==========================================
# 3. MAIN APP FUNCTION
# ==========================================
def analyze_xray(image):
    if image is None:
        return "Please upload an image.", None, None
    if model is None:
        return "⚠️ Error: Model not loaded. Please check 'fracture_model.h5' path.", None, None

    # Convert to PIL Image if needed
    image = Image.fromarray(image.astype('uint8'), 'RGB')
    
    # Auto-contrast for better visibility
    image_display = ImageOps.autocontrast(image)
    
    # Preprocess for AI
    img_resized = image_display.resize(IMG_SIZE)
    img_array = np.array(img_resized)
    img_batch = np.expand_dims(img_array, axis=0) / 255.0
    
    # Prediction
    prediction = model.predict(img_batch, verbose=0)[0][0]
    
    # Prepare Outputs
    annotated_img = img_resized.copy()
    heatmap_img = Image.new('RGB', IMG_SIZE)
    
    if prediction >= 0.5:
        # HEALTHY
        diagnosis = f"✅ DIAGNOSIS: HEALTHY BONE\nConfidence: {prediction*100:.1f}%"
        return diagnosis, image_display, None
    else:
        # FRACTURE
        confidence = 1.0 - prediction
        grade = get_medical_label(confidence)
        diagnosis = f"⚠️ DIAGNOSIS: FRACTURE DETECTED\nSeverity: {grade}\nAI Confidence: {confidence*100:.1f}%"
        
        # Saliency Analysis
        saliency_map = compute_saliency_map(model, img_batch)
        boxes = find_lesion_boxes(saliency_map, img_batch)
        
        # Draw Boxes
        draw = ImageDraw.Draw(annotated_img)
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            pad = 6
            x1, y1 = max(0, x1-pad), max(0, y1-pad)
            x2, y2 = min(IMG_SIZE[0], x2+pad), min(IMG_SIZE[1], y2+pad)
            
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            if i == 0:
                draw.rectangle([x1, y1-15, x1+100, y1], fill="black")
                draw.text((x1+5, y1-12), grade, fill="white")
        
        # Generate Heatmap Visualization
        # Normalize
        norm_saliency = (saliency_map - np.min(saliency_map)) / (np.max(saliency_map) - np.min(saliency_map) + 1e-8)
        heatmap_pil = Image.fromarray(np.uint8(255 * norm_saliency))
        heatmap_colored = ImageOps.colorize(heatmap_pil, black="black", white="red")
        
        # Blend with original
        heatmap_overlay = Image.blend(img_resized, heatmap_colored, alpha=0.5)
        
        return diagnosis, annotated_img, heatmap_overlay

# ==========================================
# 4. GRADIO INTERFACE
# ==========================================
with gr.Blocks(theme=gr.themes.Soft()) as app:
    gr.Markdown("""
    # 🏥 Medical Diagnostic Assistant: Bone Fracture Detection
    Upload an X-ray image for instant AI analysis.
    """)
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="Upload X-Ray", type="numpy")
            submit_btn = gr.Button("Analyze Image", variant="primary")
        
        with gr.Column():
            diagnosis_output = gr.Textbox(label="Diagnostic Report")
            
    with gr.Row():
        output_image = gr.Image(label="Localized Fracture Site")
        heatmap_output = gr.Image(label="AI Saliency Heatmap (Explainable AI)")

    submit_btn.click(
        fn=analyze_xray,
        inputs=input_image,
        outputs=[diagnosis_output, output_image, heatmap_output]
    )

if __name__ == "__main__":
    app.launch()