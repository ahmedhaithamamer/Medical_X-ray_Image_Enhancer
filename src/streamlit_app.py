"""
Streamlit interactive viewer for X-ray image enhancement results.

Run with:
    streamlit run src/streamlit_app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
from pathlib import Path
import cv2
from PIL import Image
import zipfile
import io

from io_utils import list_images, load_grayscale
from enhancement import enhancement_pipeline
from metrics import mse, psnr

# Page configuration
st.set_page_config(
    page_title="X-ray Image Enhancement Viewer",
    page_icon="🏥",
    layout="wide"
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"


def load_image_array(img_path: str) -> np.ndarray:
    """Load image and convert to RGB for display."""
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    return img


def numpy_to_pil(img: np.ndarray) -> Image.Image:
    """Convert numpy array to PIL Image."""
    return Image.fromarray(img)


@st.cache_data
def get_available_images():
    """Get list of available images."""
    image_paths = list_images(str(DATA_DIR))
    return [Path(p).stem for p in image_paths]


def main():
    st.title("🏥 Medical X-ray Image Enhancement Viewer")
    st.markdown("Interactive comparison tool for image enhancement techniques")
    
    # Sidebar for controls
    with st.sidebar:
        st.header("⚙️ Controls")
        
        # Image source selection
        image_source = st.radio(
            "Image Source",
            ["Upload Image", "Select from Dataset"],
            index=0
        )
        
        uploaded_file = None
        selected_image = None
        
        if image_source == "Upload Image":
            uploaded_file = st.file_uploader(
                "Upload X-ray Image",
                type=['png', 'jpg', 'jpeg'],
                help="Upload a chest X-ray image to enhance"
            )
        else:
            # Image selection from dataset
            available_images = get_available_images()
            if not available_images:
                st.warning("No images found in data/ folder.")
                st.info("You can upload an image using the 'Upload Image' option above.")
            else:
                selected_image = st.selectbox(
                    "Select Image",
                    available_images,
                    index=0
                )
        
        st.divider()
        
        # Enhancement technique selection
        st.subheader("Enhancement Techniques")
        show_original = st.checkbox("Original", value=True)
        show_best = st.checkbox("⭐ Best Pipeline", value=True, help="Optimal sequential enhancement: Denoise → Contrast → Sharpen")
        show_median = st.checkbox("Median Filter", value=False)
        show_bilateral = st.checkbox("Bilateral Filter", value=False)
        show_contrast = st.checkbox("Histogram Equalization", value=False)
        show_clahe = st.checkbox("CLAHE", value=False)
        show_sharpen = st.checkbox("Sharpening", value=False)
        show_unsharp = st.checkbox("Unsharp Masking", value=False)
        
        st.divider()
        
        # Processing options (only show for dataset images)
        if image_source == "Select from Dataset":
            st.subheader("Processing Options")
            process_live = st.checkbox("Process Live (recompute)", value=False)
        else:
            process_live = True  # Always process live for uploaded images
        
        st.divider()
        show_metrics = st.checkbox("Show Metrics", value=True)
    
    # Load original image
    if image_source == "Upload Image":
        if uploaded_file is None:
            st.info("👆 Please upload an X-ray image to get started!")
            st.stop()
        
        # Process uploaded file
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        original_img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
        
        if original_img is None:
            st.error("Failed to load uploaded image. Please try a different file.")
            st.stop()
        
        st.success(f"✅ Loaded image: {uploaded_file.name} ({original_img.shape[1]}×{original_img.shape[0]})")
        
    else:
        # Load from dataset
        if selected_image is None:
            st.stop()
        
        original_path = DATA_DIR / f"{selected_image}.png"
        if not original_path.exists():
            # Try other extensions
            for ext in [".jpg", ".jpeg", ".JPG", ".JPEG"]:
                alt_path = DATA_DIR / f"{selected_image}{ext}"
                if alt_path.exists():
                    original_path = alt_path
                    break
        
        if not original_path.exists():
            st.error(f"Image not found: {selected_image}")
            st.stop()
        
        original_img = load_grayscale(str(original_path))
    
    # Process image (always process live for uploaded images, or based on setting for dataset images)
    if image_source == "Upload Image" or process_live:
        with st.spinner("Processing image with enhancement techniques..."):
            outputs = enhancement_pipeline(
                original_img,
                use_clahe=True,
                use_bilateral=True,
                use_unsharp=True,
                include_best_pipeline=True  # Always include best pipeline
            )
    else:
        # Load from results folder (only for dataset images)
        if selected_image:
            result_dir = RESULTS_DIR / selected_image
            outputs = {"original": original_img}
            
            enhancement_files = {
                "median": "median.png",
                "bilateral": "bilateral.png",
                "contrast": "contrast.png",
                "clahe": "clahe.png",
                "sharpen": "sharpen.png",
                "unsharp": "unsharp.png",
                "best_pipeline": "best_pipeline.png",  # Automatically load best pipeline if available
            }
            
            loaded_from_disk = []
            for key, filename in enhancement_files.items():
                filepath = result_dir / filename
                if filepath.exists():
                    outputs[key] = load_grayscale(str(filepath))
                    loaded_from_disk.append(key)
            
            # Show status if best_pipeline was loaded from disk
            if "best_pipeline" in loaded_from_disk:
                st.success(f"✅ Loaded Best Pipeline from saved results")
            
            # If some files are missing, process live for those
            # Expected: original + 6 individual techniques + best_pipeline = 8 total
            # But some may be optional (bilateral, unsharp), so check for at least core ones
            core_keys = ["original", "median", "contrast", "clahe", "sharpen"]
            missing_core = [key for key in core_keys if key not in outputs]
            
            if missing_core or "best_pipeline" not in outputs:
                with st.spinner("Processing missing enhancements..."):
                    temp_outputs = enhancement_pipeline(
                        original_img,
                        use_clahe=True,
                        use_bilateral=True,
                        use_unsharp=True,
                        include_best_pipeline=True
                    )
                    # Add missing outputs, prioritizing best_pipeline
                    for key in temp_outputs:
                        if key not in outputs:
                            outputs[key] = temp_outputs[key]
                    # Ensure best_pipeline is always included
                    if "best_pipeline" not in outputs and "best_pipeline" in temp_outputs:
                        outputs["best_pipeline"] = temp_outputs["best_pipeline"]
        else:
            st.error("No image selected")
            st.stop()
    
    # Before/After Slider Comparison
    st.header("📊 Before/After Comparison")
    
    # Technique selection for comparison
    technique_map = {
        "Original": "original",
        "⭐ Best Pipeline": "best_pipeline",
        "Median Filter": "median",
        "Bilateral Filter": "bilateral",
        "Histogram Equalization": "contrast",
        "CLAHE": "clahe",
        "Sharpening": "sharpen",
        "Unsharp Masking": "unsharp",
    }
    
    # Filter available techniques (excluding Original for "After")
    # Prioritize Best Pipeline first
    available_after_techniques = []
    best_pipeline_name = None
    for name, key in technique_map.items():
        if key in outputs and key != "original":
            if key == "best_pipeline":
                best_pipeline_name = name
            else:
                available_after_techniques.append(name)
    
    # Add Best Pipeline at the beginning if available
    if best_pipeline_name:
        available_after_techniques.insert(0, best_pipeline_name)
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.write("**Before:** Original")
        before_tech = "Original"
    with col_b:
        if available_after_techniques:
            after_tech = st.selectbox("After", available_after_techniques, index=0)
        else:
            st.error("No enhancement techniques available")
            after_tech = None
    
    # Slider for before/after
    slider_value = st.slider(
        "Before ↔ After",
        0.0, 1.0, 0.5,
        help="Drag to compare before and after images"
    )
    
    # Display comparison
    if after_tech:
        before_key = technique_map[before_tech]
        after_key = technique_map[after_tech]
        
        before_img = outputs[before_key]
        after_img = outputs[after_key]
        
        # Create blended image based on slider
        # slider_value: 0.0 = all before, 0.5 = 50/50, 1.0 = all after
        before_weight = 1.0 - slider_value  # 1.0 at 0, 0.5 at 0.5, 0.0 at 1.0
        after_weight = slider_value  # 0.0 at 0, 0.5 at 0.5, 1.0 at 1.0
        
        blended = cv2.addWeighted(before_img, before_weight, after_img, after_weight, 0)
        label = f"{before_tech} ({int(before_weight*100)}%) ↔ {after_tech} ({int(after_weight*100)}%)"
        
        st.image(blended, caption=label, width='stretch')
    
    # Side-by-side comparison
    if after_tech:
        st.header("🔄 Side-by-Side Comparison")
        comp_col1, comp_col2 = st.columns(2)
        
        with comp_col1:
            st.subheader(before_tech)
            st.image(before_img, width='stretch')
        
        with comp_col2:
            st.subheader(after_tech)
            st.image(after_img, width='stretch')
            if show_metrics:
                m = mse(outputs["original"], after_img)
                p = psnr(outputs["original"], after_img)
                st.metric("MSE", f"{m:.2f}")
                st.metric("PSNR", f"{p:.2f} dB")
    
    # All techniques grid view
    st.header("🖼️ All Enhancement Techniques")
    
    # Filter techniques based on checkboxes
    display_map = {
        "original": (show_original, "Original"),
        "best_pipeline": (show_best, "⭐ Best Pipeline"),
        "median": (show_median, "Median Filter"),
        "bilateral": (show_bilateral, "Bilateral Filter"),
        "contrast": (show_contrast, "Histogram Equalization"),
        "clahe": (show_clahe, "CLAHE"),
        "sharpen": (show_sharpen, "Sharpening"),
        "unsharp": (show_unsharp, "Unsharp Masking"),
    }
    
    # Create grid - prioritize best_pipeline and original
    display_items = []
    best_item = None
    original_item = None
    
    for key, (show, name) in display_map.items():
        if show and key in outputs:
            if key == "best_pipeline":
                best_item = (key, name)
            elif key == "original":
                original_item = (key, name)
            else:
                display_items.append((key, name))
    
    # Reorder: original first, then best_pipeline, then others
    ordered_items = []
    if original_item:
        ordered_items.append(original_item)
    if best_item:
        ordered_items.append(best_item)
    ordered_items.extend(display_items)
    
    if ordered_items:
        # Create columns dynamically
        n_cols = 3
        cols = st.columns(n_cols)
        
        for idx, (key, name) in enumerate(ordered_items):
            col_idx = idx % n_cols
            with cols[col_idx]:
                st.subheader(name)
                st.image(outputs[key], width='stretch')
                if show_metrics and key != "original":
                    m = mse(outputs["original"], outputs[key])
                    p = psnr(outputs["original"], outputs[key])
                    st.caption(f"MSE: {m:.2f} | PSNR: {p:.2f} dB")
    
    # Metrics Summary
    if show_metrics:
        st.header("📈 Metrics Summary")
        
        # Group metrics by category
        noise_reduction = []
        contrast_enhancement = []
        detail_enhancement = []
        
        for key, (show, name) in display_map.items():
            if key in outputs and key != "original":
                m = mse(outputs["original"], outputs[key])
                p = psnr(outputs["original"], outputs[key])
                
                metric_item = {"name": name, "mse": m, "psnr": p}
                
                if key == "best_pipeline":
                    # Best pipeline gets its own category
                    continue  # Will be shown separately
                elif key in ["median", "bilateral"]:
                    noise_reduction.append(metric_item)
                elif key in ["contrast", "clahe"]:
                    contrast_enhancement.append(metric_item)
                elif key in ["sharpen", "unsharp"]:
                    detail_enhancement.append(metric_item)
        
        # Show Best Pipeline metrics separately if available
        if "best_pipeline" in outputs:
            st.subheader("⭐ Best Pipeline Result")
            cols = st.columns(1)
            with cols[0]:
                m = mse(outputs["original"], outputs["best_pipeline"])
                p = psnr(outputs["original"], outputs["best_pipeline"])
                st.markdown("**⭐ Best Pipeline (Denoise → Contrast → Sharpen)**")
                st.metric("MSE", f"{m:.2f}")
                st.metric("PSNR", f"{p:.2f} dB")
            st.divider()
        
        # Display metrics in organized sections
        if noise_reduction:
            st.subheader("🔇 Noise Reduction")
            cols = st.columns(len(noise_reduction))
            for idx, item in enumerate(noise_reduction):
                with cols[idx]:
                    st.markdown(f"**{item['name']}**")
                    st.metric("MSE", f"{item['mse']:.2f}")
                    st.metric("PSNR", f"{item['psnr']:.2f} dB")
        
        if contrast_enhancement:
            st.subheader("🌈 Contrast Enhancement")
            cols = st.columns(len(contrast_enhancement))
            for idx, item in enumerate(contrast_enhancement):
                with cols[idx]:
                    st.markdown(f"**{item['name']}**")
                    st.metric("MSE", f"{item['mse']:.2f}")
                    st.metric("PSNR", f"{item['psnr']:.2f} dB")
        
        if detail_enhancement:
            st.subheader("✨ Detail Enhancement")
            cols = st.columns(len(detail_enhancement))
            for idx, item in enumerate(detail_enhancement):
                with cols[idx]:
                    st.markdown(f"**{item['name']}**")
                    st.metric("MSE", f"{item['mse']:.2f}")
                    st.metric("PSNR", f"{item['psnr']:.2f} dB")
    
    # Download section - single button to download all as ZIP
    if image_source == "Upload Image" and uploaded_file is not None:
        st.header("💾 Download All Enhanced Images")
        
        # Create a zip file in memory
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # Add all enhanced images
            for key, (show, name) in display_map.items():
                if key in outputs:
                    # Convert numpy array to bytes
                    img_bytes = cv2.imencode('.png', outputs[key])[1].tobytes()
                    # Create safe filename
                    safe_name = name.lower().replace(' ', '_').replace('/', '_')
                    zip_file.writestr(f"{safe_name}.png", img_bytes)
            
        zip_buffer.seek(0)
        
        # Get base filename from uploaded file
        base_name = Path(uploaded_file.name).stem if uploaded_file else "enhanced_images"
        
        st.download_button(
            label="📦 Download All Images as ZIP",
            data=zip_buffer.getvalue(),
            file_name=f"{base_name}_all_enhancements.zip",
            mime="application/zip",
            help="Downloads all enhanced images as a ZIP file"
        )
    
    # Filter Descriptions Section
    st.header("📚 Understanding Image Enhancement Techniques")
    st.markdown("""
    This section explains what each enhancement technique does to your X-ray images in simple terms.
    """)
    
    with st.expander("🔍 Step 1: Denoising (Cleaning the Image)", expanded=True):
        st.markdown("""
        **Purpose:** Remove unwanted noise and graininess from the image while keeping important details.
        
        - **Median Filter**: Like taking a photo through a soft-focus lens. It smooths out random spots and speckles 
          by replacing each pixel with the middle value of its neighbors. Great for removing salt-and-pepper noise.
        
        - **Bilateral Filter**: A smart smoothing technique that removes noise but preserves sharp edges and important 
          details. Think of it as a selective blur that only affects noisy areas, leaving clear structures untouched.
        """)
    
    with st.expander("🌈 Step 2: Contrast Enhancement (Making Details Visible)", expanded=True):
        st.markdown("""
        **Purpose:** Improve the visibility of structures by making dark areas darker and bright areas brighter.
        
        - **Histogram Equalization**: Spreads out the brightness levels across the entire image, making everything 
          more visible. It's like adjusting the brightness and contrast of your TV to see details in both dark 
          and bright scenes. However, it can sometimes over-enhance certain areas.
        
        - **CLAHE (Contrast Limited Adaptive Histogram Equalization)**: A smarter version of histogram equalization 
          that works on small regions of the image instead of the whole image. It prevents over-enhancement and 
          produces more natural-looking results. Like having multiple brightness controls for different parts of 
          the image.
        """)
    
    with st.expander("✨ Step 3: Sharpening (Enhancing Details)", expanded=True):
        st.markdown("""
        **Purpose:** Make edges and fine details more crisp and clear, improving the visibility of small structures.
        
        - **Sharpening (Convolution)**: Uses a mathematical filter to increase the contrast at edges, making them 
          stand out more. It's like using a sharpening tool in photo editing software to make blurry edges crisp.
        
        - **Unsharp Masking**: A sophisticated sharpening technique that enhances edges and details without 
          over-sharpening. It works by creating a slightly blurred version of the image, then subtracting it from 
          the original to highlight edges. This produces natural-looking sharpness without artifacts.
        """)
    
    with st.expander("⭐ Best Pipeline (Complete Enhancement)", expanded=True):
        st.markdown("""
        **Purpose:** Apply all three steps in the optimal sequence for the best possible result.
        
        The **Best Pipeline** combines the most effective techniques from each step:
        1. **Denoise** using Median Filter (removes noise, preserves edges)
        2. **Contrast** using CLAHE (enhances visibility naturally)
        3. **Sharpen** using Unsharp Masking (brings out fine details)
        
        This sequential approach ensures that each step builds upon the previous one, resulting in a clean, 
        well-contrasted, and sharp image that's ideal for medical diagnosis.
        """)
    
    st.markdown("---")
    st.info("💡 **Tip:** For best results, use the **Best Pipeline** option, which automatically applies all three steps in the optimal order!")


if __name__ == "__main__":
    main()

