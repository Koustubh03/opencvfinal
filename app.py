import io
import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Function to apply selected filter
# Function to adjust brightness
def adjust_brightness(image, factor):
    result = image.astype(np.float32) * factor
    return np.clip(result, 0, 255).astype(np.uint8)

# Function to adjust contrast
def adjust_contrast(image, factor):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_image = gray_image.astype(np.float32)
    gray_image = (gray_image - 128) * factor + 128
    return cv2.cvtColor(np.clip(gray_image, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

# Function to adjust saturation
def adjust_saturation(image, factor):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv_image[:, :, 1] = np.clip(hsv_image[:, :, 1] * factor, 0, 255)
    return cv2.cvtColor(hsv_image.astype(np.uint8), cv2.COLOR_HSV2BGR)
def apply_edges_filter(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_image, 50, 150)
    edges_image = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    return edges_image
# Function to apply the Sepia filter using OpenCV
def apply_sepia_filter(image):
    sepia_kernel = np.array([[0.272, 0.534, 0.131],
                             [0.349, 0.686, 0.168],
                             [0.393, 0.769, 0.189]])
    sepia_image = cv2.transform(image, sepia_kernel)

    sepia_image = cv2.cvtColor(sepia_image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(sepia_image)

# Function to apply the Blur filter using OpenCV
def apply_blur_filter(image, radius):
    kernel_size = 2 * radius + 1
    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
    return cv2.filter2D(image, -1, kernel)

# Function to apply the Emboss filter using OpenCV
def apply_emboss_filter(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    emboss_kernel = np.array([[2, 0, 0],
                              [0, -1, 0],
                              [0, 0, -1]])
    emboss_image = cv2.filter2D(gray, -1, emboss_kernel)
    emboss_image = cv2.cvtColor(emboss_image, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(emboss_image)

# Function to apply the Sharpen filter using OpenCV
def apply_sharpen_filter(image, factor):
    kernel = np.array([[-1, -1, -1],
                       [-1, 9 + factor, -1],
                       [-1, -1, -1]], dtype=np.float32)
    return cv2.filter2D(image, -1, kernel)

# Function to apply the 3D filter using OpenCV
def apply_3d_filter(image, intensity):
    depth = cv2.applyColorMap(image, cv2.COLORMAP_JET)
    result = cv2.addWeighted(image, 1, depth, intensity, 0)
    return result

# Function to apply the Distort filter using OpenCV
def apply_distort_filter(image, intensity):
    height, width, _ = image.shape
    distorted_image = np.zeros_like(image)

    for x in range(width):
        for y in range(height):
            x_distort = int(x + intensity * np.sin(2 * np.pi * y / 128.0))
            y_distort = int(y + intensity * np.cos(2 * np.pi * x / 128.0))

            if 0 <= x_distort < width and 0 <= y_distort < height:
                distorted_image[y, x] = image[y_distort, x_distort]
            else:
                distorted_image[y, x] = (0, 0, 0)

    return distorted_image

# Function to apply the Noise filter using OpenCV
def apply_noise_filter(image, intensity):
    noise = np.random.randint(-intensity, intensity, size=image.shape, dtype=np.int8)
    noisy_image = cv2.add(image, noise, dtype=cv2.CV_8U)
    return noisy_image

# Function to apply the Pixelate filter using OpenCV
def apply_pixelate_filter(image, pixel_size):
    height, width, _ = image.shape
    small = cv2.resize(image, (pixel_size, pixel_size), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(small, (width, height), interpolation=cv2.INTER_NEAREST)

# Function to apply the Render filter using OpenCV
def apply_render_filter(image, intensity):
    return cv2.convertScaleAbs(image, alpha=intensity, beta=0)

# Function to apply the Stylize filter using OpenCV
def apply_stylize_filter(image, intensity):
    return cv2.stylization(image, sigma_s=60, sigma_r=intensity)

# Function to apply the Opacity filter using OpenCV
def apply_opacity_filter(image, opacity):
    image_with_alpha = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    image_with_alpha[:, :, 3] = int(255 * opacity)
    return image_with_alpha

# Function to adjust brightness, contrast, and saturation combined using OpenCV
def adjust_image_properties(image, brightness, contrast, saturation):
    result = cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)
    hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation, 0, 255)
    result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return result

# Function to add watermark to the image
def add_watermark(image, text):
    watermarked_image = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2
    font_color = (128, 128, 128)
    font_thickness = 5

    # Get the size of the watermark text
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)

    # Calculate the position for center alignment
    position = ((image.shape[1] - text_size[0]) // 2, (image.shape[0] + text_size[1]) // 2)

    # Add the watermark text
    cv2.putText(watermarked_image, text, position, font, font_scale, font_color, font_thickness, lineType=cv2.LINE_AA)

    return watermarked_image


# Streamlit app
st.title('Image Filtering: Unveil the Hidden Beauty of Pixels!')
st.markdown("")  # Add an empty line after the title
st.markdown("Transform Ordinary Photos into Extraordinary Masterpieces with Our App! Unleash Your Creativity, Enhance Details, and Add a Dash of Magic to Your Visual World. Download Now and See Pixels Come to Life!")
st.markdown("")  # Add an empty line after the title
st.markdown("")  # Add an empty line after the title
st.markdown("")  # Add an empty line after the title
# List of available filter names
filter_names = ['Original','Adjustall(BSC)', 'Grayscale', 'Sepia', 'Invert', 'Blur', 'Edges', 'Emboss', 'Sharpen', '3D', 'Blur Gallery', 'Distort', 'Noise', 'Pixelate', 'Render', 'Opacity']
filter_names1 = ['Cartoonise']
# Logo image


st.sidebar.header("Online Phototshop")
logo_image = Image.open('pngwing.com.png')
st.sidebar.image(logo_image, use_column_width=False,width=180)


# Create an unordered list of filter names in the sidebar
st.sidebar.header("Available Filters:")
st.sidebar.markdown("<ul style='list-style-type: disc;color: white;'>"+"".join(["<li>{}</li>".format(name) for name in filter_names])+"</ul>", unsafe_allow_html=True)

st.sidebar.header("Cartoon Animation:")
st.sidebar.markdown("<ul style='list-style-type: disc;color: white;'>"+"".join(["<li>{}</li>".format(name) for name in filter_names1])+"</ul>", unsafe_allow_html=True)
# Upload an image
uploaded_image = st.file_uploader('Upload an image', type=['jpg', 'png', 'jpeg'])

if uploaded_image is not None:
    try:
        # Use OpenCV to open the uploaded image
        pil_image = Image.open(uploaded_image)
        # Convert PIL image to OpenCV format (numpy array)
        image = np.array(pil_image)

        if image is not None:
            # Display the original image
            st.image(image, use_column_width=True, caption='Original Image')

            # Choose a filter
            filter_name = st.selectbox('Select a filter', ['Select Filter','Original','Adjustall(BSC)', 'Grayscale', 'Sepia', 'Invert', 'Blur', 'Edges', 'Emboss', 'Sharpen', '3D', 'Distort', 'Noise', 'Pixelate', 'Render', 'Opacity', 'Cartoonise'])

            filter_params = {}

            if filter_name =='Adjustall(BSC)':
                filter_params['brightness'] = st.slider('Brightness', 0.5, 2.0, 1.0)
                filter_params['contrast'] = st.slider('Contrast', 0.5, 2.0, 1.0)
                filter_params['saturation'] = st.slider('Saturation', 0.5, 2.0, 1.0)
            if filter_name == 'Change Color':
                filter_params['red'] = st.slider('Red', 0.0, 2.0, 1.0)
                filter_params['green'] = st.slider('Green', 0.0, 2.0, 1.0)
                filter_params['blue'] = st.slider('Blue', 0.0, 2.0, 1.0)


            elif filter_name == 'Blur':
                filter_params['blur_radius'] = st.slider('Blur Radius', 0, 10, 2)
            elif filter_name == 'Sharpen':
                filter_params['sharpen_factor'] = st.slider('Sharpen Factor', 0.5, 2.0, 1.0)
            elif filter_name == '3D':
                filter_params['intensity'] = st.slider('3D Intensity', 0.1, 2.0, 1.0)
            elif filter_name == 'Distort':
                filter_params['distort_intensity'] = st.slider('Distort Intensity', 0, 100, 20)
            elif filter_name == 'Noise':
                filter_params['noise_intensity'] = st.slider('Noise Intensity', 0, 50, 10)
            elif filter_name == 'Pixelate':
                filter_params['pixelate_intensity'] = st.slider('Pixelate Intensity', 2, 20, 5)
            elif filter_name == 'Render':
                filter_params['render_intensity'] = st.slider('Render Intensity', 0.1, 2.0, 1.0)
            elif filter_name == 'Cartoonise':
                filter_params['stylize_intensity'] = st.slider('Stylize Intensity', 0.1, 2.0, 1.0)
            elif filter_name == 'Opacity':
                filter_params['opacity'] = st.slider('Opacity', 0.0, 1.0, 1.0)

            if st.button('Apply Filter'):
                # Apply the selected filter using OpenCV
                if filter_name == 'Adjustall(BSC)':
                    filtered_image = adjust_brightness(image, filter_params['brightness'])
                elif filter_name == 'Contrast':
                    filtered_image = adjust_contrast(image, filter_params['contrast'])
                elif filter_name == 'Saturation':
                    filtered_image = adjust_saturation(image, filter_params['saturation'])
                elif filter_name == 'Grayscale':
                    filtered_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    filtered_image = cv2.cvtColor(filtered_image, cv2.COLOR_GRAY2BGR)
                elif filter_name == 'Original':
                    filtered_image = image
                elif filter_name == 'Sepia':
                    filtered_image = apply_sepia_filter(image)
                elif filter_name == 'Invert':
                    filtered_image = cv2.bitwise_not(image)
                elif filter_name == 'Blur':
                    filtered_image = apply_blur_filter(image, filter_params['blur_radius'])
                elif filter_name == 'Edges':
                    filtered_image = apply_edges_filter(image)
                elif filter_name == 'Emboss':
                    filtered_image = apply_emboss_filter(image)
                elif filter_name == 'Sharpen':
                    filtered_image = apply_sharpen_filter(image, filter_params['sharpen_factor'])
                elif filter_name == '3D':
                    filtered_image = apply_3d_filter(image, filter_params['intensity'])

                elif filter_name == 'Distort':
                    filtered_image = apply_distort_filter(image, filter_params['distort_intensity'])
                elif filter_name == 'Noise':
                    filtered_image = apply_noise_filter(image, filter_params['noise_intensity'])
                elif filter_name == 'Pixelate':
                    filtered_image = apply_pixelate_filter(image, filter_params['pixelate_intensity'])
                elif filter_name == 'Render':
                    filtered_image = apply_render_filter(image, filter_params['render_intensity'])
                elif filter_name == 'Cartoonise':
                    filtered_image = apply_stylize_filter(image, filter_params['stylize_intensity'])
                elif filter_name == 'Opacity':
                    filtered_image = apply_opacity_filter(image, filter_params['opacity'])

                # Add watermark to the filtered image

                # Display the filtered image with watermark
                st.image(filtered_image, use_column_width=True, caption=f'{filter_name} Filtered Image')
                watermark_text = "@Copyright"  # Change this to your desired watermark text
                watermarked_image = add_watermark(filtered_image, watermark_text)

                # Add a download button for the filtered image with watermark
                watermarked_image_bytes = cv2.imencode(".png", watermarked_image)[1].tobytes()
                st.download_button('Download Filtered Image', watermarked_image_bytes, 'filtered_image_with_watermark.png', 'image/png')
        else:
            st.warning('Invalid image format. Please upload a valid image file.')
    except Exception as e:
        st.error(f'An error occurred: {str(e)}')
