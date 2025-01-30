from PIL import Image

def overlay_images(rgb_image_path, rgba_image_path, output_image_path):
    # Open the RGB and RGBA images
    rgb_image = Image.open(rgb_image_path).convert("RGB")
    rgba_image = Image.open(rgba_image_path).convert("RGBA")

    # Resize the RGBA image to the same size as the RGB image (if necessary)
    rgba_image = rgba_image.resize(rgb_image.size)

    # Separate the RGBA image into its RGB channels and alpha channel
    rgba_rgb, alpha = rgba_image.split()[:3], rgba_image.split()[3]

    # Create a background image from the RGB image
    background = rgb_image.convert("RGBA")

    # Overlay the RGBA image onto the RGB background using the alpha channel for transparency
    overlay = Image.alpha_composite(background, rgba_image)

    # Save the resulting image
    overlay.save(output_image_path)

def rgb_to_rgba(rgb_image_path, output_image_path):
    # Open the RGB image
    rgb_image = Image.open(rgb_image_path).convert("RGB")
    
    # Convert the RGB image to RGBA by adding an alpha channel (fully opaque)
    rgba_image = rgb_image.convert("RGBA")
    
    # Save the resulting RGBA image
    rgba_image.save(output_image_path)

# Example usage:
overlay_images('data/images/original_0.png', 'results/ring_render_new_light.png', 'results/result_light_ring_on_hand.png')