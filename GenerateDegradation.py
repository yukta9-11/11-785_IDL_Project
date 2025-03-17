import os
import numpy as np
import random
from tqdm import tqdm
import glob
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw
import matplotlib.pyplot as plt


class ArtworkDegradation:
    """
    Class to apply various degradations commonly found in artwork that needs restoration:
    - Noise (Gaussian, Salt & Pepper)
    - Blur (simulating age and wear)
    - Fading/Discoloration (simulating light damage)
    - Cracks (simulating canvas/paint cracks)
    - Yellowing (simulating varnish aging)
    - Stains (simulating water damage or dirt)
    - Dust (simulating dusty surface artifacts)
    """

    def __init__(self, seed=42):
        np.random.seed(seed)
        random.seed(seed)

    def add_gaussian_noise(self, image, mean=0, var=0.01):
        """Add Gaussian noise to simulate film grain or scanning artifacts"""
        if isinstance(image, np.ndarray):
            row, col, ch = image.shape
            sigma = var ** 0.5
            gauss = np.random.normal(mean, sigma, (row, col, ch))
            gauss = gauss.reshape(row, col, ch)
            noisy = image + gauss
            return np.clip(noisy, 0, 1)
        else:
            image_np = np.array(image).astype(float) / 255.0
            noisy = self.add_gaussian_noise(image_np, mean, var)
            return Image.fromarray((noisy * 255).astype(np.uint8))

    def add_salt_pepper_noise(self, image, salt_prob=0.01, pepper_prob=0.01):
        """Add salt and pepper noise to simulate dust and scratches"""
        if isinstance(image, Image.Image):
            image_np = np.array(image).astype(float) / 255.0
            noisy = self.add_salt_pepper_noise(image_np, salt_prob, pepper_prob)
            return Image.fromarray((noisy * 255).astype(np.uint8))
        else:
            noisy = image.copy()
            # Salt noise (white pixels)
            salt_mask = np.random.random(image.shape[:2]) < salt_prob
            noisy[salt_mask] = 1

            # Pepper noise (black pixels)
            pepper_mask = np.random.random(image.shape[:2]) < pepper_prob
            noisy[pepper_mask] = 0

            return noisy

    def blur(self, image, kernel_size=5):
        """Apply blur to simulate age, wear or focus issues in old photographs"""
        if isinstance(image, Image.Image):
            return image.filter(ImageFilter.GaussianBlur(radius=kernel_size/3))
        else:
            # Convert numpy to PIL, apply blur, convert back
            pil_img = Image.fromarray((image * 255).astype(np.uint8))
            blurred = pil_img.filter(ImageFilter.GaussianBlur(radius=kernel_size/3))
            return np.array(blurred).astype(float) / 255.0

    def fade_colors(self, image, factor=0.7):
        """Reduce color saturation to simulate fading over time"""
        if isinstance(image, np.ndarray):
            # Convert to PIL
            pil_img = Image.fromarray((image * 255).astype(np.uint8))
            faded = self.fade_colors(pil_img, factor)
            return np.array(faded).astype(float) / 255.0
        else:
            # For PIL images
            enhancer = ImageEnhance.Color(image)
            return enhancer.enhance(factor)

    def add_cracks(self, image, num_cracks=50, max_length=100, width=1, color=0.2):
        """Add cracks to simulate aged artwork or canvas damage"""
        if isinstance(image, Image.Image):
            # Create a draw object
            img_copy = image.copy()
            draw = ImageDraw.Draw(img_copy)

            width_img, height = image.size

            # Convert color to PIL format (0-255)
            pil_color = int(color * 255)

            for _ in range(num_cracks):
                # Random starting point
                x = np.random.randint(0, width_img)
                y = np.random.randint(0, height)

                # Random length
                length = np.random.randint(10, max_length)

                # Random direction
                angle = np.random.uniform(0, 2 * np.pi)

                # Calculate endpoint
                x_end = int(x + length * np.cos(angle))
                y_end = int(y + length * np.sin(angle))

                # Draw line
                draw.line([(x, y), (x_end, y_end)], fill=(pil_color, pil_color, pil_color), width=width)

                # Add small branches occasionally
                if np.random.random() < 0.5:
                    branch_angle = angle + np.random.uniform(-np.pi/4, np.pi/4)
                    branch_length = length * np.random.uniform(0.2, 0.5)
                    x_branch = int(x_end + branch_length * np.cos(branch_angle))
                    y_branch = int(y_end + branch_length * np.sin(branch_angle))
                    draw.line([(x_end, y_end), (x_branch, y_branch)],
                             fill=(pil_color, pil_color, pil_color), width=width)

            return img_copy
        else:
            # Convert to PIL
            pil_img = Image.fromarray((image * 255).astype(np.uint8))
            cracked = self.add_cracks(pil_img, num_cracks, max_length, width, color)
            return np.array(cracked).astype(float) / 255.0

    def add_yellowing(self, image, intensity=0.3):
        """Add yellow tint to simulate aged varnish or paper"""
        if isinstance(image, Image.Image):
            # Convert to numpy for easier manipulation
            image_np = np.array(image).astype(float) / 255.0
            yellowed = self.add_yellowing(image_np, intensity)
            return Image.fromarray((yellowed * 255).astype(np.uint8))
        else:
            # Add yellow tint
            yellow_tint = np.ones_like(image) * np.array([0.9, 0.9, 0.1]) * intensity
            yellowed = image * (1 - intensity) + yellow_tint
            return np.clip(yellowed, 0, 1)

    def add_stains(self, image, num_stains=5, max_radius=50):
        """Add irregular stains to simulate water damage or dirt"""
        if isinstance(image, Image.Image):
            # Create copy and draw object
            img_copy = image.copy()
            draw = ImageDraw.Draw(img_copy)

            width_img, height = image.size

            for _ in range(num_stains):
                # Random center
                x = np.random.randint(0, width_img)
                y = np.random.randint(0, height)

                # Random radius
                radius = np.random.randint(10, max_radius)

                # Random stain color (brownish/yellowish)
                r = int(np.random.uniform(0.2, 0.6) * 255)
                g = int(np.random.uniform(0.1, 0.4) * 255)
                b = int(np.random.uniform(0, 0.2) * 255)

                # Create an irregular shape centered at (x,y)
                points = []
                num_points = 8
                for i in range(num_points):
                    angle = i * 2 * np.pi / num_points
                    dist = radius * np.random.uniform(0.5, 1.5)
                    px = int(x + dist * np.cos(angle))
                    py = int(y + dist * np.sin(angle))
                    points.append((px, py))

                # Draw the irregular polygon
                draw.polygon(points, fill=(r, g, b))

                # Apply a soft blur to the whole image to blend the stain
                img_copy = img_copy.filter(ImageFilter.GaussianBlur(radius=1))

            return img_copy
        else:
            # Convert to PIL
            pil_img = Image.fromarray((image * 255).astype(np.uint8))
            stained = self.add_stains(pil_img, num_stains, max_radius)
            return np.array(stained).astype(float) / 255.0

    def add_dust(self, image, density=0.005, size_range=(1, 3)):
        """Add dust particles to simulate dusty surface"""
        if isinstance(image, Image.Image):
            # Create copy and draw object
            img_copy = image.copy()
            draw = ImageDraw.Draw(img_copy)

            width_img, height = image.size

            # Number of dust particles
            num_particles = int(density * width_img * height)

            for _ in range(num_particles):
                # Random position
                x = np.random.randint(0, width_img)
                y = np.random.randint(0, height)

                # Random size
                size = np.random.randint(size_range[0], size_range[1]+1)

                # Random brightness (grayish)
                brightness = int(np.random.uniform(0.7, 0.9) * 255)

                # Draw dust particle
                draw.ellipse([(x-size, y-size), (x+size, y+size)],
                           fill=(brightness, brightness, brightness))

            return img_copy
        else:
            # Convert to PIL
            pil_img = Image.fromarray((image * 255).astype(np.uint8))
            dusty = self.add_dust(pil_img, density, size_range)
            return np.array(dusty).astype(float) / 255.0

    def random_degradation(self, image):
        """Apply a random degradation from the available options"""
        degradation_fns = [
            lambda img: self.add_gaussian_noise(img, var=np.random.uniform(0.01, 0.03)),
            lambda img: self.add_salt_pepper_noise(img, np.random.uniform(0.005, 0.02), np.random.uniform(0.005, 0.02)),
            lambda img: self.blur(img, kernel_size=np.random.randint(3, 7)),
            lambda img: self.fade_colors(img, factor=np.random.uniform(0.5, 0.9)),
            lambda img: self.add_cracks(img, num_cracks=np.random.randint(10, 70)),
            lambda img: self.add_yellowing(img, intensity=np.random.uniform(0.1, 0.4)),
            lambda img: self.add_stains(img, num_stains=np.random.randint(2, 8)),
            lambda img: self.add_dust(img, density=np.random.uniform(0.002, 0.01))
        ]

        # Choose a random degradation
        degradation_fn = random.choice(degradation_fns)
        return degradation_fn(image)

    def multiple_degradations(self, image, num_degradations=3):
        """Apply multiple random degradations to simulate realistic aging"""
        degraded = image

        if isinstance(image, Image.Image):
            # Define a realistic sequence of degradations for artwork
            degradation_sequence = [
                # First yellowing (age)
                lambda img: self.add_yellowing(img, intensity=np.random.uniform(0.1, 0.4)),

                # Then cracks and fading
                lambda img: self.add_cracks(img, num_cracks=np.random.randint(10, 70))
                            if np.random.random() > 0.3 else
                            self.fade_colors(img, factor=np.random.uniform(0.6, 0.9)),

                # Finally stains, dust, or noise
                lambda img: random.choice([
                    lambda x: self.add_stains(x, num_stains=np.random.randint(1, 5)),
                    lambda x: self.add_dust(x, density=np.random.uniform(0.002, 0.008)),
                    lambda x: self.add_salt_pepper_noise(x, 0.01, 0.01)
                ])(img)
            ]

            # Apply degradations in sequence
            for i in range(min(num_degradations, len(degradation_sequence))):
                degraded = degradation_sequence[i](degraded)

            # Apply additional random degradations if requested
            for i in range(min(num_degradations, len(degradation_sequence)), num_degradations):
                degraded = self.random_degradation(degraded)

            return degraded
        else:
            # Convert to PIL, then process
            pil_img = Image.fromarray((image * 255).astype(np.uint8))
            degraded_pil = self.multiple_degradations(pil_img, num_degradations)
            return np.array(degraded_pil).astype(float) / 255.0


def process_images(input_dir, output_dir=None, degradation_type='random', num_degradations=3, save_combined=True):
    """
    Process all images in the input directory and save degraded versions

    Args:
        input_dir: Directory containing input artwork images
        output_dir: Directory to save degraded images (if None, creates 'degraded' inside input_dir)
        degradation_type: Type of degradation to apply
        num_degradations: Number of degradations to apply if degradation_type is 'multiple'
        save_combined: Whether to save a combined image showing all original and degraded pairs
    """
    if output_dir is None:
        output_dir = os.path.join(input_dir, 'degraded')

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get all image files
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(input_dir, ext)))
        image_files.extend(glob.glob(os.path.join(input_dir, ext.upper())))

    if not image_files:
        print(f"No image files found in {input_dir}")
        return

    print(f"Found {len(image_files)} image files")

    # Initialize degradation class
    degrader = ArtworkDegradation()

    # Keep track of processed images for combined visualization
    processed_pairs = []

    # Process each image
    for img_path in tqdm(image_files, desc="Processing artwork images"):
        # Skip if in degraded folder
        if 'degraded' in img_path:
            continue

        try:
            # Open image
            img = Image.open(img_path).convert('RGB')

            # Apply degradation based on type
            if degradation_type == 'gaussian_noise':
                degraded = degrader.add_gaussian_noise(img)
            elif degradation_type == 'salt_pepper':
                degraded = degrader.add_salt_pepper_noise(img)
            elif degradation_type == 'blur':
                degraded = degrader.blur(img)
            elif degradation_type == 'fade':
                degraded = degrader.fade_colors(img)
            elif degradation_type == 'cracks':
                degraded = degrader.add_cracks(img)
            elif degradation_type == 'yellowing':
                degraded = degrader.add_yellowing(img)
            elif degradation_type == 'stains':
                degraded = degrader.add_stains(img)
            elif degradation_type == 'dust':
                degraded = degrader.add_dust(img)
            elif degradation_type == 'multiple':
                degraded = degrader.multiple_degradations(img, num_degradations)
            else:  # random
                degraded = degrader.random_degradation(img)

            # Save degraded image
            filename = os.path.basename(img_path)
            degraded_path = os.path.join(output_dir, filename)
            degraded.save(degraded_path)

            # Store the image pair for later visualization
            if save_combined:
                processed_pairs.append((img, degraded, filename))

        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    # Create combined visualization of all processed images
    if save_combined and processed_pairs:
        save_combined_visualization(processed_pairs, output_dir)


def save_combined_visualization(processed_pairs, output_dir):
    """
    Create and save a combined visualization of all original and degraded image pairs

    Args:
        processed_pairs: List of tuples (original_img, degraded_img, filename)
        output_dir: Directory to save the combined visualization
    """
    num_images = len(processed_pairs)

    if num_images == 0:
        return

    # Determine grid size
    if num_images <= 5:
        cols = num_images
        rows = 1
    else:
        cols = 5
        rows = (num_images + 4) // 5  # Ceiling division

    # Create a figure to hold all original-degraded pairs
    fig, axes = plt.subplots(rows * 2, cols, figsize=(cols * 4, rows * 6))

    # Make axes 2D if it's 1D
    if rows == 1:
        axes = np.array([axes[:cols], axes[cols:]])

    # Ensure axes is always 2D
    if num_images == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    # Flatten axes for easy indexing
    if rows * cols > 1:
        axes = axes.reshape(rows * 2, cols)

    # Turn off all axes first
    for ax in axes.flatten():
        ax.axis('off')

    # Plot images
    for i, (original, degraded, filename) in enumerate(processed_pairs):
        if i >= rows * cols:
            print(f"Warning: Not all images shown in the combined visualization (limit: {rows * cols})")
            break

        row_idx = (i // cols) * 2
        col_idx = i % cols

        # Plot original image
        axes[row_idx, col_idx].imshow(original)
        axes[row_idx, col_idx].set_title(f"Original: {filename}")
        axes[row_idx, col_idx].axis('off')

        # Plot degraded image
        axes[row_idx + 1, col_idx].imshow(degraded)
        axes[row_idx + 1, col_idx].set_title(f"Degraded: {filename}")
        axes[row_idx + 1, col_idx].axis('off')

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'all_processed_images.png')
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Combined visualization saved to {output_path}")


def visualize_degradations(image_path):
    """
    Visualize all artwork degradation types on a single image

    Args:
        image_path: Path to the artwork image to visualize
    """
    try:
        # Open image
        img = Image.open(image_path).convert('RGB')

        # Initialize degradation class
        degrader = ArtworkDegradation()

        # Apply all degradations
        degradations = {
            'Original': img,
            'Gaussian Noise': degrader.add_gaussian_noise(img),
            'Salt & Pepper': degrader.add_salt_pepper_noise(img),
            'Blur': degrader.blur(img),
            'Color Fading': degrader.fade_colors(img),
            'Cracks': degrader.add_cracks(img),
            'Yellowing': degrader.add_yellowing(img),
            'Stains': degrader.add_stains(img),
            'Dust': degrader.add_dust(img),
            'Multiple (Realistic)': degrader.multiple_degradations(img)
        }

        # Plot
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        axes = axes.flatten()

        for i, (title, degraded) in enumerate(degradations.items()):
            if i < len(axes):
                axes[i].imshow(degraded)
                axes[i].set_title(title)
                axes[i].axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(image_path), 'artwork_degradation_examples.png'))
        plt.close()

        print(f"Visualization saved as 'artwork_degradation_examples.png'")

    except Exception as e:
        print(f"Error visualizing degradations: {e}")


# Direct execution for Colab - modify these paths as needed
if __name__ == "__main__":
    # Set your Google Drive paths here
    input_dir = '/content/drive/MyDrive/IDL_project_team_20/data/painting'
    output_dir = '/content/drive/MyDrive/IDL_project_team_20/data/painting/degraded'

    # Set degradation parameters
    degradation_type = 'random'  # Changed to 'random' to apply only one effect per image
    num_degradations = 1  # Only used if degradation_type is 'multiple'

    # Save combined visualization of all processed images
    save_combined = True

    # For visualization (set to None to skip visualization)
    visualize = None  # Example: '/content/drive/MyDrive/IDL_project_team_20/data/sample_image.jpg'

    # Process the images
    if visualize:
        visualize_degradations(visualize)
    else:
        process_images(input_dir, output_dir, degradation_type, num_degradations, save_combined)
