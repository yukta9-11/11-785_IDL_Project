"""
This module provides tools to evaluate image quality using reference-based metrics:
- PSNR (↑): Higher values indicate better fidelity to the reference image
- SSIM (↑): Higher values indicate better structural similarity to the reference image
"""

import os
import numpy as np
import cv2
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ImageQualityEvaluator:
    """
    A class for evaluating image quality using reference-based metrics.
    
    This class provides methods to calculate:
    - PSNR (Peak Signal-to-Noise Ratio)
    - SSIM (Structural Similarity Index)
    
    It can process single images or entire directories, and provides
    visualization and statistical analysis of the results.
    """
    
    def compute_psnr(self, img1, img2):
        """
        Compute Peak Signal-to-Noise Ratio (PSNR) between two images.
        
        PSNR measures the ratio between the maximum possible power of an image
        and the power of corrupting noise. Higher values indicate better quality.
        
        Args:
            img1 (numpy.ndarray): Reference image
            img2 (numpy.ndarray): Test image to be evaluated
            
        Returns:
            float: PSNR value in decibels (dB)
        """
        return peak_signal_noise_ratio(img1, img2, data_range=255)
    
    def compute_ssim(self, img1, img2):
        """
        Compute Structural Similarity Index (SSIM) between two images.
        
        SSIM measures the similarity between two images based on luminance,
        contrast and structure. Higher values indicate greater similarity.
        
        Args:
            img1 (numpy.ndarray): Reference image
            img2 (numpy.ndarray): Test image to be evaluated
            
        Returns:
            float: SSIM value between -1 and 1 (where 1 means identical images)
        """
        # Ensure both images have the same dimensions
        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
            
        # Check if image is RGB or grayscale
        if len(img1.shape) == 3 and img1.shape[2] == 3:
            return structural_similarity(img1, img2, channel_axis=2, data_range=255)
        else:
            return structural_similarity(img1, img2, data_range=255)
    
    def load_image(self, img_path):
        """
        Load and preprocess an image from a file path.
        
        Args:
            img_path (str): Path to the image file
            
        Returns:
            numpy.ndarray: Loaded image in RGB format
            
        Raises:
            FileNotFoundError: If the image file doesn't exist
        """
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image file not found: {img_path}")
            
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Failed to load image: {img_path}")
            
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    def evaluate_image(self, img_path, reference_path, metrics=None):
        """
        Evaluate a single image against a reference image using specified metrics.
        
        Args:
            img_path (str): Path to the image to evaluate
            reference_path (str): Path to the reference image for comparison
            metrics (list, optional): List of metrics to compute. If None, computes both 'psnr' and 'ssim'
            
        Returns:
            dict: Dictionary with metric names as keys and computed values as values
        """
        if metrics is None:
            metrics = ['psnr', 'ssim']
        
        results = {}
        
        try:
            img = self.load_image(img_path)
            
            try:
                ref_img = self.load_image(reference_path)
                
                if 'psnr' in metrics:
                    results['psnr'] = self.compute_psnr(ref_img, img)
                if 'ssim' in metrics:
                    results['ssim'] = self.compute_ssim(ref_img, img)
            except Exception as e:
                logger.error(f"Error processing reference image: {e}")
                for metric in metrics:
                    results[metric] = float('nan')
        except Exception as e:
            logger.error(f"Error evaluating image {img_path}: {e}")
            for metric in metrics:
                results[metric] = float('nan')
        
        return results
    
    def evaluate_directory(self, test_dir, reference_dir, metrics=None):
        """
        Evaluate all images in a directory against their reference counterparts.
        
        Args:
            test_dir (str): Directory containing test images to evaluate
            reference_dir (str): Directory containing reference images
            metrics (list, optional): List of metrics to compute, defaults to ['psnr', 'ssim']
            
        Returns:
            dict: Dictionary with metric names as keys and lists of values as values
        """
        if not os.path.isdir(test_dir):
            raise ValueError(f"Test directory not found: {test_dir}")
        
        if not os.path.isdir(reference_dir):
            raise ValueError(f"Reference directory not found: {reference_dir}")
        
        # Get image paths
        test_images = [os.path.join(test_dir, f) for f in os.listdir(test_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if not test_images:
            logger.warning(f"No valid images found in {test_dir}")
            return {}
        
        # Initialize results dictionary
        if metrics is None:
            metrics = ['psnr', 'ssim']
            
        results = {metric: [] for metric in metrics}
        
        # Process each image
        for img_path in tqdm(test_images, desc="Evaluating images"):
            filename = os.path.basename(img_path)
            ref_path = os.path.join(reference_dir, filename)
            
            # Skip if reference doesn't exist
            if not os.path.exists(ref_path):
                logger.warning(f"Reference not found for {filename}, skipping")
                continue
            
            # Evaluate image
            image_results = self.evaluate_image(img_path, ref_path, metrics)
            
            # Add results
            for metric, value in image_results.items():
                results[metric].append(value)
        
        return results
    
    def get_summary_stats(self, results):
        """
        Calculate comprehensive summary statistics for each metric.
        
        Args:
            results (dict): Dictionary with metric names as keys and lists of values as values
            
        Returns:
            dict: Dictionary with metric names as keys and dictionaries of statistics as values
        """
        summary = {}
        
        for metric, values in results.items():
            valid_values = [v for v in values if not np.isnan(v)]
            if not valid_values:
                logger.warning(f"No valid values for metric: {metric}")
                continue
                
            summary[metric] = {
                'mean': np.mean(valid_values),
                'median': np.median(valid_values),
                'std': np.std(valid_values),
                'min': np.min(valid_values),
                'max': np.max(valid_values),
                'count': len(valid_values),
                'total_images': len(values),
                'percent_valid': 100 * len(valid_values) / len(values) if values else 0
            }
        
        return summary
    
    def plot_results(self, results, save_path=None):
        """
        Create boxplot visualizations of the evaluation results.
        
        Args:
            results (dict): Dictionary with metric names as keys and lists of values as values
            save_path (str, optional): Path to save the visualization image
            
        Returns:
            matplotlib.figure.Figure: The created figure object (if any metrics were valid)
        """
        # Filter out metrics with no valid values
        valid_metrics = {m: v for m, v in results.items() if any(not np.isnan(x) for x in v)}
        
        if not valid_metrics:
            logger.warning("No valid metrics to visualize")
            return None
            
        # Create plot
        fig, axes = plt.subplots(1, len(valid_metrics), figsize=(4 * len(valid_metrics), 5))
        if len(valid_metrics) == 1:
            axes = [axes]
            
        for i, (metric, values) in enumerate(valid_metrics.items()):
            valid_values = [v for v in values if not np.isnan(v)]
            
            # Create boxplot
            box = axes[i].boxplot(valid_values, patch_artist=True)
            
            # Color the boxes
            for patch in box['boxes']:
                patch.set_facecolor('lightblue')
            
            # Add title and labels
            axes[i].set_title(f"{metric.upper()} (n={len(valid_values)})")
            axes[i].set_ylabel("Score (↑ better)")
            
            # Add mean as a text annotation
            mean_val = np.mean(valid_values)
            axes[i].annotate(f"Mean: {mean_val:.3f}", 
                            xy=(1.05, 0.5), 
                            xycoords='axes fraction',
                            verticalalignment='center')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Plot saved to {save_path}")
        
        return fig
    
    def print_summary(self, summary):
        """
        Print a comprehensive summary of evaluation results.
        
        Args:
            summary (dict): Dictionary with metric names as keys and dictionaries of statistics as values
        """
        print("\n" + "="*50)
        print("REFERENCE-BASED IMAGE QUALITY EVALUATION SUMMARY")
        print("="*50)
        
        for metric, stats in summary.items():
            print(f"\n{metric.upper()} (↑ better):")
            print(f"  Mean: {stats['mean']:.4f}")
            print(f"  Median: {stats['median']:.4f}")
            print(f"  Std Dev: {stats['std']:.4f}")
            print(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
            print(f"  Valid samples: {stats['count']}/{stats['total_images']} ({stats['percent_valid']:.1f}%)")
        
        print("\n" + "="*50)


def evaluate_images(test_dir, reference_dir, metrics=None, save_dir=None):
    """
    Comprehensive function to evaluate image quality using reference-based metrics.
    
    This function streamlines the process of:
    1. Loading images from directories
    2. Computing reference-based quality metrics (PSNR, SSIM)
    3. Analyzing results statistically
    4. Visualizing the results
    5. Saving results to files (optional)
    
    Args:
        test_dir (str): Directory with test images to evaluate
        reference_dir (str): Directory with reference images
        metrics (list, optional): List of metrics to compute, defaults to ['psnr', 'ssim']
        save_dir (str, optional): Directory to save results and visualizations
    
    Returns:
        tuple: (results, summary) - Detailed results and summary statistics
               - results: Dictionary with metric names as keys and lists of values
               - summary: Dictionary with metric names as keys and dictionaries of statistics
    """
    evaluator = ImageQualityEvaluator()
    
    # Default metrics
    if metrics is None:
        metrics = ['psnr', 'ssim']
    
    # Evaluate images
    results = evaluator.evaluate_directory(test_dir, reference_dir, metrics)
    
    # Calculate summary statistics
    summary = evaluator.get_summary_stats(results)
    
    # Print summary
    evaluator.print_summary(summary)
    
    # Create visualization
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plot_path = os.path.join(save_dir, "metrics_plot.png")
        evaluator.plot_results(results, plot_path)
        
        # Save results as CSV
        try:
            import pandas as pd
            # Save detailed results
            pd.DataFrame(results).to_csv(os.path.join(save_dir, "detailed_results.csv"), index=False)
            
            # Save summary
            summary_data = {}
            for metric, stats in summary.items():
                for stat_name, value in stats.items():
                    summary_data[f"{metric}_{stat_name}"] = [value]
            pd.DataFrame(summary_data).to_csv(os.path.join(save_dir, "summary_stats.csv"), index=False)
            
            logger.info(f"Results saved to {save_dir}")
        except ImportError:
            logger.warning("pandas not installed, CSV export skipped")
    else:
        evaluator.plot_results(results)
    
    return results, summary
