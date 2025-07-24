#!/usr/bin/env python3
"""
Box Overlap Checker for Figure Generation
Validates that boxes don't overlap and text fits properly within boxes.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
from typing import List, Tuple, Dict, Optional
import numpy as np


class BoxValidator:
    """Validates box placements and text sizing in figures."""
    
    def __init__(self, figsize: Tuple[float, float] = (10, 8), dpi: int = 100):
        self.figsize = figsize
        self.dpi = dpi
        self.boxes = []
        self.texts = []
        
    def add_box(self, x: float, y: float, width: float, height: float, 
                label: str = "", **kwargs) -> Dict:
        """Add a box to the collection with validation."""
        box = {
            'x': x,
            'y': y,
            'width': width,
            'height': height,
            'label': label,
            'kwargs': kwargs
        }
        self.boxes.append(box)
        return box
    
    def add_text(self, x: float, y: float, text: str, fontsize: float = 12, **kwargs) -> Dict:
        """Add text to the collection."""
        text_obj = {
            'x': x,
            'y': y,
            'text': text,
            'fontsize': fontsize,
            'kwargs': kwargs
        }
        self.texts.append(text_obj)
        return text_obj
    
    def check_box_overlap(self, box1: Dict, box2: Dict) -> bool:
        """Check if two boxes overlap."""
        # Box 1 boundaries
        x1_min = box1['x']
        x1_max = box1['x'] + box1['width']
        y1_min = box1['y']
        y1_max = box1['y'] + box1['height']
        
        # Box 2 boundaries
        x2_min = box2['x']
        x2_max = box2['x'] + box2['width']
        y2_min = box2['y']
        y2_max = box2['y'] + box2['height']
        
        # Check for overlap
        x_overlap = not (x1_max <= x2_min or x2_max <= x1_min)
        y_overlap = not (y1_max <= y2_min or y2_max <= y1_min)
        
        return x_overlap and y_overlap
    
    def check_text_box_overlap(self, text: Dict, box: Dict, fig, ax) -> bool:
        """Check if text overlaps with a box."""
        # Get text bounding box
        temp_text = ax.text(text['x'], text['y'], text['text'], 
                           fontsize=text['fontsize'], **text['kwargs'])
        
        # Get renderer
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        
        # Get text bbox in data coordinates
        bbox = temp_text.get_window_extent(renderer=renderer)
        bbox_data = bbox.transformed(ax.transData.inverted())
        
        # Remove temporary text
        temp_text.remove()
        
        # Text boundaries
        text_x_min = bbox_data.x0
        text_x_max = bbox_data.x1
        text_y_min = bbox_data.y0
        text_y_max = bbox_data.y1
        
        # Box boundaries
        box_x_min = box['x']
        box_x_max = box['x'] + box['width']
        box_y_min = box['y']
        box_y_max = box['y'] + box['height']
        
        # Check for overlap
        x_overlap = not (text_x_max <= box_x_min or box_x_max <= text_x_min)
        y_overlap = not (text_y_max <= box_y_min or box_y_max <= text_y_min)
        
        return x_overlap and y_overlap
    
    def check_text_fits_in_box(self, text: str, box: Dict, fontsize: float, 
                               fig, ax, padding: float = 0.1) -> Dict[str, bool]:
        """Check if text fits properly within a box."""
        # Place text at center of box
        text_x = box['x'] + box['width'] / 2
        text_y = box['y'] + box['height'] / 2
        
        # Create temporary text object
        temp_text = ax.text(text_x, text_y, text, fontsize=fontsize,
                           ha='center', va='center')
        
        # Get renderer
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        
        # Get text bbox in data coordinates
        bbox = temp_text.get_window_extent(renderer=renderer)
        bbox_data = bbox.transformed(ax.transData.inverted())
        
        # Remove temporary text
        temp_text.remove()
        
        # Calculate text dimensions
        text_width = bbox_data.width
        text_height = bbox_data.height
        
        # Box inner dimensions (with padding)
        box_inner_width = box['width'] * (1 - 2 * padding)
        box_inner_height = box['height'] * (1 - 2 * padding)
        
        # Check if text fits
        fits_width = text_width <= box_inner_width
        fits_height = text_height <= box_inner_height
        fits_overall = fits_width and fits_height
        
        # Calculate recommended font size if it doesn't fit
        recommended_fontsize = fontsize
        if not fits_overall:
            width_ratio = box_inner_width / text_width if text_width > 0 else 1
            height_ratio = box_inner_height / text_height if text_height > 0 else 1
            scale_factor = min(width_ratio, height_ratio)
            recommended_fontsize = fontsize * scale_factor * 0.9  # 0.9 for safety margin
        
        return {
            'fits': fits_overall,
            'fits_width': fits_width,
            'fits_height': fits_height,
            'text_width': text_width,
            'text_height': text_height,
            'box_inner_width': box_inner_width,
            'box_inner_height': box_inner_height,
            'recommended_fontsize': recommended_fontsize
        }
    
    def validate_all(self) -> Dict[str, List]:
        """Validate all boxes and texts."""
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        validation_results = {
            'box_overlaps': [],
            'text_box_overlaps': [],
            'text_fit_issues': []
        }
        
        # Check box-to-box overlaps
        for i in range(len(self.boxes)):
            for j in range(i + 1, len(self.boxes)):
                if self.check_box_overlap(self.boxes[i], self.boxes[j]):
                    validation_results['box_overlaps'].append({
                        'box1_index': i,
                        'box2_index': j,
                        'box1': self.boxes[i],
                        'box2': self.boxes[j]
                    })
        
        # Check text-to-box overlaps
        for text_idx, text in enumerate(self.texts):
            for box_idx, box in enumerate(self.boxes):
                if self.check_text_box_overlap(text, box, fig, ax):
                    validation_results['text_box_overlaps'].append({
                        'text_index': text_idx,
                        'box_index': box_idx,
                        'text': text,
                        'box': box
                    })
        
        # Check text fit within boxes
        for box in self.boxes:
            if box['label']:
                fit_result = self.check_text_fits_in_box(
                    box['label'], box, 12, fig, ax
                )
                if not fit_result['fits']:
                    validation_results['text_fit_issues'].append({
                        'box': box,
                        'fit_result': fit_result
                    })
        
        plt.close(fig)
        return validation_results
    
    def visualize_with_validation(self, title: str = "Box Layout Validation") -> None:
        """Visualize the layout with validation results."""
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # Get validation results
        validation = self.validate_all()
        
        # Draw boxes
        for i, box in enumerate(self.boxes):
            # Check if this box has overlap issues
            has_overlap = any(
                (overlap['box1_index'] == i or overlap['box2_index'] == i)
                for overlap in validation['box_overlaps']
            )
            
            # Choose color based on validation
            color = 'red' if has_overlap else 'lightblue'
            
            rect = Rectangle((box['x'], box['y']), box['width'], box['height'],
                           linewidth=2, edgecolor='black', facecolor=color,
                           alpha=0.5)
            ax.add_patch(rect)
            
            # Add box label if exists
            if box['label']:
                # Check if text fits
                fit_result = self.check_text_fits_in_box(
                    box['label'], box, 12, fig, ax
                )
                
                fontsize = fit_result['recommended_fontsize'] if not fit_result['fits'] else 12
                text_color = 'red' if not fit_result['fits'] else 'black'
                
                ax.text(box['x'] + box['width']/2, box['y'] + box['height']/2,
                       box['label'], ha='center', va='center',
                       fontsize=fontsize, color=text_color, weight='bold')
        
        # Draw standalone texts
        for text in self.texts:
            ax.text(text['x'], text['y'], text['text'],
                   fontsize=text['fontsize'], **text['kwargs'])
        
        # Set axis properties
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 8)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title(title, fontsize=16)
        
        # Add validation summary
        summary_text = f"Validation Summary:\n"
        summary_text += f"Box overlaps: {len(validation['box_overlaps'])}\n"
        summary_text += f"Text-box overlaps: {len(validation['text_box_overlaps'])}\n"
        summary_text += f"Text fit issues: {len(validation['text_fit_issues'])}"
        
        ax.text(0.02, 0.98, summary_text, transform=ax.transAxes,
               verticalalignment='top', fontsize=10,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        return fig, ax, validation


def example_usage():
    """Example of how to use the BoxValidator."""
    validator = BoxValidator()
    
    # Add some boxes
    validator.add_box(1, 1, 2, 1.5, "Box 1")
    validator.add_box(2.5, 2, 2, 1.5, "Box 2")  # This will overlap with Box 1
    validator.add_box(5, 1, 2, 1.5, "Box 3")
    validator.add_box(5, 4, 3, 1, "This is a very long text that might not fit")
    
    # Add some standalone text
    validator.add_text(8, 2, "Standalone\nText", fontsize=14)
    
    # Validate
    results = validator.validate_all()
    
    # Print validation results
    print("=== Validation Results ===")
    print(f"Box overlaps found: {len(results['box_overlaps'])}")
    for overlap in results['box_overlaps']:
        print(f"  - Box {overlap['box1_index']} overlaps with Box {overlap['box2_index']}")
    
    print(f"\nText-box overlaps found: {len(results['text_box_overlaps'])}")
    for overlap in results['text_box_overlaps']:
        print(f"  - Text {overlap['text_index']} overlaps with Box {overlap['box_index']}")
    
    print(f"\nText fit issues found: {len(results['text_fit_issues'])}")
    for issue in results['text_fit_issues']:
        print(f"  - Box with label '{issue['box']['label']}' has text fit issues")
        print(f"    Recommended font size: {issue['fit_result']['recommended_fontsize']:.1f}")
    
    # Visualize
    fig, ax, validation = validator.visualize_with_validation()
    plt.savefig('box_validation_example.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    example_usage()