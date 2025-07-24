#!/usr/bin/env python3
"""
Smart Box Layout for Figure Generation
Automatically adjusts box positions and text sizes to avoid overlaps.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import itertools


@dataclass
class SmartBox:
    """Represents a box with automatic layout capabilities."""
    x: float
    y: float
    width: float
    height: float
    label: str = ""
    fontsize: float = 12
    padding: float = 0.1
    color: str = 'lightblue'
    
    def get_bounds(self) -> Tuple[float, float, float, float]:
        """Get box boundaries (x_min, y_min, x_max, y_max)."""
        return (self.x, self.y, self.x + self.width, self.y + self.height)
    
    def overlaps_with(self, other: 'SmartBox') -> bool:
        """Check if this box overlaps with another box."""
        x1_min, y1_min, x1_max, y1_max = self.get_bounds()
        x2_min, y2_min, x2_max, y2_max = other.get_bounds()
        
        x_overlap = not (x1_max <= x2_min or x2_max <= x1_min)
        y_overlap = not (y1_max <= y2_min or y2_max <= y1_min)
        
        return x_overlap and y_overlap


class SmartBoxLayout:
    """Intelligent box layout manager with automatic overlap resolution."""
    
    def __init__(self, figsize: Tuple[float, float] = (10, 8), 
                 xlim: Tuple[float, float] = (0, 10),
                 ylim: Tuple[float, float] = (0, 8)):
        self.figsize = figsize
        self.xlim = xlim
        self.ylim = ylim
        self.boxes: List[SmartBox] = []
        self.min_spacing = 0.2  # Minimum spacing between boxes
        
    def add_box(self, x: float, y: float, width: float, height: float, 
                label: str = "", **kwargs) -> SmartBox:
        """Add a box and automatically adjust position if needed."""
        box = SmartBox(x, y, width, height, label, **kwargs)
        
        # Check for overlaps and adjust position
        adjusted_box = self._find_non_overlapping_position(box)
        self.boxes.append(adjusted_box)
        
        return adjusted_box
    
    def _find_non_overlapping_position(self, new_box: SmartBox) -> SmartBox:
        """Find a non-overlapping position for the new box."""
        # Check if current position works
        if not any(new_box.overlaps_with(existing) for existing in self.boxes):
            return new_box
        
        # Try different positions
        search_radius = 0.5
        max_attempts = 100
        attempt = 0
        
        while attempt < max_attempts:
            # Generate candidate positions in a spiral pattern
            angle = attempt * 0.5
            radius = search_radius * (1 + attempt / 20)
            
            dx = radius * np.cos(angle)
            dy = radius * np.sin(angle)
            
            # Try new position
            candidate = SmartBox(
                new_box.x + dx,
                new_box.y + dy,
                new_box.width,
                new_box.height,
                new_box.label,
                new_box.fontsize,
                new_box.padding,
                new_box.color
            )
            
            # Check if within bounds
            x_min, y_min, x_max, y_max = candidate.get_bounds()
            if (x_min >= self.xlim[0] and x_max <= self.xlim[1] and
                y_min >= self.ylim[0] and y_max <= self.ylim[1]):
                
                # Check for overlaps
                if not any(candidate.overlaps_with(existing) for existing in self.boxes):
                    # Add minimum spacing
                    has_min_spacing = True
                    for existing in self.boxes:
                        dist = self._min_distance_between_boxes(candidate, existing)
                        if dist < self.min_spacing:
                            has_min_spacing = False
                            break
                    
                    if has_min_spacing:
                        return candidate
            
            attempt += 1
        
        # If no good position found, return original (will show as overlapping)
        print(f"Warning: Could not find non-overlapping position for box '{new_box.label}'")
        return new_box
    
    def _min_distance_between_boxes(self, box1: SmartBox, box2: SmartBox) -> float:
        """Calculate minimum distance between two boxes."""
        x1_min, y1_min, x1_max, y1_max = box1.get_bounds()
        x2_min, y2_min, x2_max, y2_max = box2.get_bounds()
        
        # Calculate distances
        dx = max(0, max(x1_min - x2_max, x2_min - x1_max))
        dy = max(0, max(y1_min - y2_max, y2_min - y1_max))
        
        return np.sqrt(dx**2 + dy**2)
    
    def auto_adjust_font_sizes(self, fig, ax) -> None:
        """Automatically adjust font sizes to fit within boxes."""
        for box in self.boxes:
            if box.label:
                # Test current font size
                text = ax.text(box.x + box.width/2, box.y + box.height/2,
                             box.label, fontsize=box.fontsize,
                             ha='center', va='center', alpha=0)
                
                fig.canvas.draw()
                renderer = fig.canvas.get_renderer()
                
                # Get text bbox
                bbox = text.get_window_extent(renderer=renderer)
                bbox_data = bbox.transformed(ax.transData.inverted())
                
                text.remove()
                
                # Calculate if text fits
                text_width = bbox_data.width
                text_height = bbox_data.height
                
                box_inner_width = box.width * (1 - 2 * box.padding)
                box_inner_height = box.height * (1 - 2 * box.padding)
                
                # Adjust font size if needed
                if text_width > box_inner_width or text_height > box_inner_height:
                    width_ratio = box_inner_width / text_width if text_width > 0 else 1
                    height_ratio = box_inner_height / text_height if text_height > 0 else 1
                    scale_factor = min(width_ratio, height_ratio) * 0.9
                    box.fontsize = box.fontsize * scale_factor
    
    def optimize_layout(self) -> None:
        """Optimize the overall layout to minimize overlaps and maximize spacing."""
        # Simple optimization: try to spread boxes evenly
        if len(self.boxes) <= 1:
            return
        
        # Calculate center of mass
        cx = sum(box.x + box.width/2 for box in self.boxes) / len(self.boxes)
        cy = sum(box.y + box.height/2 for box in self.boxes) / len(self.boxes)
        
        # Apply slight repulsion from center
        for box in self.boxes:
            box_cx = box.x + box.width/2
            box_cy = box.y + box.height/2
            
            # Calculate direction away from center
            dx = box_cx - cx
            dy = box_cy - cy
            
            # Normalize
            dist = np.sqrt(dx**2 + dy**2)
            if dist > 0:
                dx /= dist
                dy /= dist
                
                # Apply small displacement
                displacement = 0.1
                new_x = box.x + dx * displacement
                new_y = box.y + dy * displacement
                
                # Check bounds
                if (new_x >= self.xlim[0] and new_x + box.width <= self.xlim[1] and
                    new_y >= self.ylim[0] and new_y + box.height <= self.ylim[1]):
                    box.x = new_x
                    box.y = new_y
    
    def render(self, title: str = "Smart Box Layout", 
               output_path: Optional[str] = None) -> Tuple[plt.Figure, plt.Axes]:
        """Render the layout with all boxes."""
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Auto-adjust font sizes
        self.auto_adjust_font_sizes(fig, ax)
        
        # Draw boxes
        for i, box in enumerate(self.boxes):
            # Check for overlaps
            has_overlap = any(
                box.overlaps_with(other) 
                for j, other in enumerate(self.boxes) 
                if i != j
            )
            
            # Draw box
            edge_color = 'red' if has_overlap else 'black'
            rect = Rectangle((box.x, box.y), box.width, box.height,
                           linewidth=2, edgecolor=edge_color, 
                           facecolor=box.color, alpha=0.7)
            ax.add_patch(rect)
            
            # Add label
            if box.label:
                ax.text(box.x + box.width/2, box.y + box.height/2,
                       box.label, ha='center', va='center',
                       fontsize=box.fontsize, weight='bold')
        
        # Set axis properties
        ax.set_xlim(self.xlim)
        ax.set_ylim(self.ylim)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title(title, fontsize=16)
        
        # Add statistics
        num_overlaps = sum(
            1 for i, box1 in enumerate(self.boxes)
            for j, box2 in enumerate(self.boxes)
            if i < j and box1.overlaps_with(box2)
        )
        
        stats_text = f"Boxes: {len(self.boxes)}\nOverlaps: {num_overlaps}"
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               verticalalignment='top', fontsize=10,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
        return fig, ax


def demo_smart_layout():
    """Demonstrate the smart box layout system."""
    # Create layout manager
    layout = SmartBoxLayout(figsize=(12, 8))
    
    # Add boxes - some would overlap in original positions
    layout.add_box(1, 1, 2, 1.5, "Model\nInitialization", color='lightcoral')
    layout.add_box(2, 2, 2, 1.5, "Data\nPreprocessing", color='lightblue')
    layout.add_box(4, 1, 2, 1.5, "Feature\nExtraction", color='lightgreen')
    layout.add_box(6, 2, 2, 1.5, "Model\nTraining", color='lightyellow')
    layout.add_box(3, 4, 3, 1, "Validation &\nTesting", color='lightpink')
    layout.add_box(7, 5, 2, 1.5, "Results\nAnalysis", color='lightgray')
    
    # Add a box with long text
    layout.add_box(1, 6, 4, 1, "This is a very long text that will be automatically resized to fit within the box boundaries", color='lavender')
    
    # Optimize layout
    layout.optimize_layout()
    
    # Render
    fig, ax = layout.render(title="Smart Box Layout Demo", 
                           output_path="smart_box_layout_demo.png")
    
    # Print box positions
    print("Final box positions:")
    for i, box in enumerate(layout.boxes):
        print(f"Box {i+1} '{box.label[:20]}...': position=({box.x:.2f}, {box.y:.2f}), "
              f"size=({box.width:.2f}, {box.height:.2f}), fontsize={box.fontsize:.1f}")
    
    plt.show()


def create_workflow_diagram():
    """Create a workflow diagram with smart layout."""
    layout = SmartBoxLayout(figsize=(14, 10), xlim=(0, 14), ylim=(0, 10))
    
    # Define workflow stages
    stages = [
        ("Input Data", 2, 8, 2.5, 1.2, 'lightblue'),
        ("Preprocessing", 2, 6, 2.5, 1.2, 'lightgreen'),
        ("Feature Engineering", 5.5, 6, 2.5, 1.2, 'lightyellow'),
        ("Model Selection", 2, 4, 2.5, 1.2, 'lightcoral'),
        ("Training", 5.5, 4, 2.5, 1.2, 'lightpink'),
        ("Hyperparameter\nTuning", 9, 4, 2.5, 1.2, 'lavender'),
        ("Validation", 2, 2, 2.5, 1.2, 'peachpuff'),
        ("Testing", 5.5, 2, 2.5, 1.2, 'lightgray'),
        ("Deployment", 9, 2, 2.5, 1.2, 'palegreen'),
        ("Monitoring", 11.5, 8, 2, 1.2, 'wheat'),
    ]
    
    # Add boxes
    for label, x, y, w, h, color in stages:
        layout.add_box(x, y, w, h, label, color=color)
    
    # Render
    fig, ax = layout.render(title="Machine Learning Workflow", 
                           output_path="ml_workflow_smart.png")
    
    # Add arrows between stages (simplified)
    arrow_props = dict(arrowstyle='->', lw=2, color='gray', alpha=0.7)
    
    # Example connections
    ax.annotate('', xy=(2+1.25, 6), xytext=(2+1.25, 8-1.2),
                arrowprops=arrow_props)
    ax.annotate('', xy=(5.5+1.25, 6), xytext=(2+1.25, 6-1.2),
                arrowprops=arrow_props)
    
    plt.show()


if __name__ == "__main__":
    demo_smart_layout()
    create_workflow_diagram()