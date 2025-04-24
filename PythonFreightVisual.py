import pygame
import asyncio
import platform
import numpy as np

# Initialize Pygame
pygame.init()

# Screen dimensions
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Improved Curb Freight Visualizer")

# Colors
WHITE    = (255, 255, 255)
BLACK    = (0, 0, 0)
GREEN    = (0, 255, 0)
RED      = (255, 0, 0)
BLUE     = (0, 0, 255)
GRAY     = (150, 150, 150)
DARK_GRAY= (100, 100, 100)
YELLOW   = (255, 255, 0)  # For highlighting

# Isometric projection constants
K = 30  # Increased from 20 to make everything larger
OFFSET_X = 400
OFFSET_Y = 250  # Adjusted to better center the view with larger scaling

# Frame rate
FPS = 60

# Controls info
CONTROLS_INFO = [
    "Controls:",
    "R - Cycle through all possible orientations",
    "F - Toggle between flat and upright positions",
    "SPACE - Change stacking height",
    "A - Auto-arrange with optimized loading",
    "C - Clear all placements",
    "I - Print loading metrics in console"
]

def world_to_screen_3d(x, y, z):
    """Convert 3D world coordinates to 2D screen coordinates with isometric projection"""
    # Calculate base projection
    sx = K * (x - y) + OFFSET_X
    sy = (K / 2) * (x + y) - K * z + OFFSET_Y
    
    # Apply a slight offset to better center the truck bed with the increased zoom
    sx -= K * 1.5
    sy += K * 0.5
    
    return int(sx), int(sy)

def screen_to_world(mx, my):
    """Convert 2D screen coordinates to 3D world coordinates (x,y plane only)"""
    # Adjust the screen coordinates to account for the offsets in world_to_screen_3d
    sx = mx - OFFSET_X + K * 1.5
    sy = my - OFFSET_Y - K * 0.5
    
    # Apply the inverse isometric transformation
    x = (sx + 2 * sy) / (2 * K)
    y = (2 * sy - sx) / (2 * K)
    return x, y

def point_in_polygon(point, poly):
    """Determine if the 2D point is inside the polygon (list of points)"""
    x, y = point
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > y) != (yj > y)) and \
             (x < (xj - xi) * (y - yi) / (yj - yi + 1e-9) + xi):
            inside = not inside
        j = i
    return inside

class Curb:
    def __init__(self, l, w, h, index):
        self.l = l
        self.w = w
        self.h = h
        self.x = 0
        self.y = 0
        self.z = 0            # stacking height
        self.orientation = 0  # 0-5 representing the 6 possible orientations
        self.placed = False
        self.index = index
        self.supported = True  # Flag to track if curb is properly supported when stacked
        # Assign a unique color based on index
        self.color = [
            (0, 200, 80),     # Green
            (200, 120, 0),    # Orange
            (80, 80, 200),    # Blue
            (200, 0, 80),     # Pink
            (160, 160, 0),    # Yellow
            (120, 0, 160),    # Purple
            (0, 160, 160),    # Teal
            (200, 60, 60),    # Red
        ][index % 8]

    def get_dimensions(self):
        """
        Return dimensions based on orientation (6 possible orientations)
        0: l-w-h (default, flat)
        1: w-l-h (rotated 90 degrees horizontally)
        2: l-h-w (standing on long edge)
        3: h-l-w (standing on long edge, rotated 90)
        4: w-h-l (standing on short edge)
        5: h-w-l (standing on short edge, rotated 90)
        """
        orientations = [
            (self.l, self.w, self.h),  # Flat, default
            (self.w, self.l, self.h),  # Flat, rotated 90°
            (self.l, self.h, self.w),  # Upright on long edge
            (self.h, self.l, self.w),  # Upright on long edge, rotated
            (self.w, self.h, self.l),  # Upright on short edge
            (self.h, self.w, self.l),  # Upright on short edge, rotated
        ]
        
        return orientations[self.orientation]

    def get_rect(self):
        """Returns a 3D bounding box: [x, y, z, dx, dy, dz]"""
        dx, dy, dz = self.get_dimensions()
        return [self.x, self.y, self.z, dx, dy, dz]

    def get_corners(self):
        """Get the four corners of the curb in world coordinates"""
        dx, dy, _ = self.get_dimensions()
        return [
            (self.x, self.y),
            (self.x + dx, self.y),
            (self.x + dx, self.y + dy),
            (self.x, self.y + dy)
        ]

    def get_top_polygon(self):
        """
        Compute the projected polygon (list of 2D points) of the top face.
        Returns the coordinates for the visible surface as drawn on screen.
        """
        dx, dy, dz = self.get_dimensions()
        base_level = self.z
        top_h = base_level + dz  # top surface
        
        # Use the same inset factor as in the draw method
        inset_factor = 0.05 if self.orientation >= 2 else 0.1
        
        pts_3d = [
            (self.x + inset_factor * dx, self.y + inset_factor * dy, top_h),
            (self.x + (1-inset_factor) * dx, self.y + inset_factor * dy, top_h),
            (self.x + (1-inset_factor) * dx, self.y + (1-inset_factor) * dy, top_h),
            (self.x + inset_factor * dx, self.y + (1-inset_factor) * dy, top_h)
        ]
        return [world_to_screen_3d(*p) for p in pts_3d]

    def draw(self, screen, highlight_color=None):
        """Draw the curb in 3D with isometric projection"""
        # Use the curb's own color unless a highlight color is provided
        color = highlight_color if highlight_color else self.color
        
        # Get the dimensions based on current orientation
        dx, dy, dz = self.get_dimensions()
        base_level = self.z
        
        # Define inset factor based on orientation
        inset_factor = 0.05 if self.orientation >= 2 else 0.1
        
        # The base height is positioned at either 20% or 30% of the curb's height
        # depending on whether it's standing upright or not
        base_ratio = 0.2 if self.orientation >= 2 else 0.3
        base_h = base_level + dz * base_ratio
        top_h = base_level + dz
        
        # Define 3D points for the base and top surfaces
        base_points_3d = [
            (self.x,         self.y,         base_level),
            (self.x + dx,    self.y,         base_level),
            (self.x + dx,    self.y + dy,    base_level),
            (self.x,         self.y + dy,    base_level),
            (self.x,         self.y,         base_h),
            (self.x + dx,    self.y,         base_h),
            (self.x + dx,    self.y + dy,    base_h),
            (self.x,         self.y + dy,    base_h)
        ]
        
        # Creating an inset for the top portion to make it visually distinct
        top_points_3d = [
            (self.x + inset_factor * dx, self.y + inset_factor * dy, base_h),
            (self.x + (1-inset_factor) * dx, self.y + inset_factor * dy, base_h),
            (self.x + (1-inset_factor) * dx, self.y + (1-inset_factor) * dy, base_h),
            (self.x + inset_factor * dx, self.y + (1-inset_factor) * dy, base_h),
            (self.x + inset_factor * dx, self.y + inset_factor * dy, top_h),
            (self.x + (1-inset_factor) * dx, self.y + inset_factor * dy, top_h),
            (self.x + (1-inset_factor) * dx, self.y + (1-inset_factor) * dy, top_h),
            (self.x + inset_factor * dx, self.y + (1-inset_factor) * dy, top_h)
        ]

        # Convert 3D points to 2D screen coordinates
        base_points = [world_to_screen_3d(*p) for p in base_points_3d]
        top_points = [world_to_screen_3d(*p) for p in top_points_3d]

        # Draw the lower part of the curb with the curb's base color
        base_color = tuple(max(c // 1.5, 0) for c in self.color)  # Darker shade of the curb's color
        pygame.draw.polygon(screen, base_color, [base_points[4], base_points[5], base_points[6], base_points[7]])
        pygame.draw.polygon(screen, base_color, [base_points[0], base_points[4], base_points[7], base_points[3]])
        pygame.draw.polygon(screen, base_color, [base_points[0], base_points[1], base_points[5], base_points[4]])

        # Draw the top (upper) part of the curb
        pygame.draw.polygon(screen, color, [top_points[4], top_points[5], top_points[6], top_points[7]])
        shade = tuple(max(c // 2, 0) for c in color)  # Ensure we don't get negative values
        pygame.draw.polygon(screen, shade, [top_points[0], top_points[4], top_points[7], top_points[3]])
        pygame.draw.polygon(screen, shade, [top_points[0], top_points[1], top_points[5], top_points[4]])

        # Draw outlines for clarity
        for pts in [base_points, top_points]:
            for i in range(4):
                pygame.draw.line(screen, BLACK, pts[i], pts[(i + 1) % 4], 1)
                pygame.draw.line(screen, BLACK, pts[i + 4], pts[(i + 1) % 4 + 4], 1)
                pygame.draw.line(screen, BLACK, pts[i], pts[i + 4], 1)

        # Draw orientation indicator on top face
        center_x = (top_points[4][0] + top_points[6][0]) // 2
        center_y = (top_points[4][1] + top_points[6][1]) // 2
        
        # Display orientation type ('F' for flat, 'U' for upright)
        orientation_text = 'F' if self.orientation < 2 else 'U'
        # Also display the specific orientation number for debugging
        orientation_text += str(self.orientation)
        
        # Render text in the center of the top face
        label = font.render(orientation_text, True, BLACK)
        screen.blit(label, (center_x - label.get_width() // 2, center_y - label.get_height() // 2))

class TruckBed:
    def __init__(self, L, W):
        self.L = L  # Length
        self.W = W  # Width

    def draw(self, screen):
        """Draw the truck bed with grid"""
        # Draw the truck bed floor
        points = [(0, 0, 0), (self.L, 0, 0), (self.L, self.W, 0), (0, self.W, 0)]
        screen_points = [world_to_screen_3d(p[0], p[1], p[2]) for p in points]
        
        # Fill with a light color
        pygame.draw.polygon(screen, (200, 220, 255), screen_points)
        
        # Draw borders
        pygame.draw.lines(screen, BLACK, True, screen_points, 2)
        
        # Draw grid lines
        for i in range(0, int(self.L) + 1):
            start = world_to_screen_3d(i, 0, 0)
            end = world_to_screen_3d(i, self.W, 0)
            pygame.draw.line(screen, DARK_GRAY, start, end, 1)
        
        for j in range(0, int(self.W) + 1):
            start = world_to_screen_3d(0, j, 0)
            end = world_to_screen_3d(self.L, j, 0)
            pygame.draw.line(screen, DARK_GRAY, start, end, 1)
        
        # Add dimension markers
        length_text = font.render(f"{self.L}m", True, BLACK)
        width_text = font.render(f"{self.W}m", True, BLACK)
        
        # Position the texts
        mid_length = world_to_screen_3d(self.L/2, 0, 0)
        mid_width = world_to_screen_3d(0, self.W/2, 0)
        
        screen.blit(length_text, (mid_length[0] - length_text.get_width()//2, mid_length[1] + 10))
        screen.blit(width_text, (mid_width[0] - 40, mid_width[1] - width_text.get_height()//2))

# ---------------------------
# Collision & Placement Functions
# ---------------------------
def overlaps_3d(curb1, curb2):
    """Check if two curbs overlap in 3D space"""
    r1 = curb1.get_rect()
    r2 = curb2.get_rect()
    x1, y1, z1, dx1, dy1, dz1 = r1
    x2, y2, z2, dx2, dy2, dz2 = r2

    # Add a small epsilon to avoid floating point precision issues
    epsilon = 0.001
    
    overlap_x = (x1 + epsilon < x2 + dx2) and (x1 + dx1 - epsilon > x2)
    overlap_y = (y1 + epsilon < y2 + dy2) and (y1 + dy1 - epsilon > y2)
    overlap_z = (z1 + epsilon < z2 + dz2) and (z1 + dz1 - epsilon > z2)
    return overlap_x and overlap_y and overlap_z

def is_within_truck(curb, truck_bed):
    """Check if the curb is completely within the truck bed boundaries"""
    r = curb.get_rect()
    # Add a small epsilon for floating point comparisons
    epsilon = 0.001
    # Only check X and Y bounds (z is determined by stacking)
    return (0 <= r[0] and r[0] + r[3] <= truck_bed.L + epsilon and
            0 <= r[1] and r[1] + r[4] <= truck_bed.W + epsilon)

def is_properly_supported(curb, placed_curbs):
    """
    Check if a curb is properly supported:
    - If at ground level (z=0), it's always supported
    - If stacked, it must have sufficient support underneath
    """
    if curb.z == 0:
        return True
    
    # Get the current curb's footprint
    curb_x, curb_y, curb_z, curb_dx, curb_dy, curb_dz = curb.get_rect()
    
    # Support requirements based on orientation
    if curb.orientation >= 2:  # Upright orientation needs more support
        required_support = 0.75  # 75% for upright positions
    else:  # Flat orientation
        required_support = 0.5   # 50% for flat positions
    
    curb_area = curb_dx * curb_dy
    if curb_area < 0.01:  # Prevent division by zero and handle very thin dimensions
        return False
        
    supported_area = 0
    epsilon = 0.01  # Small tolerance for floating point comparisons
    
    # Check for support from other curbs
    for other in placed_curbs:
        if other == curb:
            continue
            
        other_x, other_y, other_z, other_dx, other_dy, other_dz = other.get_rect()
        
        # Check if this curb is directly below our curb (with epsilon tolerance)
        if abs(other_z + other_dz - curb_z) < epsilon:
            # Calculate overlap area
            x_overlap = max(0, min(curb_x + curb_dx, other_x + other_dx) - max(curb_x, other_x))
            y_overlap = max(0, min(curb_y + curb_dy, other_y + other_dy) - max(curb_y, other_y))
            
            supported_area += x_overlap * y_overlap
    
    # A curb is supported if enough of its area is supported by curbs below it
    support_ratio = supported_area / curb_area
    return support_ratio >= required_support

def optimize_load(truck_bed, curbs):
    """
    Optimized loading algorithm that follows industry standards
    for cargo arrangement and weight distribution
    """
    # Reset all curbs
    for curb in curbs:
        curb.placed = False
        curb.z = 0
        curb.orientation = 0  # Start with flat orientation
    
    # Sort curbs by volume (largest first) for better weight distribution
    sorted_curbs = sorted(curbs, key=lambda c: c.l * c.w * c.h, reverse=True)
    
    # First pass: Place larger items on the bottom for stability
    placed_curbs = []
    
    # Try to place each curb
    for curb in sorted_curbs:
        best_position = None
        min_distance = float('inf')  # Distance from center (for balance)
        
        # Try all 6 possible orientations
        for orientation in range(6):
            curb.orientation = orientation
            dx, dy, dz = curb.get_dimensions()
            
            # Skip if this orientation doesn't fit on the truck bed at all
            if dx > truck_bed.L or dy > truck_bed.W:
                continue
            
            # Try different positions on the truck bed with a finer grid for better packing
            step = 0.5  # Use 0.5 unit steps for initial placement
            for y in np.arange(0, truck_bed.W - dy + 0.01, step):
                for x in np.arange(0, truck_bed.L - dx + 0.01, step):
                    curb.x = x
                    curb.y = y
                    curb.z = 0
                    
                    # Check if position is valid
                    if not any(overlaps_3d(curb, p) for p in placed_curbs) and is_within_truck(curb, truck_bed):
                        # Calculate distance from center (for balance)
                        center_x = truck_bed.L / 2
                        center_y = truck_bed.W / 2
                        distance = abs(x + dx/2 - center_x) + abs(y + dy/2 - center_y)
                        
                        # Update best position if this is better
                        if distance < min_distance:
                            min_distance = distance
                            best_position = (x, y, 0, orientation)
        
        # Place the curb in its best position if found
        if best_position:
            curb.x, curb.y, curb.z, curb.orientation = best_position
            curb.placed = True
            placed_curbs.append(curb)
        else:
            # If no position found on floor, try stacking
            stacked = False
            
            # Sort placed curbs by their top surface area (larger surface = better base)
            stack_bases = sorted(
                placed_curbs,
                key=lambda c: c.get_dimensions()[0] * c.get_dimensions()[1],
                reverse=True
            )
            
            # Try to stack on top of placed curbs
            for base_curb in stack_bases:
                base_x, base_y, base_z, base_dx, base_dy, base_dz = base_curb.get_rect()
                
                # Try different orientations for the stacked curb
                # First try flat orientations (more stable)
                for orientation in range(2):
                    curb.orientation = orientation
                    dx, dy, dz = curb.get_dimensions()
                    
                    # Skip if this orientation is too big for the truck
                    if dx > truck_bed.L or dy > truck_bed.W:
                        continue
                    
                    # Try positions on top of the base curb
                    curb.z = base_z + base_dz
                    
                    # Try centered position first (most stable)
                    x = base_x + (base_dx - dx) / 2
                    y = base_y + (base_dy - dy) / 2
                    
                    # Make sure it's within the truck bed boundaries
                    if not (0 <= x and x + dx <= truck_bed.L and 
                            0 <= y and y + dy <= truck_bed.W):
                        # If centered doesn't work, try other positions
                        # Try a grid of positions on the base curb
                        grid_step = 0.2
                        found_valid = False
                        for offset_y in np.arange(0, base_dy - dy + 0.01, grid_step):
                            for offset_x in np.arange(0, base_dx - dx + 0.01, grid_step):
                                x = base_x + offset_x
                                y = base_y + offset_y
                                
                                # Skip if it would extend outside the truck
                                if x + dx > truck_bed.L or y + dy > truck_bed.W:
                                    continue
                                    
                                curb.x = x
                                curb.y = y
                                
                                # Check validity and support
                                valid_position = (
                                    not any(overlaps_3d(curb, p) for p in [c for c in placed_curbs if c != base_curb]) and 
                                    is_within_truck(curb, truck_bed) and
                                    is_properly_supported(curb, placed_curbs)
                                )
                                
                                if valid_position:
                                    curb.placed = True
                                    placed_curbs.append(curb)
                                    stacked = True
                                    found_valid = True
                                    break
                            if found_valid:
                                break
                        if found_valid:
                            break
                    else:
                        # The centered position works, use it
                        curb.x = x
                        curb.y = y
                        
                        # Verify it doesn't overlap and is supported
                        valid_position = (
                            not any(overlaps_3d(curb, p) for p in [c for c in placed_curbs if c != base_curb]) and 
                            is_within_truck(curb, truck_bed) and
                            is_properly_supported(curb, placed_curbs)
                        )
                        
                        if valid_position:
                            curb.placed = True
                            placed_curbs.append(curb)
                            stacked = True
                            break
                
                if stacked:
                    break
    
    # Calculate loading metrics
    total_volume = sum(c.l * c.w * c.h for c in curbs)
    placed_volume = sum(c.l * c.w * c.h for c in curbs if c.placed)
    truck_volume = truck_bed.L * truck_bed.W * 5  # Assuming max height of 5
    
    # Update global loading info
    loading_info["placed_count"] = len(placed_curbs)
    loading_info["total_count"] = len(curbs)
    loading_info["space_efficiency"] = (placed_volume / truck_volume * 100) if truck_volume > 0 else 0
    
    # Calculate weight balance
    if placed_curbs:
        center_x = truck_bed.L / 2
        center_y = truck_bed.W / 2
        total_weight = sum(c.l * c.w * c.h for c in placed_curbs)  # Use volume as proxy for weight
        
        if total_weight > 0:
            weight_moments_x = sum((c.x + c.get_dimensions()[0]/2 - center_x) * (c.l * c.w * c.h) for c in placed_curbs)
            weight_moments_y = sum((c.y + c.get_dimensions()[1]/2 - center_y) * (c.l * c.w * c.h) for c in placed_curbs)
            imbalance = (abs(weight_moments_x) + abs(weight_moments_y)) / total_weight
            loading_info["weight_balance"] = 100 - min(100, imbalance * 20)  # Scale appropriately
        else:
            loading_info["weight_balance"] = 0
    else:
        loading_info["weight_balance"] = 0
    
    # Calculate stability score
    if placed_curbs:
        stable_count = sum(1 for c in placed_curbs if (c.z == 0 or is_properly_supported(c, placed_curbs)))
        loading_info["stability_score"] = (stable_count / len(placed_curbs)) * 100
    else:
        loading_info["stability_score"] = 0
    
    print(f"\nLoading complete! Placed {len(placed_curbs)}/{len(curbs)} curbs")
    print(f"Space efficiency: {loading_info['space_efficiency']:.1f}%")
    print(f"Weight balance: {loading_info['weight_balance']:.1f}%")
    print(f"Stability score: {loading_info['stability_score']:.1f}%\n")
    
    return len(placed_curbs) == len(curbs)  # Return True if all curbs were placed

# ---------------------------
# Global State and Setup
# ---------------------------
truck_bed = TruckBed(10, 5)
curbs = [
    Curb(2, 1, 0.5, 0),   # Standard curb
    Curb(3, 2, 0.7, 1),   # Wide curb
    Curb(1, 1, 0.3, 2),   # Small curb
    Curb(2, 2, 0.8, 3),   # Square curb
    Curb(4, 1, 0.6, 4),   # Long curb
    Curb(1.5, 1.5, 1.2, 5), # Tall curb
    Curb(2.5, 1.2, 0.9, 6), # Irregular curb
    Curb(3.5, 1.8, 0.4, 7)  # Extra wide curb
]
selected_curb = None
font = pygame.font.Font(None, 24)

# Additional information for display
loading_info = {
    "placed_count": 0,
    "total_count": len(curbs),
    "space_efficiency": 0.0,
    "weight_balance": 0.0,
    "stability_score": 0.0
}

def setup():
    """Initialize or reset the simulation"""
    global truck_bed, curbs, selected_curb, loading_info
    truck_bed = TruckBed(10, 5)
    curbs = [
        Curb(2, 1, 0.5, 0),   # Standard curb
        Curb(3, 2, 0.7, 1),   # Wide curb
        Curb(1, 1, 0.3, 2),   # Small curb
        Curb(2, 2, 0.8, 3),   # Square curb
        Curb(4, 1, 0.6, 4),   # Long curb
        Curb(1.5, 1.5, 1.2, 5), # Tall curb
        Curb(2.5, 1.2, 0.9, 6), # Irregular curb
        Curb(3.5, 1.8, 0.4, 7)  # Extra wide curb
    ]
    selected_curb = None
    loading_info = {
        "placed_count": 0,
        "total_count": len(curbs),
        "space_efficiency": 0.0,
        "weight_balance": 0.0,
        "stability_score": 0.0
    }

def draw_inventory():
    """Draw the inventory sidebar with available curbs"""
    # Draw sidebar background
    pygame.draw.rect(screen, (240, 240, 240), (5, 5, 140, 400))
    pygame.draw.rect(screen, BLACK, (5, 5, 140, 400), 2)
    
    # Draw title
    inventory_title = font.render("Available Curbs:", True, BLACK)
    screen.blit(inventory_title, (15, 15))
    
    # Draw each curb in inventory
    y_pos = 45
    for i, curb in enumerate(curbs):
        if not curb.placed and curb != selected_curb:
            # Draw color indicator
            pygame.draw.rect(screen, curb.color, (15, y_pos, 20, 20))
            pygame.draw.rect(screen, BLACK, (15, y_pos, 20, 20), 1)
            
            # Draw dimensions text
            text = font.render(f"{curb.l}×{curb.w}×{curb.h}", True, BLACK)
            screen.blit(text, (45, y_pos))
            
            y_pos += 30

def draw_stats():
    """Draw loading statistics on screen"""
    # Draw stats background
    stats_x = SCREEN_WIDTH - 190
    stats_y = 410
    pygame.draw.rect(screen, (240, 240, 240), (stats_x, stats_y, 180, 180))
    pygame.draw.rect(screen, BLACK, (stats_x, stats_y, 180, 180), 2)
    
    # Draw title
    stats_title = font.render("Loading Statistics:", True, BLACK)
    screen.blit(stats_title, (stats_x + 10, stats_y + 10))
    
    # Draw stats info
    y_offset = 40
    stats = [
        f"Curbs: {loading_info['placed_count']}/{loading_info['total_count']}",
        f"Space: {loading_info['space_efficiency']:.1f}%",
        f"Balance: {loading_info['weight_balance']:.1f}%",
        f"Stability: {loading_info['stability_score']:.1f}%"
    ]
    
    for stat in stats:
        text = font.render(stat, True, BLACK)
        screen.blit(text, (stats_x + 10, stats_y + y_offset))
        y_offset += 25
    
    # Draw optimality rating based on the metrics
    avg_score = (loading_info['space_efficiency'] + 
                loading_info['weight_balance'] + 
                loading_info['stability_score']) / 3
    
    rating = "Poor"
    if avg_score > 80:
        rating = "Excellent"
        color = GREEN
    elif avg_score > 60:
        rating = "Good"
        color = (100, 200, 0)
    elif avg_score > 40:
        rating = "Average"
        color = YELLOW
    else:
        color = RED
    
    screen.blit(font.render("Overall Rating:", True, BLACK), (stats_x + 10, stats_y + y_offset + 10))
    rating_text = font.render(rating, True, color)
    screen.blit(rating_text, (stats_x + 10, stats_y + y_offset + 35))

# ---------------------------
# Main Update Loop
# ---------------------------
def update_loop():
    """Main game loop: process events and update display"""
    global selected_curb
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            return True  # Signal to quit
            
        # Print loading metrics when 'I' is pressed (for info)
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_i:
            # Calculate current metrics
            placed_count = sum(1 for c in curbs if c.placed)
            total_volume = sum(c.l * c.w * c.h for c in curbs)
            placed_volume = sum(c.l * c.w * c.h for c in curbs if c.placed)
            truck_volume = truck_bed.L * truck_bed.W * 5
            
            print("\n--- LOADING METRICS ---")
            print(f"Items Placed: {placed_count}/{len(curbs)}")
            print(f"Space Efficiency: {(placed_volume / truck_volume * 100):.1f}%")
            print(f"Weight Distribution Balance: {loading_info['weight_balance']:.1f}%")
            print(f"Stability Score: {loading_info['stability_score']:.1f}%")
            print("----------------------\n")

        # Mouse down: select curb from inventory or truck bed
        elif event.type == pygame.MOUSEBUTTONDOWN:
            mx, my = pygame.mouse.get_pos()
            
            # Check if clicking in the inventory area
            if 10 < mx < 150 and 50 < my < 50 + len(curbs) * 30:
                # Calculate which inventory item was clicked
                inventory_y_pos = 50
                for curb in curbs:
                    if not curb.placed:
                        if inventory_y_pos <= my < inventory_y_pos + 30:
                            selected_curb = curb
                            break
                        inventory_y_pos += 30
            elif selected_curb is None:
                # Try to select a placed curb from the truck bed
                for curb in reversed(curbs):  # Check from top to bottom (visually)
                    if curb.placed:
                        poly = curb.get_top_polygon()
                        if point_in_polygon((mx, my), poly):
                            selected_curb = curb
                            curb.placed = False  # Remove from placed state to allow dragging
                            
                            # Update loading info when a curb is removed
                            loading_info["placed_count"] -= 1
                            
                            # Recalculate space efficiency
                            placed_volume = sum(c.l * c.w * c.h for c in curbs if c.placed)
                            truck_volume = truck_bed.L * truck_bed.W * 5
                            loading_info["space_efficiency"] = (placed_volume / truck_volume * 100) if truck_volume > 0 else 0
                            
                            break

        # Mouse up: try to place the selected curb
        elif event.type == pygame.MOUSEBUTTONUP:
            if selected_curb:
                placed_curbs = [c for c in curbs if c.placed and c != selected_curb]
                
                # IMPORTANT: Always check if curb is within truck boundaries regardless of height
                within_truck_bounds = is_within_truck(selected_curb, truck_bed)
                no_overlap = not any(overlaps_3d(selected_curb, p) for p in placed_curbs)
                properly_supported = (selected_curb.z == 0 or is_properly_supported(selected_curb, placed_curbs))
                
                valid_position = within_truck_bounds and no_overlap and properly_supported
                
                if valid_position:
                    selected_curb.placed = True
                    selected_curb.supported = properly_supported
                    
                    # Update loading info when a curb is placed
                    loading_info["placed_count"] += 1
                    
                    # Recalculate space efficiency
                    placed_volume = sum(c.l * c.w * c.h for c in curbs if c.placed)
                    truck_volume = truck_bed.L * truck_bed.W * 5
                    loading_info["space_efficiency"] = (placed_volume / truck_volume * 100) if truck_volume > 0 else 0
                    
                    # Recalculate weight balance if curbs are placed
                    placed_curbs = [c for c in curbs if c.placed]
                    if placed_curbs:
                        center_x = truck_bed.L / 2
                        center_y = truck_bed.W / 2
                        total_weight = sum(c.l * c.w * c.h for c in placed_curbs)
                        
                        if total_weight > 0:
                            weight_moments_x = sum((c.x + c.get_dimensions()[0]/2 - center_x) * (c.l * c.w * c.h) for c in placed_curbs)
                            weight_moments_y = sum((c.y + c.get_dimensions()[1]/2 - center_y) * (c.l * c.w * c.h) for c in placed_curbs)
                            imbalance = (abs(weight_moments_x) + abs(weight_moments_y)) / total_weight
                            loading_info["weight_balance"] = 100 - min(100, imbalance * 20)
                        else:
                            loading_info["weight_balance"] = 0
                    
                    # Recalculate stability score
                    if placed_curbs:
                        stable_count = sum(1 for c in placed_curbs if (c.z == 0 or is_properly_supported(c, placed_curbs)))
                        loading_info["stability_score"] = (stable_count / len(placed_curbs)) * 100
                    else:
                        loading_info["stability_score"] = 0
                else:
                    selected_curb.placed = False
                
                selected_curb = None

        # Key events for manipulating curbs and simulation
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_r and selected_curb:
                # Cycle through all 6 possible orientations (0-5)
                selected_curb.orientation = (selected_curb.orientation + 1) % 6
            
            # F key: toggle between flat (0-1) and upright (2-5) orientations
            elif event.key == pygame.K_f and selected_curb:
                if selected_curb.orientation < 2:  # Currently flat
                    selected_curb.orientation = 2  # Switch to upright
                else:  # Currently upright
                    selected_curb.orientation = 0  # Switch to flat
            
            # Space key: cycle stacking height
            elif event.key == pygame.K_SPACE and selected_curb:
                potential_heights = [0]  # Ground level is always an option
                
                # Find all possible stacking heights from placed curbs
                for c in curbs:
                    if c.placed and c != selected_curb:
                        _,_,cz,_,_,cdz = c.get_rect()
                        potential_heights.append(cz + cdz)
                
                # Sort and remove duplicates
                potential_heights = sorted(set(potential_heights))
                
                # Cycle to the next height
                try:
                    current_index = potential_heights.index(selected_curb.z)
                    selected_curb.z = potential_heights[(current_index + 1) % len(potential_heights)]
                except ValueError:
                    # If current height not in list, go to first height
                    selected_curb.z = potential_heights[0]
            
            # Auto-arrange all curbs optimally
            elif event.key == pygame.K_a:
                optimize_load(truck_bed, curbs)
            
            # Clear all placements
            elif event.key == pygame.K_c:
                for curb in curbs:
                    curb.placed = False
                    curb.z = 0
                    curb.orientation = 0
                    curb.supported = True
                
                # Reset loading info
                loading_info["placed_count"] = 0
                loading_info["space_efficiency"] = 0.0
                loading_info["weight_balance"] = 0.0
                loading_info["stability_score"] = 0.0
                
                selected_curb = None

    # Update position of selected curb based on mouse movement (snapping to grid)
    if selected_curb:
        mx, my = pygame.mouse.get_pos()
        world_x, world_y = screen_to_world(mx, my)
        
        # Snap to a finer grid (0.2 units)
        grid_size = 0.2
        selected_curb.x = round(world_x / grid_size) * grid_size
        selected_curb.y = round(world_y / grid_size) * grid_size
        
        # Get dimensions of the curb in current orientation
        dx, dy, _ = selected_curb.get_dimensions()
        
        # Constrain within truck boundaries
        selected_curb.x = max(0, min(selected_curb.x, truck_bed.L - dx))
        selected_curb.y = max(0, min(selected_curb.y, truck_bed.W - dy))

        # Check support status when stacked
        placed_curbs = [c for c in curbs if c.placed and c != selected_curb]
        if selected_curb.z > 0:
            selected_curb.supported = is_properly_supported(selected_curb, placed_curbs)
        else:
            selected_curb.supported = True

    # Draw scene
    screen.fill(WHITE)
    
    # Draw truck bed
    truck_bed.draw(screen)
    
    # Draw truck boundaries at elevated heights to show limits
    # This helps visualize where the truck boundaries are for stacked curbs
    for h in range(1, 6):  # Draw boundaries at heights 1-5
        # Get the corners of the truck bed at this height
        corners = [
            (0, 0, h),
            (truck_bed.L, 0, h),
            (truck_bed.L, truck_bed.W, h),
            (0, truck_bed.W, h)
        ]
        screen_corners = [world_to_screen_3d(*p) for p in corners]
        
        # Draw the borders as dashed lines
        for i in range(4):
            start_point = screen_corners[i]
            end_point = screen_corners[(i+1) % 4]
            
            # Draw dashed line
            dash_length = 10
            gap_length = 5
            dx = end_point[0] - start_point[0]
            dy = end_point[1] - start_point[1]
            distance = max(1, (dx**2 + dy**2)**0.5)
            
            # Normalize the direction
            dx, dy = dx/distance, dy/distance
            
            # How many segments will we need?
            segments = int(distance / (dash_length + gap_length))
            
            # Draw each dash
            for j in range(segments):
                start_dash = (
                    int(start_point[0] + (dash_length + gap_length) * j * dx),
                    int(start_point[1] + (dash_length + gap_length) * j * dy)
                )
                end_dash = (
                    int(start_point[0] + (dash_length + gap_length) * j * dx + dash_length * dx),
                    int(start_point[1] + (dash_length + gap_length) * j * dy + dash_length * dy)
                )
                pygame.draw.line(screen, (100, 100, 150, 128), start_dash, end_dash, 1)
    
    # Draw placed curbs (in reverse order so that ones in front are drawn last)
    # Sort by z + y + x for correct drawing order (higher z and y should be drawn later)
    sorted_curbs = sorted(
        [c for c in curbs if c.placed], 
        key=lambda c: (c.z, c.y + c.x)
    )
    
    for curb in sorted_curbs:
        curb.draw(screen)
    
    # Adjust layout to accommodate larger visualization
    draw_inventory()
    
    # Draw loading statistics
    draw_stats()
    
    # Draw controls info
    # Draw background for controls
    controls_x = SCREEN_WIDTH - 220
    controls_y = 10
    pygame.draw.rect(screen, (240, 240, 240), (controls_x, controls_y, 210, 170))
    pygame.draw.rect(screen, BLACK, (controls_x, controls_y, 210, 170), 2)
    
    # Draw the controls text
    for i, line in enumerate(CONTROLS_INFO):
        text = font.render(line, True, BLACK)
        screen.blit(text, (controls_x + 10, controls_y + 10 + i * 20))
    
    # Draw selected curb last (so it's on top)
    if selected_curb:
        placed_curbs = [c for c in curbs if c.placed and c != selected_curb]
        
        # Determine if the position is valid
        valid_position = (
            is_within_truck(selected_curb, truck_bed) and
            not any(overlaps_3d(selected_curb, p) for p in placed_curbs)
        )
        
        # Determine color based on validity and support
        if valid_position:
            if selected_curb.z == 0 or selected_curb.supported:
                color = GREEN  # Valid and supported
            else:
                color = RED  # Not properly supported when stacked
        else:
            color = RED  # Invalid position
            
        selected_curb.draw(screen, color)
    
    pygame.display.flip()
    return False  # Signal to continue

# ---------------------------
# Main Async Loop
# ---------------------------
async def main():
    """Main asynchronous game loop"""
    setup()
    clock = pygame.time.Clock()
    running = True
    
    while running:
        quit_signal = update_loop()
        if quit_signal:
            running = False
        
        # Use Pygame's clock instead of asyncio.sleep for more accurate timing
        clock.tick(FPS)
        
        # A short asyncio sleep to allow other async tasks to run
        await asyncio.sleep(0.001)

# Handle different platforms (browser vs desktop)
if platform.system() == "Emscripten":
    asyncio.ensure_future(main())
else:
    if __name__ == "__main__":
        asyncio.run(main())