import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# ============= PARAMETERS - EDIT THESE =============
# Image dimensions
WIDTH = 2000
HEIGHT = 2000

# Visualization Mode
VORONOI_MODE = True  # True = Voronoi points/lines, False = Spiral streaks

# Event Horizon
EVENT_HORIZON_RADIUS = 0.2  # Radius of the black hole center (0-1 scale)
GLOW_FALLOFF = 8.0  # How quickly brightness falls off toward corners (higher = faster)
INTERIOR_FALLOFF = 4.0  # How quickly streaks fade toward center (higher = faster)

# Logarithmic Spiral (used when VORONOI_MODE = False)
SPIRAL_TIGHTNESS = 3.5  # How tightly the spiral winds (higher = tighter)
NUM_SPIRAL_ARMS = 12  # Number of main spiral arms

# Filament Interference Patterns (used when VORONOI_MODE = False)
FILAMENT_FREQUENCIES = [8, 16, 24, 32, 48]  # Sine wave frequencies for streaks
FILAMENT_WEIGHTS = [1.0, 0.8, 0.6, 0.4, 0.3]  # Relative intensity of each frequency
FILAMENT_PHASE_SHIFT = 0.5  # Phase shift for variety

# Voronoi Mode Settings (used when VORONOI_MODE = True)
VORONOI_POINTS = 500  # Number of points to generate
VORONOI_LINE_THICKNESS = 0.8  # Thickness of connecting lines
VORONOI_POINT_SIZE = 1.5  # Size of the points themselves

# Noise/Variation
NOISE_INTENSITY = 0.15  # Amount of random variation in glow (0 = none, 0.3 = high)
NOISE_SCALE = 5.0  # Scale of noise patterns (higher = larger blotches)

# Brightness and Contrast
BRIGHTNESS_BOOST = 1.0  # Overall brightness multiplier
CONTRAST_POWER = 1.0  # Power curve for contrast (higher = more dramatic)
INNER_GLOW_INTENSITY = 0  # How bright the inner rim glows

# Colormap Selection
# Options: 'inferno', 'magma', 'plasma', 'hot', 'afmhot', or 'custom'
COLORMAP = 'inferno'

# Custom colormap colors (used if COLORMAP = 'custom')
CUSTOM_COLORS = ['#000000', '#0a0015', '#1a0a2a', '#2a1540',
                 '#4a2060', '#6a3080', '#8a4090', '#aa60a0',
                 '#ca80b0', '#eaaac0', '#ffd0d0', '#ffffff']

# Stars
NUM_STARS = 800
STAR_MIN_BRIGHTNESS = 0.2
STAR_MAX_BRIGHTNESS = 1.0

# Output
OUTPUT_FILENAME = 'blackhole_vortex_10.png'
DPI = 150


# ===================================================

def create_blackhole_vortex():
    """
    Create a black hole vortex using logarithmic spirals and interference patterns
    or Voronoi-style point connections
    """
    # Create coordinate grids centered at image center
    x = np.linspace(-1, 1, WIDTH)
    y = np.linspace(-1, 1, HEIGHT)
    X, Y = np.meshgrid(x, y)

    # Convert to polar coordinates
    R = np.sqrt(X ** 2 + Y ** 2)
    theta = np.arctan2(Y, X)

    # Avoid division by zero
    R = np.maximum(R, 1e-10)

    # Initialize intensity array
    intensity = np.zeros_like(R)

    if VORONOI_MODE:
        # ===== VORONOI MODE: Points connected by lines =====
        print("Generating Voronoi-style pattern...")

        # Generate random points in polar coordinates
        # More points near the event horizon for better visual
        point_radii = np.random.beta(2, 5, VORONOI_POINTS) * 0.8 + EVENT_HORIZON_RADIUS
        point_angles = np.random.uniform(0, 2 * np.pi, VORONOI_POINTS)

        # Convert to Cartesian
        point_x = point_radii * np.cos(point_angles)
        point_y = point_radii * np.sin(point_angles)

        # Create distance field to nearest point
        for i in range(VORONOI_POINTS):
            dist = np.sqrt((X - point_x[i]) ** 2 + (Y - point_y[i]) ** 2)

            # Add points as bright spots
            intensity += np.exp(-(dist ** 2) / (VORONOI_POINT_SIZE * 0.001))

            # Connect to nearby points with lines
            for j in range(i + 1, min(i + 6, VORONOI_POINTS)):  # Connect to next 5 points
                dx = point_x[j] - point_x[i]
                dy = point_y[j] - point_y[i]
                line_dist = np.sqrt(dx ** 2 + dy ** 2)

                # Skip if points are too far apart
                if line_dist > 0.3:
                    continue

                # Calculate distance to line segment
                t = np.clip(((X - point_x[i]) * dx + (Y - point_y[i]) * dy) / (line_dist ** 2), 0, 1)
                line_point_x = point_x[i] + t * dx
                line_point_y = point_y[i] + t * dy
                dist_to_line = np.sqrt((X - line_point_x) ** 2 + (Y - line_point_y) ** 2)

                # Add line with falloff
                intensity += np.exp(-(dist_to_line ** 2) / (VORONOI_LINE_THICKNESS * 0.0001))

    else:
        # ===== SPIRAL MODE: Logarithmic spiral with filaments =====
        # Create logarithmic spiral coordinate
        theta_spiral = theta - SPIRAL_TIGHTNESS * np.log(R)

        # Create filaments using interference patterns
        for freq, weight in zip(FILAMENT_FREQUENCIES, FILAMENT_WEIGHTS):
            # Main spiral pattern
            pattern = np.sin(freq * theta_spiral + FILAMENT_PHASE_SHIFT * freq)
            # Add some radial modulation for variety
            radial_mod = np.sin(freq * np.log(R + 0.1) * 0.5)
            combined = pattern * (1 + 0.3 * radial_mod)
            intensity += weight * combined

    # Normalize to 0-1
    intensity = (intensity - intensity.min()) / (intensity.max() - intensity.min())

    # Add noise variation to create organic glow variations
    if NOISE_INTENSITY > 0:
        # Create multi-scale noise
        noise_x = X * NOISE_SCALE
        noise_y = Y * NOISE_SCALE
        noise = np.sin(noise_x * 3.7) * np.cos(noise_y * 4.3)
        noise += 0.5 * np.sin(noise_x * 7.1) * np.cos(noise_y * 8.9)
        noise += 0.25 * np.sin(noise_x * 13.3) * np.cos(noise_y * 17.7)
        noise = (noise - noise.min()) / (noise.max() - noise.min())
        noise = (noise - 0.5) * 2  # Scale to -1 to 1

        # Apply noise as multiplicative factor
        noise_factor = 1 + noise * NOISE_INTENSITY
        intensity = intensity * noise_factor
        intensity = np.clip(intensity, 0, 1)

    # Create separate falloff functions for interior and exterior

    # INTERIOR FALLOFF: Streaks fade as they approach the center
    # Use a smooth power function based on radius
    interior_fade = (R / EVENT_HORIZON_RADIUS) ** INTERIOR_FALLOFF
    interior_fade = np.clip(interior_fade, 0, 1)

    # EXTERIOR FALLOFF: Brightness decreases toward corners
    # Exponential falloff from event horizon outward
    distance_from_horizon = R - EVENT_HORIZON_RADIUS
    exterior_fade = np.exp(-GLOW_FALLOFF * np.maximum(distance_from_horizon, 0))

    # Combine interior and exterior fading
    # Inside the event horizon: use interior fade
    # Outside the event horizon: use exterior fade
    combined_fade = np.where(R < EVENT_HORIZON_RADIUS, interior_fade, exterior_fade)

    # Inner rim super-glow (right at the event horizon edge)
    inner_rim = np.exp(-50 * (R - EVENT_HORIZON_RADIUS) ** 2) * INNER_GLOW_INTENSITY

    # Combine filaments with fading and glow
    intensity = intensity * combined_fade + inner_rim

    # Apply brightness boost
    intensity = intensity * BRIGHTNESS_BOOST

    # Normalize again
    intensity = np.clip(intensity, 0, 1)

    # Apply contrast curve
    intensity = intensity ** CONTRAST_POWER

    # Final normalization
    intensity = (intensity - intensity.min()) / (intensity.max() - intensity.min() + 1e-10)

    # Select colormap
    if COLORMAP == 'custom':
        cmap = LinearSegmentedColormap.from_list('custom_blackhole', CUSTOM_COLORS, N=256)
    else:
        cmap = plt.get_cmap(COLORMAP)

    # Generate stars (avoiding center region)
    star_x = []
    star_y = []
    star_brightness = []

    for _ in range(NUM_STARS):
        sx = np.random.uniform(0, WIDTH)
        sy = np.random.uniform(0, HEIGHT)
        # Convert to normalized coordinates to check distance from center
        nx = (sx / WIDTH) * 2 - 1
        ny = (sy / HEIGHT) * 2 - 1
        dist = np.sqrt(nx ** 2 + ny ** 2)
        # Only add stars outside a certain radius
        if dist > EVENT_HORIZON_RADIUS + 0.3:
            star_x.append(sx)
            star_y.append(sy)
            star_brightness.append(np.random.uniform(STAR_MIN_BRIGHTNESS, STAR_MAX_BRIGHTNESS))

    return intensity, cmap, (star_x, star_y, star_brightness)


# Generate the image
mode_text = "Voronoi points/lines" if VORONOI_MODE else "logarithmic spirals"
print(f"Generating black hole vortex with {mode_text}...")
print("This may take a moment for high-resolution images...")
intensity, cmap, stars = create_blackhole_vortex()

# Create figure
fig, ax = plt.subplots(figsize=(12, 12), facecolor='black')
ax.set_facecolor('black')

# Display the vortex
im = ax.imshow(intensity, cmap=cmap, interpolation='bilinear')

# Add stars
star_x, star_y, star_brightness = stars
if len(star_x) > 0:
    star_sizes = np.array(star_brightness) * 2
    ax.scatter(star_x, star_y, s=star_sizes, c='white',
               alpha=np.array(star_brightness) * 0.6, marker='*')

# Remove axes
ax.axis('off')
plt.tight_layout(pad=0)

# Save the image
plt.savefig(OUTPUT_FILENAME, dpi=DPI, facecolor='black', bbox_inches='tight', pad_inches=0)
print(f"Image saved as '{OUTPUT_FILENAME}'")
print(f"Mode: {mode_text}")
print(f"Using colormap: {COLORMAP}")

plt.show()