import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# ============= PARAMETERS - EDIT THESE =============
# Image dimensions
WIDTH = 2000
HEIGHT = 2000

# Event Horizon
EVENT_HORIZON_RADIUS = 0.2  # Radius of the black hole center (0-1 scale)
GLOW_FALLOFF = 3.0  # How quickly brightness falls off toward corners (higher = faster)
INTERIOR_FALLOFF = 4.0  # How quickly streaks fade toward center (higher = faster)

# Logarithmic Spiral
SPIRAL_TIGHTNESS = 1.5  # How tightly the spiral winds (higher = tighter)
NUM_SPIRAL_ARMS = 12  # Number of main spiral arms

# Filament Interference Patterns
FILAMENT_FREQUENCIES = [8, 16, 24, 32, 48]  # Sine wave frequencies for streaks
FILAMENT_WEIGHTS = [1.0, 0.8, 0.6, 0.4, 0.3]  # Relative intensity of each frequency
FILAMENT_PHASE_SHIFT = 0.5  # Phase shift for variety

# Brightness and Contrast
BRIGHTNESS_BOOST = 1  # Overall brightness multiplier
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
OUTPUT_FILENAME = 'blackhole_vortex_8.png'
DPI = 150


# ===================================================

def create_blackhole_vortex():
    """
    Create a black hole vortex using logarithmic spirals and interference patterns
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

    # Create logarithmic spiral coordinate
    # This creates the twisted, swirling effect
    theta_spiral = theta - SPIRAL_TIGHTNESS * np.log(R)

    # Initialize intensity array
    intensity = np.zeros_like(R)

    # Create filaments using interference patterns
    # Combine multiple sine waves at different frequencies
    for freq, weight in zip(FILAMENT_FREQUENCIES, FILAMENT_WEIGHTS):
        # Main spiral pattern
        pattern = np.sin(freq * theta_spiral + FILAMENT_PHASE_SHIFT * freq)
        # Add some radial modulation for variety
        radial_mod = np.sin(freq * np.log(R + 0.1) * 0.5)
        combined = pattern * (1 + 0.3 * radial_mod)
        intensity += weight * combined

    # Normalize to 0-1
    intensity = (intensity - intensity.min()) / (intensity.max() - intensity.min())

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
print("Generating black hole vortex with logarithmic spirals...")
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
print(f"Using colormap: {COLORMAP}")

plt.show()