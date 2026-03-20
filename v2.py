import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# ============= PARAMETERS - EDIT THESE =============
# Image dimensions
WIDTH = 1000
HEIGHT = 1000

# Vortex parameters
NUM_SPIRALS = 14  # Number of spiral arms
SPIRAL_TIGHTNESS = 6  # How tightly the spirals wind
EVENT_HORIZON_RADIUS = 0.4  # Radius of the black hole center circle (0-1 scale)
FALLOFF_WIDTH = 1  # Width of the fade zone around event horizon (larger = smoother)
OUTER_FADE = 1.5  # How much brightness decreases toward center

# Visual effects
NOISE_AMOUNT = 0.5  # Amount of texture/turbulence
CONTRAST_POWER = 2  # Higher = more contrast (try 1.2-2.0)

# Stars
NUM_STARS = 500
STAR_MIN_BRIGHTNESS = 0.3
STAR_MAX_BRIGHTNESS = 1.0

# Colormap (gradient from black to bright)
COLORMAP_COLORS = ['#000000', '#0a0a1a', '#1a1a3a', '#2a2a5a',
                   '#3a4a7a', '#4a6a9a', '#6a8aba', '#8aaacc',
                   '#aaccee', '#d0e0f0', '#ffffff']

# Output
OUTPUT_FILENAME = 'blackhole_vortex_5.png'
DPI = 150


# ===================================================

def create_blackhole_vortex():
    """
    Create a black hole vortex image with swirling effect and depth
    """
    # Create coordinate grids
    x = np.linspace(-2, 2, WIDTH)
    y = np.linspace(-2, 2, HEIGHT)
    X, Y = np.meshgrid(x, y)

    # Convert to polar coordinates
    R = np.sqrt(X ** 2 + Y ** 2)
    theta = np.arctan2(Y, X)

    # Create the spiral/vortex effect
    spiral = np.sin(NUM_SPIRALS * theta - SPIRAL_TIGHTNESS * np.log(R + 0.1))

    # Create radial falloff (brightness decreases toward center)
    radial_gradient = np.exp(-R * OUTER_FADE)

    # Combine spiral with radial gradient
    vortex = spiral * radial_gradient

    # Add some turbulence/noise for texture
    noise = np.random.randn(HEIGHT, WIDTH) * NOISE_AMOUNT
    vortex += noise

    # Create smooth circular mask for event horizon
    # This creates a smooth falloff from EVENT_HORIZON_RADIUS to EVENT_HORIZON_RADIUS + FALLOFF_WIDTH
    fade_start = EVENT_HORIZON_RADIUS
    fade_end = EVENT_HORIZON_RADIUS + FALLOFF_WIDTH

    # Smooth transition: 1 outside fade_end, 0 inside fade_start, smooth in between
    mask = np.clip((R - fade_start) / (fade_end - fade_start), 0, 1)
    mask = mask ** 2  # Quadratic falloff for smoother transition

    # Apply mask to create the black hole center
    vortex = vortex * mask

    # Normalize to 0-1 range
    vortex = (vortex - vortex.min()) / (vortex.max() - vortex.min())

    # Apply power curve for more dramatic contrast
    vortex = vortex ** CONTRAST_POWER

    # Create custom colormap
    n_bins = 256
    cmap = LinearSegmentedColormap.from_list('blackhole', COLORMAP_COLORS, N=n_bins)

    # Add stars (avoid the center region)
    star_x = np.random.uniform(0, WIDTH, NUM_STARS)
    star_y = np.random.uniform(0, HEIGHT, NUM_STARS)
    star_brightness = np.random.uniform(STAR_MIN_BRIGHTNESS, STAR_MAX_BRIGHTNESS, NUM_STARS)

    return vortex, cmap, (star_x, star_y, star_brightness)


# Generate the image
print("Generating black hole vortex...")
vortex, cmap, stars = create_blackhole_vortex()

# Create figure
fig, ax = plt.subplots(figsize=(12, 12), facecolor='black')
ax.set_facecolor('black')

# Display the vortex
im = ax.imshow(vortex, cmap=cmap, interpolation='bilinear')

# Add stars
star_x, star_y, star_brightness = stars
star_sizes = star_brightness * 3
ax.scatter(star_x, star_y, s=star_sizes, c='white', alpha=star_brightness * 0.8, marker='*')

# Remove axes
ax.axis('off')
plt.tight_layout(pad=0)

# Save the image
plt.savefig(OUTPUT_FILENAME, dpi=DPI, facecolor='black', bbox_inches='tight', pad_inches=0)
print(f"Image saved as '{OUTPUT_FILENAME}'")

plt.show()