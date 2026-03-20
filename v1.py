import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def create_blackhole_vortex(width=1000, height=1000, num_spirals=8):
    """
    Create a black hole vortex image with swirling effect and depth
    """
    # Create coordinate grids
    x = np.linspace(-2, 2, width)
    y = np.linspace(-2, 2, height)
    X, Y = np.meshgrid(x, y)

    # Convert to polar coordinates
    R = np.sqrt(X ** 2 + Y ** 2)
    theta = np.arctan2(Y, X)

    # Create the spiral/vortex effect
    spiral = np.sin(num_spirals * theta - 5 * np.log(R + 0.1))

    # Create radial falloff (brightness decreases toward center)
    radial_gradient = np.exp(-R * 1.5)

    # Combine spiral with radial gradient
    vortex = spiral * radial_gradient

    # Add some turbulence/noise for texture
    noise = np.random.randn(height, width) * 0.05
    vortex += noise

    # Create the black hole center (strong falloff)
    center_mask = np.exp(-R ** 2 * 10)
    vortex = vortex * (1 - center_mask * 0.95)

    # Normalize to 0-1 range
    vortex = (vortex - vortex.min()) / (vortex.max() - vortex.min())

    # Apply power curve for more dramatic contrast
    vortex = vortex ** 2

    # Create custom colormap (black to blue/cyan to white)
    colors = ['#000000', '#0a0a1a', '#1a1a3a', '#2a2a5a',
              '#3a4a7a', '#4a6a9a', '#6a8aba', '#8aaacc',
              '#aaccee', '#d0e0f0', '#ffffff']
    n_bins = 256
    cmap = LinearSegmentedColormap.from_list('blackhole', colors, N=n_bins)

    # Add stars
    num_stars = 500
    star_x = np.random.uniform(0, width, num_stars)
    star_y = np.random.uniform(0, height, num_stars)
    star_brightness = np.random.uniform(0.3, 1.0, num_stars)

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
plt.savefig('blackhole_vortex_26.png', dpi=150, facecolor='black', bbox_inches='tight', pad_inches=0)
print("Image saved as 'blackhole_vortex_1.png'")

plt.show()