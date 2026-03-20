import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.spatial import Voronoi

# ============= PARAMETERS - EDIT THESE =============
# Image dimensions
WIDTH = 2000
HEIGHT = 2000

# Visualization Mode
VORONOI_MODE = False  # True = Voronoi points/lines, False = Spiral streaks

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
VORONOI_RINGS = 40  # Number of concentric rings
VORONOI_POINTS_PER_RING = 60  # Points per ring
VORONOI_MAX_LINE_DIST = 0.08  # Maximum distance for connecting lines
VORONOI_LINE_WIDTH = 0.3  # Width of Voronoi lines
VORONOI_MANDALA_ROTATIONS = 80  # Number of mandala pattern rotations

# Noise/Variation
NOISE_INTENSITY = 0.35  # Amount of random variation in glow (0 = none, 0.3 = high)
NOISE_SCALE = 3.0  # Scale of noise patterns (higher = larger blotches)

# Brightness and Contrast
BRIGHTNESS_BOOST = 1.0  # Overall brightness multiplier
CONTRAST_POWER = 1.0  # Power curve for contrast (higher = more dramatic)
INNER_GLOW_INTENSITY = 0  # How bright the inner rim glows

# Colormap Selection
# Options: 'inferno', 'magma', 'plasma', 'hot', 'afmhot', or 'custom'
COLORMAP = 'afmhot'

# Custom colormap colors (used if COLORMAP = 'custom')
CUSTOM_COLORS = ['#000000', '#0a0015', '#1a0a2a', '#2a1540',
                 '#4a2060', '#6a3080', '#8a4090', '#aa60a0',
                 '#ca80b0', '#eaaac0', '#ffd0d0', '#ffffff']

# Stars
NUM_STARS = 800
STAR_MIN_BRIGHTNESS = 0.2
STAR_MAX_BRIGHTNESS = 1.0

# Output
OUTPUT_FILENAME = 'blackhole_vortex_24.png'
DPI = 600 if VORONOI_MODE else 150  # Higher DPI for Voronoi artwork mode


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
        # ===== VORONOI MODE: Radial rings with increasing chaos =====
        print("Generating Voronoi-style radial pattern...")

        # Generate radial ring points (symmetry → chaos)
        vor_points = []
        radius_max = 0.5

        for i in range(VORONOI_RINGS):
            r = (i + 1) / VORONOI_RINGS * radius_max
            n = VORONOI_POINTS_PER_RING
            fade = i / VORONOI_RINGS  # 0 at center, 1 at edge

            for j in range(n):
                theta = 2 * np.pi * j / n
                # Add increasing noise as we move outward
                angle_noise = fade * (np.pi / n) * np.random.randn()
                radial_noise = fade * (0.02 * np.random.randn())

                px = (r + radial_noise) * np.cos(theta + angle_noise)
                py = (r + radial_noise) * np.sin(theta + angle_noise)
                vor_points.append((px, py))

        vor_points = np.array(vor_points)

        # Generate mandala-style rotated pattern for complexity
        def generate_base_pattern_points(length=1.0, n_ticks=20, tick_size=0.05):
            pts = []
            x = np.linspace(-length / 2, length / 2, 200)
            y = np.zeros_like(x)
            pts.extend(list(zip(x, y)))
            tick_positions = np.linspace(-length / 2, length / 2, n_ticks)
            for pos in tick_positions:
                pts.append((pos, -tick_size))
                pts.append((pos, tick_size))
            return np.array(pts)

        def rotate_points(x, y, angle, scale=1.0):
            R = np.array([[np.cos(angle), -np.sin(angle)],
                          [np.sin(angle), np.cos(angle)]])
            coords = np.dot(R, np.vstack([x, y]))
            return coords[0] * scale, coords[1] * scale

        mandala_points = []
        base_pts = generate_base_pattern_points()
        for k in range(VORONOI_MANDALA_ROTATIONS):
            angle = 2 * np.pi * k / VORONOI_MANDALA_ROTATIONS
            scale = 0.5 + k * 0.01
            px, py = rotate_points(base_pts[:, 0], base_pts[:, 1], angle, scale)
            mandala_points.extend(list(zip(px * 0.5, py * 0.5)))

        mandala_points = np.array(mandala_points)

        # Fuse both point sets
        all_points = np.vstack([vor_points, mandala_points])

        # Create Voronoi tessellation
        vor = Voronoi(all_points)

        # Rasterize Voronoi edges onto intensity map
        for vpair in vor.ridge_vertices:
            if -1 in vpair:  # Skip infinite edges
                continue

            p1, p2 = vor.vertices[vpair]
            dist = np.linalg.norm(p1 - p2)

            # Only draw short lines
            if dist < VORONOI_MAX_LINE_DIST:
                # Calculate fade based on distance from center
                mid = (p1 + p2) / 2
                fade = np.linalg.norm(mid) / radius_max
                alpha = max(0.05, 1 - fade) * np.random.uniform(0.5, 1.0)

                # Draw line by sampling points along it
                num_samples = int(dist * 5000)
                for t in np.linspace(0, 1, num_samples):
                    px = p1[0] * (1 - t) + p2[0] * t
                    py = p1[1] * (1 - t) + p2[1] * t

                    # Convert to array indices
                    ix = int((px + 1) / 2 * WIDTH)
                    iy = int((py + 1) / 2 * HEIGHT)

                    if 0 <= ix < WIDTH and 0 <= iy < HEIGHT:
                        # Add with gaussian falloff for line thickness
                        for dx in range(-2, 3):
                            for dy in range(-2, 3):
                                nx, ny = ix + dx, iy + dy
                                if 0 <= nx < WIDTH and 0 <= ny < HEIGHT:
                                    line_dist = np.sqrt(dx ** 2 + dy ** 2)
                                    intensity[ny, nx] += alpha * np.exp(-(line_dist ** 2) / (VORONOI_LINE_WIDTH ** 2))

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