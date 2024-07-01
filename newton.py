import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root
import sympy as sp
from matplotlib import patheffects
from scipy.ndimage import gaussian_filter
import os
import datetime

# Set plot parameters
width, height = 1000, 1000
max_iter = 2000  # Maximum number of iterations for the fractal generation
max_unchanged_iterations = 200  # Maximum number of iterations with unchanged non-converged points
show_root_labels = True

a = 1 # Relaxation parameter
def P(z): # Choose the polynomial
    #return z**30 - 1
    return (z - 1) * (z - 6) * (z - 3j)

def P_sympy(z):
    return sp.expand(P(sp.Symbol('z')))

def P_numpy(z):
    return sp.lambdify(sp.Symbol('z'), P_sympy(z), 'numpy')(z)

def dP_numpy(z):
    dP = sp.diff(P_sympy(z), sp.Symbol('z'))
    return sp.lambdify(sp.Symbol('z'), dP, 'numpy')(z)

def newton_iteration(z, a):
    Pz = P_numpy(z)
    dPz = dP_numpy(z)
    small_derivative = np.abs(dPz) < 1e-10
    z_new = np.where(small_derivative, z, z - a * Pz / dPz)
    max_abs_z = 1e6
    z_new = np.where(np.abs(z_new) > max_abs_z, max_abs_z * z_new / np.abs(z_new), z_new)
    return z_new

def determine_degree(P_sympy):
    return sp.degree(P_sympy)

def find_roots(P_sympy):
    z = sp.Symbol('z')
    try:
        roots = sp.solve(P_sympy, z)
        roots = [complex(root.evalf()) for root in roots]
    except:
        print("SymPy root-finding failed. Falling back to numerical method.")
        P_func = sp.lambdify(z, P_sympy, 'numpy')
        roots = []
        for x in np.linspace(-10, 10, 100):
            for y in np.linspace(-10, 10, 100):
                root = root(P_func, x + y*1j)
                if root.success and not any(np.isclose(root.x, r) for r in roots):
                    roots.append(root.x)
    
    print(f"Found {len(roots)} roots")
    return roots

def generate_fractal(xmin, xmax, ymin, ymax, width, height, max_iter, max_unchanged_iterations, a, roots):
    x = np.linspace(xmin, xmax, width)
    y = np.linspace(ymin, ymax, height)
    X, Y = np.meshgrid(x, y)
    Z = X + Y * 1j
    
    C = np.zeros((height, width), dtype=int)
    
    eps = 1e-6
    
    jitter_x = np.random.uniform(-0.5, 0.5, Z.shape) * (xmax - xmin) / width
    jitter_y = np.random.uniform(-0.5, 0.5, Z.shape) * (ymax - ymin) / height
    Z += jitter_x + 1j * jitter_y
    
    initial_non_converged = width * height
    last_non_converged = initial_non_converged
    unchanged_count = 0
    
    for i in range(max_iter):
        Z_new = newton_iteration(Z, a)
        non_converged = np.sum(C == 0)
        print(f"Iteration {i}/{max_iter}: Non-converged points: {non_converged}")
        
        if non_converged == last_non_converged:
            unchanged_count += 1
        else:
            unchanged_count = 0
        
        if unchanged_count >= max_unchanged_iterations and non_converged != initial_non_converged:
            print(f"Non-converged points unchanged for {max_unchanged_iterations} iterations. Halting at iteration {i}.")
            break
        
        if non_converged == 0:
            print(f"All points converged after {i} iterations.")
            break
        
        for j, root in enumerate(roots):
            mask = np.abs(Z_new - root) < eps
            C[mask & (C == 0)] = j + 1
        Z = Z_new
        last_non_converged = non_converged
    
    C = gaussian_filter(C.astype(float), sigma=0.5)
    
    print(f"Final non-converged points: {np.sum(C == 0)}")
    return C

# Determine the degree and find roots
P_sympy_expr = P_sympy(sp.Symbol('z'))
degree = determine_degree(P_sympy_expr)
roots = find_roots(P_sympy_expr)

# Determine plot range based on roots
real_parts = [root.real for root in roots]
imag_parts = [root.imag for root in roots]
xmin, xmax = min(real_parts), max(real_parts)
ymin, ymax = min(imag_parts), max(imag_parts)

# Add padding (50% of the range)
padding_x = 0.5 * (xmax - xmin)
padding_y = 0.5 * (ymax - ymin)
xmin -= padding_x
xmax += padding_x
ymin -= padding_y
ymax += padding_y

# Ensure a minimum plot range
min_range = 2.0
if xmax - xmin < min_range:
    center = (xmax + xmin) / 2
    xmin = center - min_range / 2
    xmax = center + min_range / 2
if ymax - ymin < min_range:
    center = (ymax + ymin) / 2
    ymin = center - min_range / 2
    ymax = center + min_range / 2

# Adjust the range to match the image aspect ratio while maintaining scale
image_aspect_ratio = width / height
range_x = xmax - xmin
range_y = ymax - ymin
current_aspect_ratio = range_x / range_y

if current_aspect_ratio > image_aspect_ratio:
    # Too wide, increase height
    new_range_y = range_x / image_aspect_ratio
    diff = new_range_y - range_y
    ymin -= diff / 2
    ymax += diff / 2
else:
    # Too tall, increase width
    new_range_x = range_y * image_aspect_ratio
    diff = new_range_x - range_x
    xmin -= diff / 2
    xmax += diff / 2

# Generate the fractal
fractal = generate_fractal(xmin, xmax, ymin, ymax, width, height, max_iter, max_unchanged_iterations, a, roots)

# Create a discrete color map
n_roots = len(roots)
colors = plt.cm.viridis(np.linspace(0, 1, n_roots))
cmap = plt.cm.colors.ListedColormap(['black'] + list(colors))
bounds = np.arange(n_roots + 2) - 0.5
norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)

# Create the output directory if it doesn't exist
output_dir = "newton-output"
os.makedirs(output_dir, exist_ok=True)

# Plot the fractal for saving
fig_save = plt.figure(figsize=(width/100, height/100), dpi=100)
ax_save = fig_save.add_subplot(111)
im = ax_save.imshow(fractal, extent=(xmin, xmax, ymin, ymax), cmap=cmap, norm=norm, interpolation='nearest', aspect='equal')

# Remove all axes, labels, and white space
ax_save.axis('off')
plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
plt.margins(0,0)

# Plot the roots on the saving figure
for i, root in enumerate(roots):
    ax_save.plot(root.real, root.imag, 'ro', markersize=10)
    if show_root_labels:
        text_x = root.real + 0.01 * (xmax - xmin)
        text_y = root.imag + 0.01 * (ymax - ymin)
        ax_save.text(text_x, text_y, f'Root {i+1}', color='white', fontweight='bold', 
                     verticalalignment='bottom', horizontalalignment='left',
                     path_effects=[patheffects.withStroke(linewidth=3, foreground='black')])

# Get current date and time
current_time = datetime.datetime.now()
time_string = current_time.strftime("%Y%m%d-%H%M%S")

# Create the filename with date, time, and dimensions
output_filename = os.path.join(output_dir, f"newton_fractal_{time_string}_{width}x{height}.png")

# Save the figure
fig_save.savefig(output_filename, dpi=100, bbox_inches='tight', pad_inches=0)
print(f"Image saved as {output_filename}")
plt.close(fig_save)

print(f"Polynomial degree: {degree}")
print(f"Roots found: {roots}")
print(f"Plot range: x: [{xmin:.2f}, {xmax:.2f}], y: [{ymin:.2f}, {ymax:.2f}]")