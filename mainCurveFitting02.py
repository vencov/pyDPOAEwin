import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


plt.close('all')
# New data points
L2 = np.array([65, 60, 55, 50, 45, 40, 35, 30, 25])  # L2 values
y_data = np.array([11.556697258920057, 11.861393781825743, 11.729244579811294, 
                   11.865495319201491, 10.104326505666553, 7.129411873380711, 
                   2.530054496242678, -0.02670970832198989, -5.128651569306521])

# Define the fitting function
def custom_function(L, A, A0, A1, a, b):
    return A * (1 + ((L / A0) ** a) / ((1 + L / A1) ** (a - b)))

# Try different initial guesses
initial_guesses = [15, 50, 50, 1, 1]  # Adjusted guesses based on data

# Fit the model to the original y_data
popt, pcov = curve_fit(custom_function, L2, y_data, p0=initial_guesses, maxfev=5000)

# Generate data for the fitted curve
x_fit = np.linspace(25, 65, 100)
y_fit = custom_function(x_fit, *popt)

# Calculate slopes numerically for the fitted data
dy = np.gradient(y_fit, x_fit)

# Find the maximum slope and its corresponding L2 level
max_slope_index = np.argmax(dy)
max_slope = dy[max_slope_index]
L2_at_max_slope = x_fit[max_slope_index]
OAE_level_at_max_slope = y_fit[max_slope_index]

# Find the point where the slope decreases to 50% of the maximum slope
half_max_slope = max_slope / 2
point_below_half_max_slope = np.where(dy < half_max_slope)[0]

if len(point_below_half_max_slope) > 0:
    L2_half_max_slope = x_fit[point_below_half_max_slope[0]]
    OAE_level_half_max_slope = y_fit[point_below_half_max_slope[0]]
else:
    L2_half_max_slope = None
    OAE_level_half_max_slope = None

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(L2, y_data, 'o', label='Original Data Points', color='blue')
plt.plot(x_fit, y_fit, '-', label='Fitted Curve', color='orange')
plt.axvline(L2_at_max_slope, color='green', linestyle='--', label='Max Slope Point')
plt.axhline(OAE_level_at_max_slope, color='red', linestyle='--', label='OAE Level at Max Slope')
plt.xlabel('$L_2$ (dB SPL)')
plt.ylabel('Amplitude')
plt.title('Curve Fitting with Custom Function (Non-transformed Data)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Output the results
print(f'Fitted parameters:\n A = {popt[0]:.4f}\n A0 = {popt[1]:.4f}\n A1 = {popt[2]:.4f}\n a = {popt[3]:.4f}\n b = {popt[4]:.4f}')
print(f'Maximum slope: {max_slope:.4f} at L2 = {L2_at_max_slope:.2f} dB')
print(f'OAE level at maximum slope: {OAE_level_at_max_slope:.4f}')
if L2_half_max_slope is not None:
    print(f'L2 at 50% max slope: {L2_half_max_slope:.2f} dB, OAE level: {OAE_level_half_max_slope:.4f}')
else:
    print('No point found where the slope decreases to 50% of maximum slope.')
