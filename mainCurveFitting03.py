# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 11:00:24 2024

@author: audiobunka
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import Polynomial

# New data points
L2 = np.array([65, 60, 55, 50, 45, 40, 35, 30, 25])
# %%
y_data = DPioNL[0]


# Fit a polynomial of degree 4 (you can experiment with different degrees)
degree = 4
p = Polynomial.fit(L2, y_data, deg=degree)

# Generate fitted values
x_fit = np.linspace(25, 65, 100)
y_fit = p(x_fit)

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
plt.plot(x_fit, y_fit, '-', label='Fitted Polynomial Curve', color='orange')
plt.axvline(L2_at_max_slope, color='green', linestyle='--', label='Max Slope Point')
plt.axhline(OAE_level_at_max_slope, color='red', linestyle='--', label='OAE Level at Max Slope')
plt.xlabel('$L_2$ (dB SPL)')
plt.ylabel('Amplitude')
plt.title('Polynomial Curve Fitting')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Output the results
print(f'Polynomial coefficients: {p}')
print(f'Maximum slope: {max_slope:.4f} at L2 = {L2_at_max_slope:.2f} dB')
print(f'OAE level at maximum slope: {OAE_level_at_max_slope:.4f}')
if L2_half_max_slope is not None:
    print(f'L2 at 50% max slope: {L2_half_max_slope:.2f} dB, OAE level: {OAE_level_half_max_slope:.4f}')
else:
    print('No point found where the slope decreases to 50% of maximum slope.')
