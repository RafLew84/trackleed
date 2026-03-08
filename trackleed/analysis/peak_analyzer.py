# stal/analysis/peak_analyzer.py

import numpy as np
from scipy.optimize import curve_fit
from typing import Dict, Optional, Tuple, List
import random
from skimage.measure import EllipseModel
from skimage.color import rgb2gray

from scipy.ndimage import center_of_mass

def gaussian_2d(
    xy_data: Tuple[np.ndarray, np.ndarray],
    amplitude: float,
    center_x: float, center_y: float, # CHANGE: keep naming consistent
    sigma_x: float, sigma_y: float,
    theta: float,
    offset: float
) -> np.ndarray:
    """
    Defines the mathematical model for a 2D elliptical Gaussian function.

    This model can describe a Gaussian peak that is elliptical and rotated by
    an angle theta. It's used as the target function for curve fitting.

    Args:
        xy_data: A tuple containing two 2D numpy arrays (X, Y), which are the
                 coordinate grids for the evaluation.
        amplitude: The amplitude of the Gaussian peak (its height above the offset).
        center_x: The x-coordinate of the center of the peak.
        center_y: The y-coordinate of the center of the peak.
        sigma_x: The standard deviation (width) of the Gaussian along its x-axis.
        sigma_y: The standard deviation (width) of the Gaussian along its y-axis.
        theta: The rotation angle of the ellipse in radians.
        offset: A constant vertical offset (the background level).

    Returns:
        A 1D flattened numpy array of the calculated Gaussian values, suitable for
        use with scipy.optimize.curve_fit.
    """
    # Unpack coordinate grids
    x, y = xy_data

    # Coefficients for the quadratic form of the rotated ellipse exponent
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    
    # Calculate the Gaussian function value at each point
    g = offset + amplitude * np.exp( - (a*((x-center_x)**2) + 2*b*(x-center_x)*(y-center_y) + c*((y-center_y)**2)))

    # Flatten the 2D result array into a 1D array, as required by curve_fit
    return g.ravel()

def calculate_spot_properties(
    fit_params: Dict[str, float],
    roi_data: np.ndarray,
    xy_coords: Tuple[np.ndarray, np.ndarray]
) -> Dict[str, float]:
    """
    Calculates derived physical and statistical properties from Gaussian fit results.

    Based on the parameters obtained from curve fitting, this function computes
    metrics like peak volume, FWHM, asymmetry, and goodness-of-fit statistics
    (R-squared, Chi-squared).

    Args:
        fit_params: A dictionary containing the fitted parameters from the
                    `fit_spot_to_gaussian` function.
        roi_data: The original 2D numpy array of the ROI data, required for
                  calculating goodness-of-fit.
        xy_coords: A tuple of X and Y coordinate grids, required to reconstruct
                   the fitted model data.

    Returns:
        A dictionary containing the calculated physical and statistical properties.
    """
    # --- Physical Properties ---

    # Volume: The integral of the 2D Gaussian peak.
    volume = 2 * np.pi * fit_params['amplitude'] * fit_params['sigma_x'] * fit_params['sigma_y']

    # Full Width at Half Maximum (FWHM) for both axes.
    # The conversion factor is 2 * sqrt(2 * ln(2)) ≈ 2.355.
    fwhm_x = 2 * np.sqrt(2 * np.log(2)) * fit_params['sigma_x']
    fwhm_y = 2 * np.sqrt(2 * np.log(2)) * fit_params['sigma_y']

    # Asymmetry: The ratio of the widths, quantifies the peak's ellipticity.
    asymmetry = fit_params['sigma_x'] / fit_params['sigma_y'] if fit_params['sigma_y'] != 0 else 0

    # --- Goodness-of-Fit Statistics ---
    
    # Recreate the fitted model data based on the optimal parameters.
    # We need to filter out non-model keys like 'covariance_matrix' before unpacking.
    fitted_data_flat = gaussian_2d(xy_coords, **{k: v for k, v in fit_params.items() if k != 'covariance_matrix'})
    
    # Flatten the original data for comparison.
    roi_data_flat = roi_data.ravel()

    # Sum of squared residuals (the squared difference between data and model).
    residuals = roi_data_flat - fitted_data_flat
    ss_res = np.sum(residuals**2)

    # Total sum of squares (proportional to the variance of the data).
    ss_tot = np.sum((roi_data_flat - np.mean(roi_data_flat))**2)

    # R-squared (coefficient of determination). A value close to 1.0 indicates a good fit.
    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 1.0
    
    # Reduced Chi-squared. A value close to 1.0 indicates a good fit.
    # Degrees of freedom (dof) = (number of data points) - (number of fitted parameters).
    # The covariance_matrix is not a model parameter, so we subtract 1 from len(fit_params).
    dof = len(roi_data_flat) - (len(fit_params) - 1) # -1 bo macierz kowariancji nie jest parametrem modelu
    chi_squared_reduced = ss_res / dof if dof > 0 else 0

    # --- Error Calculations ---

    perr = np.sqrt(np.diag(fit_params['covariance_matrix']))
    param_errors = {
        'amplitude_error': perr[0], 'center_x_error': perr[1], 'center_y_error': perr[2],
        'sigma_x_error': perr[3], 'sigma_y_error': perr[4], 'theta_error': perr[5], 'offset_error': perr[6]
    }
    

    fwhm_x_error = 2 * np.sqrt(2 * np.log(2)) * param_errors['sigma_x_error']
    fwhm_y_error = 2 * np.sqrt(2 * np.log(2)) * param_errors['sigma_y_error']
    
    A, sX, sY = fit_params['amplitude'], fit_params['sigma_x'], fit_params['sigma_y']
    err_A, err_sX, err_sY = param_errors.get('amplitude_error', 0), param_errors.get('sigma_x_error', 0), param_errors.get('sigma_y_error', 0)
    
    volume_error = volume * np.sqrt((err_A/A)**2 + (err_sX/sX)**2 + (err_sY/sY)**2) if A*sX*sY != 0 else 0

    
    # Return all calculated properties in a dictionary.
    return {
        'volume': volume, 
        'fwhm_x': fwhm_x, 
        'fwhm_y': fwhm_y, 
        'asymmetry': asymmetry,
        'r_squared': r_squared, 
        'chi_squared_reduced': chi_squared_reduced,
        **param_errors,
        'fwhm_x_error': fwhm_x_error,
        'fwhm_y_error': fwhm_y_error,
        'volume_error': volume_error
    }

def fit_spot_to_gaussian(roi_data: np.ndarray) -> Optional[Dict[str, float]]:
    """
    Performs a full analysis pipeline for a single spot's ROI.

    This function takes a 2D array of a Region of Interest (ROI), performs a
    2D elliptical Gaussian fit using scipy.optimize.curve_fit, and then
    calculates derived physical and statistical properties from the fit results.

    Args:
        roi_data: A 2D numpy array representing the pixel intensities of the spot.

    Returns:
        A dictionary containing all fitted parameters (like center, amplitude)
        and derived properties (like volume, FWHM, R-squared), or None if the
        fitting procedure fails to converge.
    """
    # Guard clause: Return None if the input data is invalid.
    if roi_data is None or roi_data.size == 0:
        return None

    roi_data = np.asarray(roi_data)
    # Gaussian fit is 2D; collapse multi-channel ROI to grayscale first.
    if roi_data.ndim >= 3:
        if roi_data.shape[-1] >= 3:
            roi_data = rgb2gray(roi_data[..., :3])
        else:
            roi_data = roi_data[..., 0]
    if roi_data.ndim != 2:
        return None
    roi_data = roi_data.astype(np.float64, copy=False)

    # Get the dimensions of the ROI.
    height, width = roi_data.shape
    # Create X and Y coordinate grids corresponding to the pixel positions.
    x, y = np.meshgrid(np.arange(width), np.arange(height))

    try:
        y0_init, x0_init = center_of_mass(roi_data)
    except ValueError:
        y0_init, x0_init = height / 2, width / 2

    initial_guess = [
        np.max(roi_data) - np.min(roi_data),  # amplitude: peak height over background
        x0_init,                             # center_x: position of the brightest pixel
        y0_init,                             # center_y: position of the brightest pixel
        width / 4,                           # sigma_x: a reasonable fraction of the ROI width
        height / 4,                          # sigma_y: a reasonable fraction of the ROI height
        0,                                   # theta: initial rotation angle
        np.min(roi_data)                     # offset: the dimmest pixel value as background
    ]

    lower_bounds = [0,       -width,   -height,   0.5,   0.5,   -np.pi, 0]
    upper_bounds = [np.inf,  2*width,  2*height,  width, height, np.pi, np.inf]
    bounds = (lower_bounds, upper_bounds)
    
    # Names of the parameters corresponding to the initial_guess list.
    param_names = ['amplitude', 'center_x', 'center_y', 'sigma_x', 'sigma_y', 'theta', 'offset']

    try:
        # Perform the actual curve fitting.
        # `popt` will contain the optimal parameters, `pcov` the covariance matrix.
        popt, pcov = curve_fit(f=gaussian_2d, xdata=(x, y), ydata=roi_data.ravel(), p0=initial_guess, bounds=bounds)
        
        # --- Process Results ---
        
        # Create a dictionary from the optimized parameters for readability.
        fit_params = dict(zip(param_names, popt))
        fit_params['covariance_matrix'] = pcov

        # Ensure sigma values are positive, as the fit can sometimes yield negative values.
        fit_params['sigma_x'] = abs(fit_params['sigma_x'])
        fit_params['sigma_y'] = abs(fit_params['sigma_y'])
        
        # Calculate derived properties (volume, FWHM, etc.) using the fitted parameters.
        spot_properties = calculate_spot_properties(fit_params, roi_data, (x, y))
        
        # Merge the basic fit parameters and the derived properties into a single result dictionary.
        final_results = {**fit_params, **spot_properties}
        
        return final_results

    except (RuntimeError, ValueError) as e:
        print(f"Error during Gaussian fitting: {e}")
        return None


def fit_circle_center(spot_positions: List[Dict[str, float]]) -> Optional[Dict[str, float]]:
    """
    Estimate the pattern center by least-squares circle fitting.

    Returns center_x/center_y and approximate errors derived from the fit residuals.
    """
    if len(spot_positions) < 3:
        return None

    pts = np.array([[p["center_x"], p["center_y"]] for p in spot_positions], dtype=np.float64)
    x = pts[:, 0]
    y = pts[:, 1]

    # Solve x^2 + y^2 + A x + B y + C = 0
    A_mat = np.column_stack((x, y, np.ones_like(x)))
    b_vec = -(x * x + y * y)
    coeffs, residuals, rank, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
    if rank < 3:
        return None

    a, b, c = coeffs
    center_x = -a / 2.0
    center_y = -b / 2.0
    radius_sq = center_x * center_x + center_y * center_y - c
    if radius_sq <= 0:
        return None
    radius = np.sqrt(radius_sq)

    # Estimate parameter uncertainty from residuals
    n = len(x)
    if residuals.size and n > 3:
        sigma2 = float(residuals[0] / (n - 3))
        cov = sigma2 * np.linalg.inv(A_mat.T @ A_mat)
        err_a = np.sqrt(max(cov[0, 0], 0.0))
        err_b = np.sqrt(max(cov[1, 1], 0.0))
        error_x = err_a / 2.0
        error_y = err_b / 2.0
    else:
        # Fallback: use radial residual scatter
        radial = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
        error = float(np.std(radial - radius)) if radial.size else 0.0
        error_x = error
        error_y = error

    return {
        "center_x": float(center_x),
        "center_y": float(center_y),
        "error_x": float(error_x),
        "error_y": float(error_y),
    }


def _ellipse_residuals_px(
    pts: np.ndarray,
    center_x: float,
    center_y: float,
    radius_x: float,
    radius_y: float,
    theta_rad: float,
) -> np.ndarray:
    """Compute ellipse residual in pixel-like units."""
    dx = pts[:, 0] - center_x
    dy = pts[:, 1] - center_y
    cos_t = np.cos(theta_rad)
    sin_t = np.sin(theta_rad)
    xr = cos_t * dx + sin_t * dy
    yr = -sin_t * dx + cos_t * dy
    inv_rx = 1.0 / max(radius_x, 1e-9)
    inv_ry = 1.0 / max(radius_y, 1e-9)
    q = np.sqrt((xr * inv_rx) ** 2 + (yr * inv_ry) ** 2)
    return np.abs(q - 1.0) * np.sqrt(max(radius_x * radius_y, 1e-9))


def _fit_ellipse_by_moments(pts: np.ndarray) -> Optional[Tuple[float, float, float, float, float, np.ndarray]]:
    """Fallback ellipse estimation from 2nd moments."""
    if pts.shape[0] < 3:
        return None
    center = np.mean(pts, axis=0)
    centered = pts - center
    cov = np.cov(centered.T)
    if cov.shape != (2, 2) or not np.all(np.isfinite(cov)):
        return None
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]
    if evals[0] <= 0 or evals[1] <= 0:
        return None

    radius_x = float(np.sqrt(max(2.0 * evals[0], 1e-9)))
    radius_y = float(np.sqrt(max(2.0 * evals[1], 1e-9)))
    theta = float(np.arctan2(evecs[1, 0], evecs[0, 0]))
    residuals = _ellipse_residuals_px(pts, float(center[0]), float(center[1]), radius_x, radius_y, theta)
    return float(center[0]), float(center[1]), radius_x, radius_y, theta, residuals


def fit_ellipse_center(spot_positions: List[Dict[str, float]]) -> Optional[Dict[str, float]]:
    """
    Estimate pattern center by fitting an ellipse to detected spot positions.

    Returns center, approximate errors, ellipse axes and angle for downstream
    corrected-radius computation.
    """
    if len(spot_positions) < 3:
        return None

    pts = np.array([[p["center_x"], p["center_y"]] for p in spot_positions], dtype=np.float64)
    pts = pts[np.all(np.isfinite(pts), axis=1)]
    if pts.shape[0] < 3:
        return None

    center_x = center_y = radius_x = radius_y = theta = None
    residuals = None

    # Primary method: direct geometric ellipse model.
    if pts.shape[0] >= 5:
        model = EllipseModel()
        ok = model.estimate(pts)
        if ok and model.params is not None:
            xc, yc, a, b, theta0 = model.params
            if np.isfinite(xc) and np.isfinite(yc) and np.isfinite(a) and np.isfinite(b) and a > 0 and b > 0:
                center_x = float(xc)
                center_y = float(yc)
                radius_x = float(a)
                radius_y = float(b)
                theta = float(theta0)
                residuals = _ellipse_residuals_px(pts, center_x, center_y, radius_x, radius_y, theta)

                # Robust refinement on inliers.
                if residuals.size >= 5:
                    med = float(np.median(residuals))
                    mad = float(np.median(np.abs(residuals - med)))
                    thr = med + max(1.5, 2.5 * mad)
                    inliers = residuals <= thr
                    if np.count_nonzero(inliers) >= 5 and np.count_nonzero(inliers) < pts.shape[0]:
                        refined = EllipseModel()
                        if refined.estimate(pts[inliers]) and refined.params is not None:
                            xc, yc, a, b, theta0 = refined.params
                            if np.isfinite(xc) and np.isfinite(yc) and np.isfinite(a) and np.isfinite(b) and a > 0 and b > 0:
                                center_x = float(xc)
                                center_y = float(yc)
                                radius_x = float(a)
                                radius_y = float(b)
                                theta = float(theta0)
                                residuals = _ellipse_residuals_px(pts[inliers], center_x, center_y, radius_x, radius_y, theta)

    # Fallback for degenerate/insufficient geometric fit.
    if center_x is None or center_y is None or radius_x is None or radius_y is None or theta is None:
        fallback = _fit_ellipse_by_moments(pts)
        if fallback is None:
            return None
        center_x, center_y, radius_x, radius_y, theta, residuals = fallback

    # Ensure major/minor axis ordering.
    if radius_y > radius_x:
        radius_x, radius_y = radius_y, radius_x
        theta += np.pi / 2.0

    # Normalize angle to [-pi, pi].
    theta = (theta + np.pi) % (2.0 * np.pi) - np.pi

    # Approximate center uncertainty from residual scatter.
    if residuals is None or residuals.size == 0:
        center_err = 0.0
    else:
        center_err = float(np.std(residuals) / np.sqrt(max(residuals.size, 1)))

    return {
        "center_x": float(center_x),
        "center_y": float(center_y),
        "error_x": float(center_err),
        "error_y": float(center_err),
        "radius_x": float(radius_x),
        "radius_y": float(radius_y),
        "angle_deg": float(np.degrees(theta)),
        "radius_scale_px": float(np.sqrt(max(radius_x * radius_y, 1e-9))),
    }
    
def find_pattern_center(spot_positions: List[Dict[str, float]], num_iterations: int = 100, threshold: float = 15.0) -> Optional[Dict[str, float]]:
    """
    Finds the pattern center using a RANSAC-based approach for robustness.

    This method is robust against asymmetric spot distributions and outliers.
    It randomly samples pairs of spots, hypothesizes their midpoint as the
    true center, and scores this hypothesis by counting how many other spots
    find a valid symmetric partner. The hypothesis with the best score is chosen.

    Args:
        spot_positions: A list of dictionaries, each with 'center_x' and 'center_y'.
        num_iterations: The number of RANSAC iterations to perform.
        threshold: The max distance (in pixels) to consider a spot a valid partner.

    Returns:
        A dictionary with the center coordinates and their error, or None.
    """
    if len(spot_positions) < 4:  # Need at least two pairs for a meaningful result
        return None

    best_inliers = []
    
    # Step 1-3: Iterate multiple times to find the best hypothesis
    for _ in range(num_iterations):
        # 1. Randomly sample two distinct points to form a hypothesis
        p1_dict, p2_dict = random.sample(spot_positions, 2)
        p1 = np.array([p1_dict['center_x'], p1_dict['center_y']])
        p2 = np.array([p2_dict['center_x'], p2_dict['center_y']])
        
        # The midpoint is our hypothesis for the true center
        center_hypothesis = (p1 + p2) / 2
        
        # 2. Find the consensus set (inliers) for this hypothesis
        current_inliers = []
        used_indices = set()
        
        all_points = [np.array([p['center_x'], p['center_y']]) for p in spot_positions]

        for i in range(len(all_points)):
            if i in used_indices: continue
            
            p_reflected = 2 * center_hypothesis - all_points[i]
            
            # Find the closest partner to the reflected point
            min_dist_sq = float('inf')
            best_j = -1
            for j in range(i + 1, len(all_points)):
                if j in used_indices: continue
                dist_sq = np.sum((all_points[j] - p_reflected)**2)
                if dist_sq < min_dist_sq:
                    min_dist_sq = dist_sq
                    best_j = j
            
            # If a partner is found within the threshold, they are inliers
            if best_j != -1 and np.sqrt(min_dist_sq) < threshold:
                current_inliers.append(all_points[i])
                current_inliers.append(all_points[best_j])
                used_indices.add(i)
                used_indices.add(best_j)

        # 3. If this hypothesis has more support, update our best model
        if len(current_inliers) > len(best_inliers):
            best_inliers = current_inliers
            best_center_hypothesis = center_hypothesis

    if not best_inliers:
        return None # Failed to find any consistent center

    # Step 4-5: Final Refinement using only the best set of inliers
    midpoints = []
    for i in range(0, len(best_inliers), 2):
        midpoint = (best_inliers[i] + best_inliers[i+1]) / 2
        midpoints.append(midpoint)
        
    midpoints_array = np.array(midpoints)
    final_center = np.mean(midpoints_array, axis=0)
    errors = np.std(midpoints_array, axis=0)

    return {
        'center_x': final_center[0], 'center_y': final_center[1],
        'error_x': errors[0], 'error_y': errors[1]
    }

# def find_pattern_center(spot_positions: List[Dict[str, float]]) -> Optional[Dict[str, float]]:
#     """
#     Finds the center of a diffraction pattern by leveraging its centrosymmetry.

#     This method, based on Friedel's Law, assumes that for every spot at a
#     position (kx, ky) relative to the center, a corresponding spot exists at
#     (-kx, -ky). The center is determined by finding these symmetric pairs and
#     averaging their midpoints. The function also calculates the standard
#     deviation of these midpoints as a measure of the result's uncertainty.

#     Args:
#         spot_positions: A list of dictionaries, where each dictionary represents
#                         a single spot and must contain 'center_x' and 'center_y'
#                         keys with their global coordinates.

#     Returns:
#         A dictionary containing the calculated center coordinates ('center_x',
#         'center_y') and their corresponding errors ('error_x', 'error_y'),
#         or None if not enough spots are provided.
#     """
#     # Guard clause: We need at least two spots to find a pair.
#     if len(spot_positions) < 2: return None

#     # Step 1: Calculate a rough estimate of the center using the geometric mean.
#     # This will be the reference point for finding symmetric pairs.
#     all_x = [p['center_x'] for p in spot_positions]
#     all_y = [p['center_y'] for p in spot_positions]
#     rough_center_x, rough_center_y = np.mean(all_x), np.mean(all_y)

#     # Step 2: Find all centrosymmetric pairs (Friedel pairs).
#     midpoints, used_indices = [], set()  # Keep track of spots that have already been paired.

#     for i in range(len(spot_positions)):
#         # Skip if this spot is already part of another pair.
#         if i in used_indices: continue

#         p1 = np.array([spot_positions[i]['center_x'], spot_positions[i]['center_y']])

#         # Calculate the ideal position of the symmetric partner.
#         p_reflected = 2 * np.array([rough_center_x, rough_center_y]) - p1

#         # Find the actual spot in the list that is closest to the ideal reflected position.
#         best_j, min_dist_sq = -1, float('inf')
#         for j in range(i + 1, len(spot_positions)):
#             if j in used_indices: continue
#             p2 = np.array([spot_positions[j]['center_x'], spot_positions[j]['center_y']])
#             dist_sq = np.sum((p2 - p_reflected)**2)
#             if dist_sq < min_dist_sq:
#                 min_dist_sq, best_j = dist_sq, j

#         # If a suitable partner was found within a reasonable threshold, accept the pair.
#         # The threshold (e.g., 30 pixels) prevents incorrect pairings if a spot is missing.
#         if best_j != -1 and np.sqrt(min_dist_sq) < 30:
#             p2 = np.array([spot_positions[best_j]['center_x'], spot_positions[best_j]['center_y']])

#             # The midpoint of a symmetric pair is a strong candidate for the true pattern center.
#             midpoints.append((p1 + p2) / 2)

#             # Mark both spots as used so they are not considered again.
#             used_indices.add(i)
#             used_indices.add(best_j)

#     # If no valid pairs were found, return the initial rough estimate as a fallback.
#     if not midpoints: return {'center_x': rough_center_x, 'center_y': rough_center_y, 'error_x': -1, 'error_y': -1}

#     # Step 3: Calculate the final center.
#     # The average of all calculated midpoints gives a robust and precise center position.
#     midpoints = np.array(midpoints)
#     return {
#         'center_x': np.mean(midpoints[:, 0]), 'center_y': np.mean(midpoints[:, 1]),
#         'error_x': np.std(midpoints[:, 0]), 'error_y': np.std(midpoints[:, 1])
#     }

