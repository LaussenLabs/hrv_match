from sklearn import linear_model
import numpy as np

from peakalignment.constants import max_lines, residual_threshold_ms, max_trials, min_line_inliers


def get_regression_lines(offset_t: np.ndarray, offset_v: np.ndarray):
    """
    Find the linear regression lines in the offset_t and offset_v data. Fit each line, remove the inliers,
    and repeat until stopping criteria are met. The fitted lines are sorted in descending order by the
    number of inliers.

    :param offset_t: Numpy array of time offsets.
    :type offset_t: np.ndarray
    :param offset_v: Numpy array of value offsets.
    :type offset_v: np.ndarray

    :return: A list of lists containing the start, end, coefficients, score, and number of inliers for each line.
    :rtype: List[List]

    """
    # Initialize an empty list to store line data
    line_data = []

    # Iterate through all the possible lines up to the maximum allowed lines
    for line in range(max_lines):
        try:
            # Get one line from the data using RANSAC linear regression and the given stopping criteria
            coef, score, trimmed_inlier_mask, new_outlier_mask = \
                get_one_line(offset_t, offset_v, residual_threshold=residual_threshold_ms,
                             max_trials=max_trials, min_line_inliers=min_line_inliers)
        except ValueError:
            # If there is an error while fitting the line, break the loop
            break

        # If the fitted line has no coefficients, break the loop
        if coef is None:
            break

        # Calculate the limits for the fitted line
        n_min = int(np.min(offset_t[trimmed_inlier_mask]))
        n_max = int(np.max(offset_t[trimmed_inlier_mask]))

        # Append the line data to the line_data list
        line_data.append([n_min, n_max, coef, score, np.count_nonzero(trimmed_inlier_mask)])

        # Remove the inliers from the original data
        offset_t, offset_v = offset_t[new_outlier_mask], offset_v[new_outlier_mask]

        # If the remaining data points are less than the minimum inliers required, break the loop
        if offset_v.size < min_line_inliers:
            break

    # Return the line data
    return line_data

def get_one_line(match_n_times, offset, residual_threshold=10_000_000, max_trials=10_000_000, min_line_inliers=None):
    coef, score, inlier_mask, outlier_mask = \
        get_ransac_output(match_n_times, offset, residual_threshold=residual_threshold,
                          max_trials=max_trials)

    av = np.mean(match_n_times[inlier_mask])
    std = np.std(match_n_times[inlier_mask])

    trimmed_inlier_mask = np.logical_and(inlier_mask, np.logical_and(
        match_n_times > av - (2.2 * std), match_n_times < av + (2.2 * std)))

    if np.count_nonzero(trimmed_inlier_mask) < min_line_inliers:
        return None, None, None, None

    new_outlier_mask = np.logical_not(trimmed_inlier_mask)

    coef, _, inlier_mask, outlier_mask = \
        get_ransac_output(match_n_times[trimmed_inlier_mask], offset[trimmed_inlier_mask],
                          residual_threshold=residual_threshold, max_trials=max_trials)

    inlier_times, inlier_offsets = match_n_times[trimmed_inlier_mask], offset[trimmed_inlier_mask]

    b, m = coef

    # Calculate distance for each point and store it in a list
    distances = np.array(abs(m * inlier_times - inlier_offsets + b)) / np.sqrt(m ** 2 + 1)

    # Calculate the mean of the distances
    score = np.mean(distances)

    return coef, score, trimmed_inlier_mask, new_outlier_mask


def get_ransac_output(input_arr, output_arr, residual_threshold=None, max_trials=100):
    X = input_arr.reshape((input_arr.size, 1))
    y = output_arr

    ransac = linear_model.RANSACRegressor(residual_threshold=residual_threshold, max_trials=max_trials)
    ransac.fit(X, y)
    score = ransac.score(X, y)

    inlier_mask = ransac.inlier_mask_
    outlier_mask = np.logical_not(inlier_mask)

    coef = (ransac.estimator_.intercept_, ransac.estimator_.coef_[0])

    return coef, score, inlier_mask, outlier_mask
