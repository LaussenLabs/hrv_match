from typing import List, Tuple

from peakalignment.constants import min_inlier_density_ms, min_slope, max_slope


def get_filtered_lines(line_data: List[Tuple]) -> List[Tuple]:
    """
    Filter the given line data based on the minimum inlier_density and remove any collisions.

    :param List[Tuple] line_data: A list of tuples containing line data
                                  (start_time, end_time, (b, m), score, num_inliers).

    :return: A list of filtered line data tuples.
    :rtype: List[Tuple]

    """
    # Filter Inlier Density
    line_data = [(start_time, end_time, (b, m), score, num_inliers) for
                 (start_time, end_time, (b, m), score, num_inliers) in line_data if
                 (num_inliers / (end_time - start_time)) >= min_inlier_density_ms]

    # Filter Slope
    line_data = [(start_time, end_time, (b, m), score, num_inliers) for
                 (start_time, end_time, (b, m), score, num_inliers) in line_data if
                 min_slope <= m <= max_slope]

    # Filter Collisions
    line_data = collision_filter(line_data)

    return line_data


def collision_filter(line_data: List[Tuple]) -> List[Tuple]:
    """
    Filter out lines that collide with other lines, keeping only the lines with the highest inlier densities.

    :param List line_data: List of tuples containing the start, end, angle, count, and inlier count of the lines.

    :return: List of filtered line data tuples.
    :rtype: List[Tuple[float, float, float, float, int]]
    """
    # Initialize an empty list to store the indices of lines to be excluded from the output
    excluded_indices = []

    # Iterate over each line (index_1) and its data (start_1, end_1, count_1)
    for index_1, (start_1, end_1, _, _, count_1) in enumerate(line_data):
        # Calculate the inlier_density_1 as the count_1 divided by the length of the line
        inlier_density_1 = count_1 / (end_1 - start_1)

        # Iterate over each line (index_2) and its data (start_2, end_2, count_2)
        for index_2, (start_2, end_2, _, _, count_2) in enumerate(line_data):
            # Skip the line if it is the same as the first line
            if index_1 == index_2:
                continue

            # Calculate the inlier_density_2 as the count_2 divided by the length of the line
            inlier_density_2 = count_2 / (end_2 - start_2)

            # If the two lines collide and the second line has a higher inlier density, add the first line's index to the list of excluded indices
            if start_1 <= end_2 and start_2 <= end_1 and inlier_density_2 > inlier_density_1:
                excluded_indices.append(index_1)
                break

    # Return a list of the line data without the excluded lines
    return [row for i, row in enumerate(line_data) if i not in excluded_indices]
