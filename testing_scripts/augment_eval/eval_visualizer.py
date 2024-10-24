import sqlite3
import numpy as np
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import re
import time
import argparse
from matplotlib.colors import LinearSegmentedColormap


class EvaluationVisualizer:
    def __init__(self, db_path):
        """ Initialize with the path to the SQLite database. """
        self.db_path = db_path
        self.conn = None
        self.rotation_combinations = None
        
    def connect_database(self):
        """Connect to the SQLite database. """
        self.conn = sqlite3.connect(self.db_path)
        
    def close_database(self):
        """Close the database connection. """
        if self.conn:
            self.conn.close()
            
    def parse_filename(self):
        """
        Parse the database filename to extract the dataset, model size, and rotation ranges.
        Example filename: 'eval/Kitti_eval_base_(-6.0_6.0_2.0).db'
        Returns: dataset_name, model_size, (rotation_start, rotation_stop, rotation_step)
        """
        pattern = r'([a-zA-Z]+)_eval_([a-zA-Z]+)_\(([-\d.]+)_([-\d.]+)_([-\d.]+)\)'
        match = re.search(pattern, self.db_path)

        if match:
            dataset_name = match.group(1).capitalize()  # e.g., 'Kitti'
            model_size = match.group(2)                 # e.g., 'base'
            rotation_start = float(match.group(3))      # e.g., -6.0
            rotation_stop = float(match.group(4))       # e.g., 6.0
            rotation_step = float(match.group(5))       # e.g., 2.0
            return dataset_name, model_size, rotation_start, rotation_stop, rotation_step
        else:
            raise ValueError(f"Filename {self.db_path} doesn't match the expected format.")

            
    def compute_statistics(self):
        """Compute and store statistics (mean, std, min, max) for each (roll, pitch, yaw) combination."""
        # Parse the filename for dataset, model size, and rotation range
        dataset_name, model_size, rotation_start, rotation_stop, rotation_step = self.parse_filename()

        # Calculate the number of steps and generate the roll, pitch, and yaw values using np.linspace
        num_steps = int(np.ceil((rotation_stop - rotation_start) / rotation_step)) + 1
        roll_values = np.linspace(rotation_start, rotation_stop, num_steps)
        pitch_values = np.linspace(rotation_start, rotation_stop, num_steps)
        yaw_values = np.linspace(rotation_start, rotation_stop, num_steps)

        self.rotation_combinations = list(itertools.product(roll_values, pitch_values, yaw_values))

        self.connect_database()
        cursor = self.conn.cursor()
        
        # Create statistics table if not exists
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS statistics (
                roll REAL NOT NULL,
                pitch REAL NOT NULL,
                yaw REAL NOT NULL,
                metric_name TEXT NOT NULL,
                mean REAL NOT NULL,
                std REAL NOT NULL,
                min REAL NOT NULL,
                max REAL NOT NULL,
                PRIMARY KEY (roll, pitch, yaw, metric_name)
            )
        ''')
        
        self.rotation_combinations = list(itertools.product(roll_values, pitch_values, yaw_values))
        metrics = ['RMSE', 'MAE', 'iRMSE', 'iMAE', 'REL', 'D1', 'D2', 'D3'] # List of evaluation metrics
        
        # Resume progress based on the number of unique (roll, pitch, yaw) combinations
        cursor.execute('SELECT COUNT(*) FROM statistics')
        total_rows = cursor.fetchone()[0]
        
        # Since we have multiple metrics per combination, we need to divide by the number of metrics
        unique_processed_combinations = total_rows // len(metrics)
        
        # Adjust the list ot start from where we left off
        if unique_processed_combinations > 0:
            print(f"Resuming progress from {unique_processed_combinations} unique combinations...")
            self.rotation_combinations = self.rotation_combinations[unique_processed_combinations:]
            
        for roll, pitch, yaw in tqdm(self.rotation_combinations):
            for metric in metrics:
                cursor.execute(f'''
                    SELECT {metric} FROM evaluations WHERE roll = ? AND pitch = ? AND yaw = ?
                ''', (roll, pitch, yaw))
                results = cursor.fetchall()
                if results:
                    metric_values = np.array([x[0] for x in results if x[0] is not None])

                    # If we have metric values, compute statistics
                    if len(metric_values) > 1:
                        mean_val = np.mean(metric_values)
                        std_val = np.std(metric_values)
                        min_val = np.min(metric_values)
                        max_val = np.max(metric_values)
                        
                        # Insert statistics into the statistics table
                        cursor.execute('''
                            INSERT OR REPLACE INTO statistics (roll, pitch, yaw, metric_name, mean, std, min, max)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (roll, pitch, yaw, metric, mean_val, std_val, min_val, max_val))
                    else:
                        raise ValueError(f"Length of metric_values is {len(metric_values)}")
            self.conn.commit()
            
        self.close_database()
        
    def load_statistics(self, metric_name):
        """
        Load statistics for a specific metric from the database into a dictionary.
        This method only loads data for the specified metric to save memory.
        """
        self.connect_database()
        cursor = self.conn.cursor()
        
        # Fetch data only for the specified metric
        cursor.execute('''
                SELECT roll, pitch, yaw, mean, std, min, max 
                FROM statistics 
                WHERE metric_name = ?
            ''', (metric_name,))
        
        data = {}
        
        for row in cursor.fetchall():
            roll, pitch, yaw, mean_val, std_val, min_val, max_val = row
            data[(roll, pitch, yaw)] = {
                'mean': mean_val,
                'std': std_val,
                'min': min_val,
                'max': max_val
            }
            
        self.close_database()
        return data
    
    # def plot_voxel_3d(self, metric_name):
    #     """Plot 3D voxelized representation of the metric score."""
        
    #     # Parse the filename for dataset and model size
    #     dataset_name, model_size, rotation_start, rotation_stop, rotation_step = self.parse_filename()

    #     # Calculate roll, pitch, yaw values
    #     num_steps = int(np.ceil((rotation_stop - rotation_start) / rotation_step)) + 1
    #     roll_values = np.linspace(rotation_start, rotation_stop, num_steps)
    #     pitch_values = np.linspace(rotation_start, rotation_stop, num_steps)
    #     yaw_values = np.linspace(rotation_start, rotation_stop, num_steps)
        
    #     # Load statistics for the specified metric
    #     data = self.load_statistics(metric_name)
        
    #     fig = plt.figure()
    #     ax = fig.add_subplot(111, projection='3d')
        
    #     # Create voxel grid
    #     voxel_grid = np.zeros((len(roll_values), len(pitch_values), len(yaw_values)))
    #     alpha_grid = np.zeros((len(roll_values), len(pitch_values), len(yaw_values)))
        
    #     # Error-based metrics (0 is best) and accuracy-based metrics (1 is best)
    #     error_metrics = ['RMSE', 'MAE', 'iRMSE', 'iMAE', 'REL']
    #     accuracy_metrics = ['D1', 'D2', 'D3']
        
    #     # Fill the grid with the data values
    #     for idx, (roll, pitch, yaw) in enumerate(itertools.product(roll_values, pitch_values, yaw_values)):
    #         values = data.get((roll, pitch, yaw), {})
    #         if not values:
    #             continue  # Skip if no data available for this combination
            
    #         mean_val = data.get((roll, pitch, yaw), {}).get('mean', None)
    #         std_val = data.get((roll, pitch, yaw), {}).get('std', None)
            
    #         # Determine alpha value based on the metric
    #         if metric_name in error_metrics:
    #             # Error-based metrics (0 is best)
    #             if mean_val <= 0:
    #                 alpha = 1.0 
    #             else:
    #                 alpha = max(0, 1-(mean_val / (2 * std_val)))
    #         elif metric_name in accuracy_metrics:
    #             # Accuracy-based metrics (1 is best)
    #             if mean_val >= 1:
    #                 alpha = 1.0 # compleately opaque
    #             else:
    #                 alpha = max(0, 1- ((1- mean_val) / (2 * std_val)))
    #         else:
    #             raise ValueError(f"Invalid metric name: {metric_name}")
            
    #         # Fill the voxela and alpha grids.
    #         voxel_grid[idx // len(pitch_values), idx % len(pitch_values), yaw_values.index(yaw)] = mean_val
    #         alpha_grid[idx // len(pitch_values), idx % len(pitch_values), yaw_values.index(yaw)] = alpha
            
    #     # Visualize the voxel grid with colos and alpha transparency
    #     ax.voxels(voxel_grid, facecolors=plt.cm.viridis(voxel_grid), alpha=alpha_grid)
    #     plt.title(f"3D Visualization of {metric_name} - Dataset: {dataset_name}, Model: LRRU_{model_size}")
    #     plt.show()

    # def plot_voxel_3d(self, metric_name):
    #     """Plot 3D voxelized representation of the metric score."""
        
    #     # Parse the filename for dataset and model size
    #     dataset_name, model_size, rotation_start, rotation_stop, rotation_step = self.parse_filename()

    #     # Calculate roll, pitch, yaw values
    #     num_steps = int(np.ceil((rotation_stop - rotation_start) / rotation_step)) + 1
    #     roll_values = np.linspace(rotation_start, rotation_stop, num_steps)
    #     pitch_values = np.linspace(rotation_start, rotation_stop, num_steps)
    #     yaw_values = np.linspace(rotation_start, rotation_stop, num_steps)
        
    #     # Load statistics for the specified metric
    #     data = self.load_statistics(metric_name)
        
    #     fig = plt.figure()
    #     ax = fig.add_subplot(111, projection='3d')
        
    #     # Create voxel grid
    #     voxel_grid = np.zeros((len(roll_values), len(pitch_values), len(yaw_values)))
    #     alpha_grid = np.zeros((len(roll_values), len(pitch_values), len(yaw_values)))
        
    #     # Error-based metrics (0 is best) and accuracy-based metrics (1 is best)
    #     error_metrics = ['RMSE', 'MAE', 'iRMSE', 'iMAE', 'REL']
    #     accuracy_metrics = ['D1', 'D2', 'D3']
        
    #     # Fill the grid with the data values
    #     for idx, (roll_idx, pitch_idx, yaw_idx) in enumerate(itertools.product(range(len(roll_values)), 
    #                                                                         range(len(pitch_values)), 
    #                                                                         range(len(yaw_values)))):
    #         roll, pitch, yaw = roll_values[roll_idx], pitch_values[pitch_idx], yaw_values[yaw_idx]
    #         values = data.get((roll, pitch, yaw), {})
    #         if not values:
    #             continue  # Skip if no data available for this combination
                
    #         mean_val = values.get('mean', None)
    #         std_val = values.get('std', None)
            
    #         # Check if mean or std is None (missing data)
    #         if mean_val is None or std_val is None or std_val == 0:
    #             continue  # Skip this entry if data is missing or std_val is 0
                
    #         # Determine alpha value based on the metric
    #         if metric_name in error_metrics:
    #             # Error-based metrics (0 is best)
    #             alpha = max(0, 1 - (mean_val / (2 * std_val))) if mean_val > 0 else 1.0
    #         elif metric_name in accuracy_metrics:
    #             # Accuracy-based metrics (1 is best)
    #             alpha = max(0, 1 - ((1 - mean_val) / (2 * std_val))) if mean_val < 1 else 1.0
    #         else:
    #             raise ValueError(f"Invalid metric name: {metric_name}")
            
    #         # Fill the voxel and alpha grids
    #         voxel_grid[roll_idx, pitch_idx, yaw_idx] = mean_val
    #         alpha_grid[roll_idx, pitch_idx, yaw_idx] = alpha
        
    #     # Create a boolean mask where data exists for the voxels
    #     filled_voxels = voxel_grid > 0  # or use np.isnan(voxel_grid) for stricter checks
        
    #     # Create a facecolors array with RGBA values, combining viridis colormap with alpha
    #     facecolors = plt.cm.viridis(voxel_grid)  # Get RGBA colors from the viridis colormap
    #     facecolors[..., 3] = alpha_grid  # Set the alpha (transparency) from the alpha_grid
        
    #     # Visualize the voxel grid with colors and transparency
    #     ax.voxels(filled_voxels, facecolors=facecolors)
        
    #     plt.title(f"3D Visualization of {metric_name} - Dataset: {dataset_name}, Model: LRRU_{model_size}")
    #     plt.show()

    def plot_voxel_3d(self, metric_name):
        """Plot 3D scatter plot representation of the metric score."""
        
        # Parse the filename for dataset and model size
        dataset_name, model_size, rotation_start, rotation_stop, rotation_step = self.parse_filename()

        # Calculate roll, pitch, yaw values
        num_steps = int(np.ceil((rotation_stop - rotation_start) / rotation_step)) + 1
        roll_values = np.linspace(rotation_start, rotation_stop, num_steps)
        pitch_values = np.linspace(rotation_start, rotation_stop, num_steps)
        yaw_values = np.linspace(rotation_start, rotation_stop, num_steps)
        
        # Load statistics for the specified metric
        data = self.load_statistics(metric_name)
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        # Lists to store the points
        x_points = []
        y_points = []
        z_points = []
        colors = []
        sizes = []
        
        # Error-based metrics (0 is best) and accuracy-based metrics (1 is best)
        error_metrics = ['RMSE', 'MAE', 'iRMSE', 'iMAE', 'REL']
        accuracy_metrics = ['D1', 'D2', 'D3']
        
        # Custom colormap: Red (low) to White (high)
        red_to_white = LinearSegmentedColormap.from_list("red_to_white", ["red", "white"])
        
        # Fill the grid with the data values
        min_val = 0
        max_std = 0
        for roll, pitch, yaw in itertools.product(roll_values, pitch_values, yaw_values):
            values = data.get((roll, pitch, yaw), {})
            if not values:
                continue  # Skip if no data available for this combination
                
            mean_val = values.get('mean', None)
            min_val = min(min_val, mean_val)
            std_val = values.get('std', None)
            max_std = max(max_std, std_val)
            
            # Check if mean or std is None (missing data)
            if mean_val is None or std_val is None:
                continue  # Skip this entry if data is missing
                
            # Add points for scatter plot
            x_points.append(roll)
            y_points.append(pitch)
            z_points.append(yaw)
            
            # Determine color and size based on the metric
            colormap = plt.cm.viridis
            if metric_name in error_metrics:
                # Error-based metrics (0 is best)
                normalized_val = (-1 * (mean_val - min_val)/(2*max_std)) + 1  # Assuming metric is normalized (0 to 1)
                normalized_val = np.where(normalized_val < 0, 0, normalized_val)
                # color = colormap((-1 * (normalized_val - min_val)/(2*max_std)) + 1)  # Use the viridis colormap
                color = colormap(mean_val) # Use the viridis colormap
                size = max(100, 50 * (normalized_val))  # Larger scaling factor for size
            elif metric_name in accuracy_metrics:
                # Accuracy-based metrics (1 is best)
                normalized_val = mean_val  # Assuming metric is normalizded (0 to 1)
                color = colormap(normalized_val)  # Use the viridis colormap
                size = max(30, 300 * normalized_val)  # Larger scaling factor for size
            else:
                raise ValueError(f"Invalid metric name: {metric_name}")
            
            colors.append(color)
            sizes.append(size)
        
        # Scatter plot for the 3D points
        scatter = ax.scatter(x_points, y_points, z_points, c=[c[:3] for c in colors], s=sizes, depthshade=True)
            
        # Add color bar to the side
        fig.colorbar(scatter, ax=ax, label=metric_name)
        
        # Set labels
        ax.set_xlabel('Roll')
        ax.set_ylabel('Pitch')
        ax.set_zlabel('Yaw')
        
        plt.title(f"3D Scatter Plot of {metric_name} - Dataset: {dataset_name}, Model: LRRU_{model_size}")
        plt.show()


    def plot_2d_heatmaps(self, axis_zeroed, metric_name):
        """Plot 2D heatmaps for mean, std, and max metrics."""
        
        # Load statistics for the specified metric
        data = self.load_statistics(metric_name)

        # Parse the filename for dataset and model size
        dataset_name, model_size, rotation_start, rotation_stop, rotation_step = self.parse_filename()

        # Calculate roll, pitch, yaw values
        num_steps = int(np.ceil((rotation_stop - rotation_start) / rotation_step)) + 1
        roll_values = np.linspace(rotation_start, rotation_stop, num_steps)
        pitch_values = np.linspace(rotation_start, rotation_stop, num_steps)
        yaw_values = np.linspace(rotation_start, rotation_stop, num_steps)
        
        # Zeroing one of the axes to plot 2D heatmaps
        if axis_zeroed == 'roll':
            axis1_values, axis2_values = pitch_values, yaw_values
            axis1_name, axis2_name = 'pitch', 'yaw'
            axis_fixed = roll_values[0]  # Fix roll to the first value
        elif axis_zeroed == 'pitch':
            axis1_values, axis2_values = roll_values, yaw_values
            axis1_name, axis2_name = 'roll', 'yaw'
            axis_fixed = pitch_values[0]  # Fix pitch to the first value
        elif axis_zeroed == 'yaw':
            axis1_values, axis2_values = roll_values, pitch_values
            axis1_name, axis2_name = 'roll', 'pitch'
            axis_fixed = yaw_values[0]  # Fix yaw to the first value
        else:
            raise ValueError("Invalid axis_zeroed. Choose from 'roll', 'pitch', 'yaw'.")

        # Create matrices for the heatmaps
        mean_map = np.zeros((len(axis1_values), len(axis2_values)))
        std_map = np.zeros((len(axis1_values), len(axis2_values)))
        max_map = np.zeros((len(axis1_values), len(axis2_values)))

        for i, axis1 in enumerate(axis1_values):
            for j, axis2 in enumerate(axis2_values):
                if axis_zeroed == 'roll':
                    coord = (axis_fixed, axis1, axis2)  # (roll, pitch, yaw)
                elif axis_zeroed == 'pitch':
                    coord = (axis1, axis_fixed, axis2)  # (roll, pitch, yaw)
                elif axis_zeroed == 'yaw':
                    coord = (axis1, axis2, axis_fixed)  # (roll, pitch, yaw)

                # Get values for mean, std, and max from the data dictionary
                mean_map[i, j] = data.get(coord, {}).get('mean', np.nan)
                std_map[i, j] = data.get(coord, {}).get('std', np.nan)
                max_map[i, j] = data.get(coord, {}).get('max', np.nan)

        # Plot heatmaps
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        
        c1 = ax[0].imshow(mean_map, cmap='coolwarm', origin='lower')
        fig.colorbar(c1, ax=ax[0])
        ax[0].set_title(f"{metric_name} Mean")
        ax[0].set_xlabel(axis2_name)
        ax[0].set_ylabel(axis1_name)
        
        c2 = ax[1].imshow(std_map, cmap='coolwarm', origin='lower')
        fig.colorbar(c2, ax=ax[1])
        ax[1].set_title(f"{metric_name} Std")
        ax[1].set_xlabel(axis2_name)
        ax[1].set_ylabel(axis1_name)
        
        c3 = ax[2].imshow(max_map, cmap='coolwarm', origin='lower')
        fig.colorbar(c3, ax=ax[2])
        ax[2].set_title(f"{metric_name} Max")
        ax[2].set_xlabel(axis2_name)
        ax[2].set_ylabel(axis1_name)

        plt.show()

    def plot_1d_boxplot(self, axis_varied, metric_name):
        """Plot 1D boxplot or bar chart for a single axis."""
        
        # Parse the filename for dataset and model size
        dataset_name, model_size, rotation_start, rotation_stop, rotation_step = self.parse_filename()

        # Calculate roll, pitch, yaw values
        num_steps = int(np.ceil((rotation_stop - rotation_start) / rotation_step)) + 1
        roll_values = np.linspace(rotation_start, rotation_stop, num_steps)
        pitch_values = np.linspace(rotation_start, rotation_stop, num_steps)
        yaw_values = np.linspace(rotation_start, rotation_stop, num_steps)

        data = self.load_statistics(metric_name)
        
        values = []
        labels = []

        if axis_varied == 'roll':
            axis_values = roll_values
            fixed_pitch = pitch_values[0]  # Fix pitch to the first value
            fixed_yaw = yaw_values[0]      # Fix yaw to the first value
        elif axis_varied == 'pitch':
            axis_values = pitch_values
            fixed_roll = roll_values[0]    # Fix roll to the first value
            fixed_yaw = yaw_values[0]      # Fix yaw to the first value
        elif axis_varied == 'yaw':
            axis_values = yaw_values
            fixed_roll = roll_values[0]    # Fix roll to the first value
            fixed_pitch = pitch_values[0]  # Fix pitch to the first value
        else:
            raise ValueError("Invalid axis_varied. Choose from 'roll', 'pitch', 'yaw'.")

        for val in axis_values:
            if axis_varied == 'roll':
                coord = (val, fixed_pitch, fixed_yaw)  # (roll, pitch, yaw)
            elif axis_varied == 'pitch':
                coord = (fixed_roll, val, fixed_yaw)   # (roll, pitch, yaw)
            elif axis_varied == 'yaw':
                coord = (fixed_roll, fixed_pitch, val) # (roll, pitch, yaw)

            metric_val = data.get(coord, {}).get('mean', np.nan)
            values.append(metric_val)
            labels.append(val)

        # Plot boxplot or bar chart
        sns.barplot(x=labels, y=values)
        plt.title(f"1D Visualization of {metric_name} for {axis_varied}")
        plt.xlabel(axis_varied)
        plt.ylabel(f"{metric_name} (Mean)")
        plt.show()


# Example Usage
if __name__ == "__main__":
    
    # Define command-line arguments
    parser = argparse.ArgumentParser(description='3D Visualization of Evaluation Metrics')
    parser.add_argument('--db_path', type=str, required=True, help="Path to the SQLite database file")
    parser.add_argument('--metric', type=str, required=True, help="Evaluation metric to eval")
    args = parser.parse_args()
    db_path = args.db_path

    # Create an instance of the EvalVisualizer
    visualizer = EvaluationVisualizer(db_path)
    
    t0 = time.time()
    # Compute statistics (this will extract the rotation range from the filename)
    visualizer.compute_statistics()
    t1 = time.time()
    print(f"Time to compute statistics: {t1 - t0} seconds")
    
    # Plot 3D voxel grid for a specific metric
    # visualizer.plot_voxel_3d(metric_name=args.metric) # Error based
    # visualizer.plot_voxel_3d('D1') # Accuracy based
    visualizer.plot_2d_heatmaps(axis_zeroed = "roll", metric_name=args.metric)
    visualizer.plot_1d_boxplot(axis_varied = "yaw", metric_name=args.metric)

