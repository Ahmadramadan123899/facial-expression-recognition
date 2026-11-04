import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Dict, Any, Union


class MovementSampler:
    """
    A class for sampling frames based on movement patterns in landmark data.

    This sampler analyzes spatial movement between frames and selects the most
    dynamic/interesting frames based on velocity and acceleration patterns.
    """

    # Recommended hyperparameters for different use cases
    RECOMMENDED_HYPERPARAMETERS = {
        'diverse_peak': {
            'sampling_strategy': 'diverse_peaks',
            'method': 'euclidean',
            'aggregate': 'max',
            'window_size': 1,
            'include_velocity': True,
            'include_acceleration': True,
            'min_distance': 3
        },
        'top_k': {
            'sampling_strategy': 'top_k',
            'method': 'euclidean',
            'aggregate': 'max',
            'window_size': 1,
            'include_velocity': True,
            'include_acceleration': True,
            'min_distance': 3
        },
        'adaptive': {
            'sampling_strategy': 'adaptive',
            'method': 'euclidean',
            'aggregate': 'max',
            'window_size': 1,
            'include_velocity': True,
            'include_acceleration': True,
            'min_distance': 3
        }
    }

    def __init__(self,
                 sampling_strategy: str = 'diverse_peaks',
                 method: str = 'euclidean',
                 aggregate: str = 'mean',
                 window_size: int = 1,
                 include_velocity: bool = True,
                 include_acceleration: bool = True,
                 min_distance: int = 3):
        """
        Initialize the MovementSampler.

        Args:
            sampling_strategy: Strategy for sampling ('top_k', 'diverse_peaks', 'adaptive')
            method: Distance metric ('euclidean' or 'manhattan')
            aggregate: Aggregation method ('mean', 'max', 'sum')
            window_size: Size of temporal window for movement computation
            include_velocity: Whether to include velocity (1st derivative) in scoring
            include_acceleration: Whether to include acceleration (2nd derivative) in scoring
            min_distance: Minimum distance between selected frames (for diverse_peaks)
        """
        self.sampling_strategy = sampling_strategy
        self.method = method
        self.aggregate = aggregate
        self.window_size = window_size
        self.include_velocity = include_velocity
        self.include_acceleration = include_acceleration
        self.min_distance = min_distance

        # Validate parameters
        self._validate_parameters()

        # Storage for last computation results
        self.last_frame_scores = None
        self.last_selected_indices = None

    def _validate_parameters(self):
        """Validate initialization parameters."""
        valid_strategies = ['top_k', 'diverse_peaks', 'adaptive']
        if self.sampling_strategy not in valid_strategies:
            raise ValueError(f"sampling_strategy must be one of {valid_strategies}")

        valid_methods = ['euclidean', 'manhattan']
        if self.method not in valid_methods:
            raise ValueError(f"method must be one of {valid_methods}")

        valid_aggregates = ['mean', 'max', 'sum']
        if self.aggregate not in valid_aggregates:
            raise ValueError(f"aggregate must be one of {valid_aggregates}")

        if self.window_size < 1:
            raise ValueError("window_size must be >= 1")

        if self.min_distance < 1:
            raise ValueError("min_distance must be >= 1")

    @classmethod
    def from_preset(cls, preset: str) -> 'MovementSampler':
        """
        Create a MovementSampler instance from a preset configuration.

        Args:
            preset: Preset name ('general', 'high_frequency', 'smooth_sampling')

        Returns:
            MovementSampler instance with preset parameters
        """
        if preset not in cls.RECOMMENDED_HYPERPARAMETERS:
            raise ValueError(
                f"Unknown preset: {preset}. Available presets: {list(cls.RECOMMENDED_HYPERPARAMETERS.keys())}")

        params = cls.RECOMMENDED_HYPERPARAMETERS[preset]
        return cls(**params)

    def compute_frame_movement_scores(self, landmarks: torch.Tensor) -> torch.Tensor:
        """
        Compute movement scores for each frame considering multiple movement aspects.

        Args:
            landmarks: Tensor of shape (478, 3, T) where T is number of frames

        Returns:
            torch.Tensor: Movement scores for each frame, shape (T,)
        """
        if landmarks.dim() != 3:
            raise ValueError(f"Expected landmarks shape (N, 3, T), got {landmarks.shape}")

        xy_landmarks = landmarks[:, :2, :]  # Shape: (478, 2, T)
        num_frames = xy_landmarks.shape[2]
        frame_scores = torch.zeros(num_frames)

        # Compute velocity (1st derivative) if requested
        if self.include_velocity:
            velocity_scores = torch.zeros(num_frames)
            for t in range(self.window_size, num_frames - self.window_size):
                # Compute movement in a window around frame t
                movements = []
                for offset in range(-self.window_size, self.window_size + 1):
                    if offset == 0:
                        continue
                    current_frame = xy_landmarks[:, :, t]
                    comparison_frame = xy_landmarks[:, :, t + offset]
                    displacement = comparison_frame - current_frame

                    if self.method == 'euclidean':
                        distances = torch.norm(displacement, dim=1)
                    elif self.method == 'manhattan':
                        distances = torch.sum(torch.abs(displacement), dim=1)

                    if self.aggregate == 'mean':
                        movement = torch.mean(distances)
                    elif self.aggregate == 'max':
                        movement = torch.max(distances)
                    elif self.aggregate == 'sum':
                        movement = torch.sum(distances)

                    movements.append(movement)

                velocity_scores[t] = torch.mean(torch.tensor(movements))

            frame_scores += velocity_scores

        # Compute acceleration (2nd derivative) if requested
        if self.include_acceleration:
            acceleration_scores = torch.zeros(num_frames)
            for t in range(2, num_frames - 2):
                # Compute acceleration as change in velocity
                prev_vel = xy_landmarks[:, :, t] - xy_landmarks[:, :, t - 1]
                curr_vel = xy_landmarks[:, :, t + 1] - xy_landmarks[:, :, t]
                next_vel = xy_landmarks[:, :, t + 2] - xy_landmarks[:, :, t + 1]

                # Acceleration as second derivative
                accel1 = curr_vel - prev_vel
                accel2 = next_vel - curr_vel

                if self.method == 'euclidean':
                    accel_dist1 = torch.norm(accel1, dim=1)
                    accel_dist2 = torch.norm(accel2, dim=1)
                elif self.method == 'manhattan':
                    accel_dist1 = torch.sum(torch.abs(accel1), dim=1)
                    accel_dist2 = torch.sum(torch.abs(accel2), dim=1)

                if self.aggregate == 'mean':
                    accel_score = torch.mean(accel_dist1 + accel_dist2)
                elif self.aggregate == 'max':
                    accel_score = torch.max(accel_dist1 + accel_dist2)
                elif self.aggregate == 'sum':
                    accel_score = torch.sum(accel_dist1 + accel_dist2)

                acceleration_scores[t] = accel_score

            frame_scores += acceleration_scores * 0.5  # Weight acceleration less than velocity

        return frame_scores

    def sample(self,
               landmarks: torch.Tensor,
               nb_target_frames: int,
               return_scores: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Sample frames based on movement patterns.

        Args:
            landmarks: Tensor of shape (478, 3, T) where T is number of frames
            nb_target_frames: Number of frames to sample
            return_scores: Whether to return movement scores along with indices

        Returns:
            torch.Tensor: Indices of selected frames, optionally with scores
        """
        if landmarks.dim() != 3:
            raise ValueError(f"Expected 3D landmarks shape (N, C, T), got {landmarks.shape}")

        nb_frames = landmarks.shape[-1]
        if nb_frames <= nb_target_frames:
            indices = torch.arange(nb_frames)
            if return_scores:
                scores = self.compute_frame_movement_scores(landmarks)
                self.last_frame_scores = scores
                self.last_selected_indices = indices
                return indices, scores
            self.last_selected_indices = indices
            return indices

        # Compute movement scores for all frames
        frame_scores = self.compute_frame_movement_scores(landmarks)
        self.last_frame_scores = frame_scores

        if self.sampling_strategy == 'top_k':
            # Simply select top K frames
            _, top_indices = torch.topk(frame_scores, nb_target_frames, sorted=True)
            selected_indices = torch.sort(top_indices)[0]  # Sort to maintain temporal order

        elif self.sampling_strategy == 'diverse_peaks':
            # Select peaks with minimum distance constraint
            selected_indices = []
            remaining_scores = frame_scores.clone()

            for _ in range(nb_target_frames):
                # Find the frame with highest remaining score
                best_idx = torch.argmax(remaining_scores).item()

                # Check if this frame is already selected (shouldn't happen, but safety check)
                if best_idx in selected_indices:
                    # Find next best frame that's not selected
                    remaining_scores[best_idx] = -float('inf')
                    continue

                selected_indices.append(best_idx)

                # Suppress nearby frames to ensure diversity
                start_suppress = max(0, best_idx - self.min_distance)
                end_suppress = min(len(remaining_scores), best_idx + self.min_distance + 1)
                remaining_scores[start_suppress:end_suppress] = -float('inf')

                # Additional safety: ensure we don't run out of valid frames
                if torch.all(remaining_scores == -float('inf')) and len(selected_indices) < nb_target_frames:
                    # If we've suppressed all frames but still need more samples,
                    # fill with remaining frames in order
                    all_indices = set(range(nb_frames))
                    remaining_indices = sorted(all_indices - set(selected_indices))
                    selected_indices.extend(remaining_indices[:nb_target_frames - len(selected_indices)])
                    break

            selected_indices = torch.tensor(sorted(selected_indices))

        elif self.sampling_strategy == 'adaptive':
            # Adaptive sampling based on score distribution
            # Use a combination of high movement and temporal diversity

            # Normalize scores to [0, 1]
            normalized_scores = (frame_scores - frame_scores.min()) / (frame_scores.max() - frame_scores.min() + 1e-8)

            # Create a probability distribution that favors high-movement frames
            # but also maintains some temporal diversity
            probabilities = torch.softmax(normalized_scores * 3, dim=0)  # Temperature = 3

            # Sample without replacement
            selected_indices = torch.multinomial(probabilities, nb_target_frames, replacement=False)
            selected_indices = torch.sort(selected_indices)[0]

        self.last_selected_indices = selected_indices

        if return_scores:
            return selected_indices, frame_scores[selected_indices]

        return selected_indices

    def __call__(self, landmarks: torch.Tensor, num_samples: int, return_scores: bool = False) -> torch.Tensor:
        """
        Make the sampler callable. Same as sample() method.

        Args:
            landmarks: Tensor of shape (478, 3, T)
            num_samples: Number of frames to sample
            return_scores: Whether to return movement scores along with indices

        Returns:
            torch.Tensor: Indices of selected frames, optionally with scores
        """
        return self.sample(landmarks, num_samples, return_scores)

    def visualize_results(self,
                          landmarks: torch.Tensor,
                          selected_indices: Optional[torch.Tensor] = None,
                          title: Optional[str] = None):
        """
        Visualize the sampling results showing selected frames and their movement scores.

        Args:
            landmarks: Tensor of shape (478, 3, T)
            selected_indices: Indices of selected frames (uses last sampling if None)
            title: Plot title (auto-generated if None)
        """
        if selected_indices is None:
            if self.last_selected_indices is None:
                raise ValueError("No previous sampling results found. Run sample() first or provide selected_indices.")
            selected_indices = self.last_selected_indices

        if self.last_frame_scores is None:
            frame_scores = self.compute_frame_movement_scores(landmarks)
        else:
            frame_scores = self.last_frame_scores

        if title is None:
            title = f"Movement-Based Frame Sampling ({self.sampling_strategy})"

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

        # Plot 1: Movement scores with selected frames highlighted
        all_indices = torch.arange(len(frame_scores))
        ax1.plot(all_indices, frame_scores, 'b-', alpha=0.7, linewidth=1, label='All frames')
        ax1.scatter(selected_indices, frame_scores[selected_indices],
                    c='red', s=100, zorder=5, label='Selected frames')

        for i, idx in enumerate(selected_indices):
            ax1.annotate(f'F{idx}', (idx, frame_scores[idx]),
                         xytext=(5, 5), textcoords='offset points', fontsize=8)

        ax1.set_xlabel('Frame Index')
        ax1.set_ylabel('Movement Score')
        ax1.set_title(f'{title} - Movement Scores')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Sample landmarks trajectory for a few selected landmarks
        # Show trajectory of first 5 landmarks across selected frames
        colors = plt.cm.Set1(np.linspace(0, 1, 5))

        for landmark_idx in range(min(5, landmarks.shape[0])):
            x_coords = landmarks[landmark_idx, 0, selected_indices].cpu().numpy()
            y_coords = landmarks[landmark_idx, 1, selected_indices].cpu().numpy()

            ax2.plot(x_coords, y_coords, 'o-', color=colors[landmark_idx],
                     label=f'Landmark {landmark_idx}', markersize=6, alpha=0.8)

            # Add frame numbers
            for i, (x, y) in enumerate(zip(x_coords, y_coords)):
                ax2.annotate(f'{selected_indices[i]}', (x, y),
                             xytext=(3, 3), textcoords='offset points', fontsize=7)

        ax2.set_xlabel('X Coordinate')
        ax2.set_ylabel('Y Coordinate')
        ax2.set_title('Landmark Trajectories (Selected Frames)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the last sampling operation.

        Returns:
            Dict containing various statistics
        """
        if self.last_frame_scores is None or self.last_selected_indices is None:
            raise ValueError("No previous sampling results found. Run sample() first.")

        selected_scores = self.last_frame_scores[self.last_selected_indices]

        return {
            'num_frames_total': len(self.last_frame_scores),
            'num_frames_selected': len(self.last_selected_indices),
            'selected_indices': self.last_selected_indices.tolist(),
            'mean_movement_all': torch.mean(self.last_frame_scores).item(),
            'mean_movement_selected': torch.mean(selected_scores).item(),
            'max_movement_selected': torch.max(selected_scores).item(),
            'min_movement_selected': torch.min(selected_scores).item(),
            'total_movement_selected': torch.sum(selected_scores).item(),
            'improvement_ratio': torch.mean(selected_scores).item() / torch.mean(self.last_frame_scores).item()
        }

    def __repr__(self) -> str:
        """String representation of the MovementSampler."""
        return (f"MovementSampler(strategy='{self.sampling_strategy}', "
                f"method='{self.method}', aggregate='{self.aggregate}', "
                f"window_size={self.window_size}, min_distance={self.min_distance})")


# Example usage and demonstration
if __name__ == "__main__":
    # Create sample data with realistic movement patterns
    torch.manual_seed(42)
    sample_landmarks = torch.randn(478, 3, 40)

    # Add some interesting movement patterns
    for t in range(1, 40):
        # Add periodic movement with some frames having more activity
        if t in [35]:  # High activity frames
            sample_landmarks[:, :2, t] += torch.randn(478, 2) * 1.2
        else:  # Low activity frames
            sample_landmarks[:, :2, t] += torch.randn(478, 2) * 0.1

    # Example 1: Using preset configurations
    print("=== USING PRESET CONFIGURATIONS ===")

    for preset_name in MovementSampler.RECOMMENDED_HYPERPARAMETERS.keys():
        print(f"\nTesting {preset_name} preset:")
        sampler = MovementSampler.from_preset(preset_name)
        print(f"Sampler: {sampler}")

        # Sample 16 frames
        selected_indices, scores = sampler.sample(sample_landmarks, 8, return_scores=True)

        sampler.visualize_results(sample_landmarks)
        print(f"Selected frames: {selected_indices.tolist()}")
        print(f"Statistics: {sampler.get_statistics()}")

    # Example 2: Custom configuration
    print("\n=== CUSTOM CONFIGURATION ===")
    custom_sampler = MovementSampler(
        sampling_strategy='diverse_peaks',
        method='euclidean',
        aggregate='max',
        min_distance=3
    )

    # Use as callable
    selected_indices = custom_sampler(sample_landmarks, 8)
    print(f"Custom sampler selected: {selected_indices.tolist()}")

    # Visualize results
    custom_sampler.visualize_results(sample_landmarks)

    # Example 3: Comparison of strategies
    print("\n=== STRATEGY COMPARISON ===")
    strategies = ['top_k', 'diverse_peaks', 'adaptive']

    for strategy in strategies:
        sampler = MovementSampler(sampling_strategy=strategy)
        selected_indices = sampler.sample(sample_landmarks, 16)
        stats = sampler.get_statistics()
        print(f"{strategy:15} | Improvement: {stats['improvement_ratio']:.3f} | "
              f"Mean: {stats['mean_movement_selected']:.3f}")