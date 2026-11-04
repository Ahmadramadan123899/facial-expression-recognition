import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List
from scipy import interpolate
import torch.nn.functional as F


class MovementExtender:
    """Extends a landmark sequence to a target frame count via linear/spline
    interpolation, movement-pattern continuation, physics-based extrapolation,
    or noise-augmented interpolation."""

    def __init__(self,
                 extension_method: str = 'spline_interpolation',
                 noise_level: float = 0.01,
                 smoothing_factor: float = 0.0,
                 extrapolation_method: str = 'velocity',
                 preserve_boundaries: bool = True):
        """
        extension_method: 'linear_interpolation' | 'spline_interpolation' |
            'movement_pattern' | 'physics_based' | 'noise_augmented'
        extrapolation_method (used by movement_pattern/physics_based):
            'velocity' | 'acceleration' | 'periodic' | 'damped'
        preserve_boundaries: keep original timestamps unchanged in the output
        """
        self.extension_method = extension_method
        self.noise_level = noise_level
        self.smoothing_factor = smoothing_factor
        self.extrapolation_method = extrapolation_method
        self.preserve_boundaries = preserve_boundaries

        self._validate_parameters()

    def _validate_parameters(self):
        """Validate initialization parameters."""
        valid_methods = ['linear_interpolation', 'spline_interpolation', 'movement_pattern',
                         'physics_based', 'noise_augmented']
        if self.extension_method not in valid_methods:
            raise ValueError(f"extension_method must be one of {valid_methods}")

        valid_extrapolation = ['velocity', 'acceleration', 'periodic', 'damped']
        if self.extrapolation_method not in valid_extrapolation:
            raise ValueError(f"extrapolation_method must be one of {valid_extrapolation}")

        if self.noise_level < 0:
            raise ValueError("noise_level must be >= 0")

    def _linear_interpolation(self, landmarks: torch.Tensor, target_frames: int) -> torch.Tensor:
        """Linear interpolation between existing frames."""

        # Add batch dimension for F.interpolate
        landmarks_batched = landmarks.unsqueeze(0)  # Shape: (1, 478, 3, current_frames)

        # Reshape to combine landmark and coordinate dimensions
        # (1, 478, 3, T) -> (1, 478*3, T)
        B, L, C, T = landmarks_batched.shape
        landmarks_reshaped = landmarks_batched.view(B, L * C, T)

        # Use torch.nn.functional.interpolate for efficient interpolation
        extended_reshaped = F.interpolate(
            landmarks_reshaped,
            size=target_frames,
            mode='linear',
            align_corners=True
        )

        # Reshape back to original format
        # (1, 478*3, target_frames) -> (1, 478, 3, target_frames) -> (478, 3, target_frames)
        extended_landmarks = extended_reshaped.view(B, L, C, target_frames).squeeze(0)

        return extended_landmarks

    def _spline_interpolation(self, landmarks: torch.Tensor, target_frames: int) -> torch.Tensor:
        """Smooth spline interpolation between existing frames."""
        current_frames = landmarks.shape[2]

        # Create time indices
        original_times = np.linspace(0, 1, current_frames)
        target_times = np.linspace(0, 1, target_frames)

        extended_landmarks = torch.zeros(landmarks.shape[0], landmarks.shape[1], target_frames)

        for landmark_idx in range(landmarks.shape[0]):
            for coord_idx in range(landmarks.shape[1]):
                # Convert to numpy for scipy interpolation
                y_values = landmarks[landmark_idx, coord_idx].cpu().numpy()

                # Use cubic spline interpolation
                if current_frames >= 4:  # Need at least 4 points for cubic spline
                    spline = interpolate.UnivariateSpline(original_times, y_values,
                                                          s=self.smoothing_factor, k=3)
                elif current_frames >= 2:  # Linear spline for fewer points
                    spline = interpolate.UnivariateSpline(original_times, y_values,
                                                          s=self.smoothing_factor, k=1)
                else:
                    # Just repeat the single point
                    extended_landmarks[landmark_idx, coord_idx] = landmarks[landmark_idx, coord_idx, 0]
                    continue

                # Evaluate spline at target times
                interpolated = spline(target_times)
                extended_landmarks[landmark_idx, coord_idx] = torch.from_numpy(interpolated)

        return extended_landmarks

    def _movement_pattern_continuation(self, landmarks: torch.Tensor, target_frames: int) -> torch.Tensor:
        """Continue movement patterns by analyzing velocity and acceleration."""
        current_frames = landmarks.shape[2]

        if current_frames < 2:
            # Not enough frames to analyze movement, fall back to repetition
            return landmarks.repeat(1, 1, target_frames // current_frames + 1)[:, :, :target_frames]

        # Start with spline interpolation as base
        extended_landmarks = self._spline_interpolation(landmarks, target_frames)

        if target_frames <= current_frames:
            return extended_landmarks

        # Analyze movement patterns in the original sequence
        # Compute velocities and accelerations
        velocities = torch.diff(landmarks, dim=2)  # Shape: (478, 3, T-1)

        if current_frames >= 3:
            accelerations = torch.diff(velocities, dim=2)  # Shape: (478, 3, T-2)

            # Use the last few accelerations to predict future movement
            last_velocity = velocities[:, :, -1]
            last_acceleration = accelerations[:, :, -1] if accelerations.shape[2] > 0 else torch.zeros_like(
                last_velocity)

            # Extrapolate beyond the spline interpolation for additional frames
            for t in range(current_frames, target_frames):
                time_step = (t - current_frames + 1) / current_frames

                # Predict position using velocity and acceleration
                if self.extrapolation_method == 'velocity':
                    delta = last_velocity * time_step
                elif self.extrapolation_method == 'acceleration':
                    delta = last_velocity * time_step + 0.5 * last_acceleration * time_step ** 2
                elif self.extrapolation_method == 'damped':
                    # Exponentially damped motion
                    damping = torch.exp(torch.tensor(-time_step * 2))  # Damping factor
                    delta = last_velocity * time_step * damping
                else:  # periodic
                    # Assume periodic motion with period equal to sequence length
                    phase = 2 * np.pi * time_step
                    delta = last_velocity * torch.sin(torch.tensor(phase))

                # Update the extended landmarks
                extended_landmarks[:, :, t] = extended_landmarks[:, :, current_frames - 1] + delta

        return extended_landmarks

    def _physics_based_extension(self, landmarks: torch.Tensor, target_frames: int) -> torch.Tensor:
        """Physics-based extrapolation using velocity and acceleration."""
        current_frames = landmarks.shape[2]

        if target_frames <= current_frames:
            return self._spline_interpolation(landmarks, target_frames)

        # Start with original frames
        extended_landmarks = torch.zeros(landmarks.shape[0], landmarks.shape[1], target_frames)
        extended_landmarks[:, :, :current_frames] = landmarks

        if current_frames < 2:
            # Not enough data for physics, just repeat last frame
            extended_landmarks[:, :, current_frames:] = landmarks[:, :, -1:].repeat(1, 1,
                                                                                    target_frames - current_frames)
            return extended_landmarks

        # Compute velocities
        velocities = torch.diff(landmarks, dim=2)  # Shape: (478, 3, T-1)

        # Estimate current velocity (average of last few velocities)
        if velocities.shape[2] >= 3:
            current_velocity = torch.mean(velocities[:, :, -3:], dim=2)
        else:
            current_velocity = velocities[:, :, -1]

        # Compute accelerations if possible
        if current_frames >= 3:
            accelerations = torch.diff(velocities, dim=2)
            if accelerations.shape[2] >= 2:
                current_acceleration = torch.mean(accelerations[:, :, -2:], dim=2)
            else:
                current_acceleration = accelerations[:, :, -1]
        else:
            current_acceleration = torch.zeros_like(current_velocity)

        # Extrapolate future frames
        dt = 1.0  # Time step
        for t in range(current_frames, target_frames):
            time_elapsed = (t - current_frames + 1) * dt

            if self.extrapolation_method == 'velocity':
                # Simple linear extrapolation
                new_position = landmarks[:, :, -1] + current_velocity * time_elapsed
            elif self.extrapolation_method == 'acceleration':
                # Quadratic extrapolation with acceleration
                new_position = (landmarks[:, :, -1] +
                                current_velocity * time_elapsed +
                                0.5 * current_acceleration * time_elapsed ** 2)
            elif self.extrapolation_method == 'damped':
                # Exponentially damped motion
                damping_factor = torch.exp(torch.tensor(-time_elapsed) * 0.5)
                new_position = landmarks[:, :, -1] + current_velocity * time_elapsed * damping_factor
            else:  # periodic
                # Simple harmonic motion
                omega = 2 * np.pi / current_frames  # Angular frequency
                amplitude = torch.norm(current_velocity, dim=1, keepdim=True)
                new_position = landmarks[:, :, -1] + amplitude * torch.sin(torch.tensor(omega * time_elapsed))

            extended_landmarks[:, :, t] = new_position

        return extended_landmarks

    def _add_realistic_noise(self, landmarks: torch.Tensor) -> torch.Tensor:
        """Add realistic noise that respects the movement patterns."""
        if self.noise_level <= 0:
            return landmarks

        # Compute movement statistics to scale noise appropriately
        if landmarks.shape[2] > 1:
            movements = torch.diff(landmarks, dim=2)
            # Compute std per landmark and coordinate
            movement_std = torch.std(movements, dim=2, keepdim=True)  # Shape: (478, 3, 1)
            # Scale noise relative to typical movement magnitude
            noise_scale = self.noise_level * movement_std
        else:
            # Fallback if only one frame
            noise_scale = self.noise_level * torch.std(landmarks, dim=2, keepdim=True)

        # Generate correlated noise (smooth across time)
        noise = torch.randn_like(landmarks)

        # Apply temporal smoothing to make noise more realistic
        if landmarks.shape[2] > 3:
            # Use a simple moving average to smooth noise
            kernel_size = min(5, landmarks.shape[2] // 2)
            if kernel_size >= 3 and kernel_size % 2 == 1:  # Ensure odd kernel size
                # Create a smoothing kernel
                kernel = torch.ones(1, 1, kernel_size, device=landmarks.device) / kernel_size

                # Apply smoothing more efficiently
                for landmark_idx in range(landmarks.shape[0]):
                    for coord_idx in range(landmarks.shape[1]):
                        # Get 1D noise signal and reshape properly for conv1d
                        noise_1d = noise[landmark_idx, coord_idx].unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, T)
                        smoothed = F.conv1d(noise_1d, kernel, padding=kernel_size // 2)
                        noise[landmark_idx, coord_idx] = smoothed.squeeze()

        return landmarks + noise * noise_scale

    def extend(self, landmarks: torch.Tensor, nb_target_frames: int) -> torch.Tensor:
        """landmarks: (N, 3, T). Returns (N, 3, nb_target_frames)."""
        if landmarks.dim() != 3:
            raise ValueError(f"Expected landmarks shape (N, 3, T), got {landmarks.shape}")

        if landmarks.shape[-1] >= nb_target_frames:
            return landmarks

        current_frames = landmarks.shape[2]

        if nb_target_frames == current_frames:
            return landmarks.clone()

        # Apply the selected extension method
        if self.extension_method == 'linear_interpolation':
            extended = self._linear_interpolation(landmarks, nb_target_frames)
        elif self.extension_method == 'spline_interpolation':
            extended = self._spline_interpolation(landmarks, nb_target_frames)
        elif self.extension_method == 'movement_pattern':
            extended = self._movement_pattern_continuation(landmarks, nb_target_frames)
        elif self.extension_method == 'physics_based':
            extended = self._physics_based_extension(landmarks, nb_target_frames)
        elif self.extension_method == 'noise_augmented':
            extended = self._spline_interpolation(landmarks, nb_target_frames)
            extended = self._add_realistic_noise(extended)
        else:
            raise ValueError(f"Unknown extension method: {self.extension_method}")

        extended = extended.to(landmarks.device)

        # Preserve original timestamps if requested
        if self.preserve_boundaries and nb_target_frames >= current_frames:
            # Map original frame indices to new frame indices
            original_indices = torch.linspace(0, nb_target_frames - 1, current_frames, device=landmarks.device, dtype=torch.long)
            extended[:, :, original_indices] = landmarks

        # Add noise if specified (for all methods except noise_augmented which already has it)
        if self.extension_method != 'noise_augmented' and self.noise_level > 0:
            extended = self._add_realistic_noise(extended)

        return extended

    def __call__(self, landmarks: torch.Tensor, target_frames: int) -> torch.Tensor:
        """Make the extender callable."""
        return self.extend(landmarks, target_frames)

    def visualize_extension(self,
                            original: torch.Tensor,
                            extended: torch.Tensor,
                            landmark_indices: List[int] = None,
                            title: str = "Landmark Extension"):
        """Plot original vs extended trajectories for landmark_indices (default: first 3)."""
        original = original.cpu()
        extended = extended.cpu()

        if landmark_indices is None:
            landmark_indices = [0, 1, 2]

        fig, axes = plt.subplots(len(landmark_indices), 2, figsize=(15, 4 * len(landmark_indices)))
        if len(landmark_indices) == 1:
            axes = axes.reshape(1, -1)

        original_frames = original.shape[2]
        extended_frames = extended.shape[2]

        original_times = torch.arange(original_frames)
        extended_times = torch.arange(extended_frames)

        for i, landmark_idx in enumerate(landmark_indices):
            # X coordinate
            axes[i, 0].plot(original_times, original[landmark_idx, 0], 'bo-',
                            label='Original', markersize=8, linewidth=2)
            axes[i, 0].plot(extended_times, extended[landmark_idx, 0], 'r.-',
                            label='Extended', alpha=0.7)
            axes[i, 0].axvline(x=original_frames - 1, color='gray', linestyle='--', alpha=0.5)
            axes[i, 0].set_title(f'Landmark {landmark_idx} - X Coordinate')
            axes[i, 0].set_xlabel('Frame')
            axes[i, 0].set_ylabel('X Position')
            axes[i, 0].legend()
            axes[i, 0].grid(True, alpha=0.3)

            # Y coordinate
            axes[i, 1].plot(original_times, original[landmark_idx, 1], 'bo-',
                            label='Original', markersize=8, linewidth=2)
            axes[i, 1].plot(extended_times, extended[landmark_idx, 1], 'r.-',
                            label='Extended', alpha=0.7)
            axes[i, 1].axvline(x=original_frames - 1, color='gray', linestyle='--', alpha=0.5)
            axes[i, 1].set_title(f'Landmark {landmark_idx} - Y Coordinate')
            axes[i, 1].set_xlabel('Frame')
            axes[i, 1].set_ylabel('Y Position')
            axes[i, 1].legend()
            axes[i, 1].grid(True, alpha=0.3)

        plt.suptitle(f'{title} ({self.extension_method})', fontsize=16)
        plt.tight_layout()
        plt.show()

    def __repr__(self) -> str:
        """String representation of the MovementExtender."""
        return (f"MovementExtender(method='{self.extension_method}', "
                f"noise_level={self.noise_level}, "
                f"extrapolation='{self.extrapolation_method}')")


# Example usage and demonstration
if __name__ == "__main__":
    # Create sample data
    torch.manual_seed(42)

    # Create a short sequence with 11 frames
    original_landmarks = torch.randn(478, 3, 11)

    # Add some realistic movement patterns
    for t in range(1, 11):
        # Add some smooth movement
        original_landmarks[:, :2, t] = (original_landmarks[:, :2, t - 1] +
                                        torch.randn(478, 2) * 0.1 +
                                        torch.sin(torch.tensor(t * 0.5)) * 0.05)

    # Test different extension methods
    methods = ['linear_interpolation', 'spline_interpolation', 'movement_pattern',
               'physics_based', 'noise_augmented']

    target_frames = 16  # Extend from 11 to 16 frames

    print(f"Extending from {original_landmarks.shape[2]} to {target_frames} frames")
    print("=" * 60)

    # Store extended results for comparison
    extended_results = {}

    for method in methods:
        print(f"\nTesting {method}:")

        # Create extender
        extender = MovementExtender(
            extension_method=method,
            noise_level=0.02,
            extrapolation_method='velocity'
        )

        # Extend the sequence
        extended_landmarks = extender.extend(original_landmarks, target_frames)
        extended_results[method] = extended_landmarks

        print(f"Original shape: {original_landmarks.shape}")
        print(f"Extended shape: {extended_landmarks.shape}")

    # Visualization 1: Compare all methods for specific landmarks
    print("\n" + "=" * 60)
    print("VISUALIZATION 1: Comparing all methods")
    print("=" * 60)

    landmark_indices = [0, 100, 200]  # Different landmarks to visualize
    fig, axes = plt.subplots(len(landmark_indices), 2, figsize=(20, 5 * len(landmark_indices)))

    original_times = torch.arange(original_landmarks.shape[2])
    extended_times = torch.arange(target_frames)

    colors = {'linear_interpolation': 'red',
              'spline_interpolation': 'green',
              'movement_pattern': 'blue',
              'physics_based': 'purple',
              'noise_augmented': 'orange'}

    for i, landmark_idx in enumerate(landmark_indices):
        # X coordinate
        axes[i, 0].plot(original_times, original_landmarks[landmark_idx, 0], 'ko-',
                        label='Original', markersize=8, linewidth=2, zorder=10)

        for method, extended in extended_results.items():
            axes[i, 0].plot(extended_times, extended[landmark_idx, 0],
                            color=colors[method], linestyle='--', alpha=0.7,
                            label=method.replace('_', ' ').title(), linewidth=1.5)

        axes[i, 0].axvline(x=original_landmarks.shape[2] - 1, color='gray',
                           linestyle=':', alpha=0.5)
        axes[i, 0].set_title(f'Landmark {landmark_idx} - X Coordinate')
        axes[i, 0].set_xlabel('Frame')
        axes[i, 0].set_ylabel('X Position')
        axes[i, 0].legend(loc='best', fontsize=8)
        axes[i, 0].grid(True, alpha=0.3)

        # Y coordinate
        axes[i, 1].plot(original_times, original_landmarks[landmark_idx, 1], 'ko-',
                        label='Original', markersize=8, linewidth=2, zorder=10)

        for method, extended in extended_results.items():
            axes[i, 1].plot(extended_times, extended[landmark_idx, 1],
                            color=colors[method], linestyle='--', alpha=0.7,
                            label=method.replace('_', ' ').title(), linewidth=1.5)

        axes[i, 1].axvline(x=original_landmarks.shape[2] - 1, color='gray',
                           linestyle=':', alpha=0.5)
        axes[i, 1].set_title(f'Landmark {landmark_idx} - Y Coordinate')
        axes[i, 1].set_xlabel('Frame')
        axes[i, 1].set_ylabel('Y Position')
        axes[i, 1].legend(loc='best', fontsize=8)
        axes[i, 1].grid(True, alpha=0.3)

    plt.suptitle('Comparison of All Extension Methods', fontsize=16)
    plt.tight_layout()
    plt.show()

    # Visualization 2: Individual method details
    print("\n" + "=" * 60)
    print("VISUALIZATION 2: Individual method details")
    print("=" * 60)

    # Show detailed visualization for each method
    # for method in ['spline_interpolation', 'movement_pattern', 'physics_based']:
    for method in methods:
        extender = MovementExtender(extension_method=method, noise_level=0.02)
        extended = extender.extend(original_landmarks, target_frames)
        extender.visualize_extension(original_landmarks, extended,
                                     landmark_indices=[0, 1],
                                     title=f"{method.replace('_', ' ').title()} Method")

    # Visualization 3: Different extrapolation methods comparison
    print("\n" + "=" * 60)
    print("VISUALIZATION 3: Comparing extrapolation methods")
    print("=" * 60)

    # Test with longer extension to see extrapolation differences
    long_target = 30  # Extend much more to see extrapolation behavior

    extrapolation_methods = ['velocity', 'acceleration', 'damped', 'periodic']
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()

    landmark_to_plot = 0  # Pick one landmark for clarity

    for idx, extrap_method in enumerate(extrapolation_methods):
        extender = MovementExtender(
            extension_method='physics_based',
            extrapolation_method=extrap_method,
            noise_level=0.0  # No noise for clearer comparison
        )

        extended = extender.extend(original_landmarks, long_target)

        original_times = torch.arange(original_landmarks.shape[2])
        extended_times = torch.arange(long_target)

        # Plot both X and Y on same plot
        axes[idx].plot(original_times, original_landmarks[landmark_to_plot, 0],
                       'bo-', label='Original X', markersize=6)
        axes[idx].plot(original_times, original_landmarks[landmark_to_plot, 1],
                       'ro-', label='Original Y', markersize=6)

        axes[idx].plot(extended_times, extended[landmark_to_plot, 0],
                       'b--', alpha=0.7, label='Extended X')
        axes[idx].plot(extended_times, extended[landmark_to_plot, 1],
                       'r--', alpha=0.7, label='Extended Y')

        axes[idx].axvline(x=original_landmarks.shape[2] - 1,
                          color='gray', linestyle=':', alpha=0.5)
        axes[idx].set_title(f'Extrapolation: {extrap_method}')
        axes[idx].set_xlabel('Frame')
        axes[idx].set_ylabel('Position')
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)

    plt.suptitle('Comparison of Extrapolation Methods (Physics-based)', fontsize=16)
    plt.tight_layout()
    plt.show()

    # Visualization 4: Effect of noise levels
    print("\n" + "=" * 60)
    print("VISUALIZATION 4: Effect of noise levels")
    print("=" * 60)

    noise_levels = [0.0, 0.01, 0.05, 0.1]
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()

    landmark_to_plot = 50

    for idx, noise in enumerate(noise_levels):
        extender = MovementExtender(
            extension_method='spline_interpolation',
            noise_level=noise
        )

        extended = extender.extend(original_landmarks, target_frames)

        original_times = torch.arange(original_landmarks.shape[2])
        extended_times = torch.arange(target_frames)

        axes[idx].plot(original_times, original_landmarks[landmark_to_plot, 0],
                       'bo-', label='Original', markersize=8, linewidth=2)
        axes[idx].plot(extended_times, extended[landmark_to_plot, 0],
                       'r.-', alpha=0.7, label='Extended')

        axes[idx].axvline(x=original_landmarks.shape[2] - 1,
                          color='gray', linestyle=':', alpha=0.5)
        axes[idx].set_title(f'Noise Level: {noise}')
        axes[idx].set_xlabel('Frame')
        axes[idx].set_ylabel('X Position')
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)

    plt.suptitle('Effect of Noise Levels on Spline Interpolation', fontsize=16)
    plt.tight_layout()
    plt.show()


    # Example: Combining with MovementSampler
    print("\n" + "=" * 60)
    print("COMBINING EXTENDER WITH SAMPLER")
    print("=" * 60)

    # First extend the sequence
    extender = MovementExtender(extension_method='spline_interpolation')
    extended_landmarks = extender(original_landmarks, 40)  # Extend to 40 frames

    print(f"Extended to: {extended_landmarks.shape}")

    try:
        from Components.Sampler.LandmarksSampler.MovementSampler import MovementSampler

        # Create a sampler instance
        sampler = MovementSampler.from_preset('general')
        # Sample movements from the extended landmarks
        selected_indices, scores = sampler(extended_landmarks, num_samples=10, return_scores=True)

        print(f"Selected indices: {selected_indices}")
        print(f"Scores: {scores}")

        # Extract sampled landmarks
        sampled_landmarks = extended_landmarks[:, :, selected_indices]

        # Visualization 5: Extended sequence visualization
        print("\n" + "=" * 60)
        print("VISUALIZATION 5: Extended sequence (40 frames)")
        print("=" * 60)

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()

        # Select diverse landmarks to visualize
        landmarks_to_plot = [0, 100, 200, 300]

        for idx, landmark_idx in enumerate(landmarks_to_plot):
            # Plot original and extended
            original_times = torch.arange(original_landmarks.shape[2])
            extended_times = torch.arange(extended_landmarks.shape[2])

            axes[idx].plot(original_times, original_landmarks[landmark_idx, 0],
                           'bo-', label='Original X', markersize=6, linewidth=2)
            axes[idx].plot(original_times, original_landmarks[landmark_idx, 1],
                           'ro-', label='Original Y', markersize=6, linewidth=2)

            axes[idx].plot(extended_times, extended_landmarks[landmark_idx, 0],
                           'b--', alpha=0.5, label='Extended X', linewidth=1)
            axes[idx].plot(extended_times, extended_landmarks[landmark_idx, 1],
                           'r--', alpha=0.5, label='Extended Y', linewidth=1)

            axes[idx].axvline(x=original_landmarks.shape[2] - 1,
                              color='gray', linestyle=':', alpha=0.5,
                              label='Original end')
            axes[idx].set_title(f'Landmark {landmark_idx} - Extended to 40 frames')
            axes[idx].set_xlabel('Frame')
            axes[idx].set_ylabel('Position')
            axes[idx].legend(fontsize=8)
            axes[idx].grid(True, alpha=0.3)

        plt.suptitle('Extended Sequence Visualization (Movement Pattern Method)', fontsize=16)
        plt.tight_layout()
        plt.show()

        # Visualization 6: Sampled frames visualization
        print("\n" + "=" * 60)
        print("VISUALIZATION 6: Sampled frames from extended sequence")
        print("=" * 60)

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()

        for idx, landmark_idx in enumerate(landmarks_to_plot):
            extended_times = torch.arange(extended_landmarks.shape[2])

            # Plot full extended sequence
            axes[idx].plot(extended_times, extended_landmarks[landmark_idx, 0],
                           'b-', alpha=0.3, label='Extended X', linewidth=1)
            axes[idx].plot(extended_times, extended_landmarks[landmark_idx, 1],
                           'r-', alpha=0.3, label='Extended Y', linewidth=1)

            # Highlight sampled frames
            axes[idx].scatter(selected_indices, extended_landmarks[landmark_idx, 0, selected_indices],
                              color='blue', s=100, zorder=5, label='Sampled X', edgecolor='black')
            axes[idx].scatter(selected_indices, extended_landmarks[landmark_idx, 1, selected_indices],
                              color='red', s=100, zorder=5, label='Sampled Y', edgecolor='black')

            # Add scores as text annotations for first subplot only
            if idx == 0:
                for i, (frame_idx, score) in enumerate(zip(selected_indices[:5], scores[:5])):
                    axes[idx].annotate(f'{score:.2f}',
                                       xy=(frame_idx, extended_landmarks[landmark_idx, 0, frame_idx]),
                                       xytext=(5, 5), textcoords='offset points', fontsize=8)

            axes[idx].set_title(f'Landmark {landmark_idx} - Sampled Frames')
            axes[idx].set_xlabel('Frame')
            axes[idx].set_ylabel('Position')
            axes[idx].legend(fontsize=8)
            axes[idx].grid(True, alpha=0.3)

        plt.suptitle('Sampled Frames from Extended Sequence', fontsize=16)
        plt.tight_layout()
        plt.show()

        # Visualization 7: Comparison - Original vs Extended+Sampled
        print("\n" + "=" * 60)
        print("VISUALIZATION 7: Original vs Extended+Sampled comparison")
        print("=" * 60)

        fig, axes = plt.subplots(3, 2, figsize=(15, 12))

        landmark_idx = 50  # Pick one landmark for detailed comparison

        # Row 1: Original sequence
        original_times = torch.arange(original_landmarks.shape[2])
        axes[0, 0].plot(original_times, original_landmarks[landmark_idx, 0],
                        'bo-', markersize=8, linewidth=2)
        axes[0, 0].set_title(f'Original X - {original_landmarks.shape[2]} frames')
        axes[0, 0].set_ylabel('X Position')
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].plot(original_times, original_landmarks[landmark_idx, 1],
                        'ro-', markersize=8, linewidth=2)
        axes[0, 1].set_title(f'Original Y - {original_landmarks.shape[2]} frames')
        axes[0, 1].set_ylabel('Y Position')
        axes[0, 1].grid(True, alpha=0.3)

        # Row 2: Extended sequence
        extended_times = torch.arange(extended_landmarks.shape[2])
        axes[1, 0].plot(extended_times, extended_landmarks[landmark_idx, 0],
                        'b-', alpha=0.7, linewidth=1.5)
        axes[1, 0].axvline(x=original_landmarks.shape[2] - 1,
                           color='gray', linestyle=':', alpha=0.5)
        axes[1, 0].set_title(f'Extended X - {extended_landmarks.shape[2]} frames')
        axes[1, 0].set_ylabel('X Position')
        axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].plot(extended_times, extended_landmarks[landmark_idx, 1],
                        'r-', alpha=0.7, linewidth=1.5)
        axes[1, 1].axvline(x=original_landmarks.shape[2] - 1,
                           color='gray', linestyle=':', alpha=0.5)
        axes[1, 1].set_title(f'Extended Y - {extended_landmarks.shape[2]} frames')
        axes[1, 1].set_ylabel('Y Position')
        axes[1, 1].grid(True, alpha=0.3)

        # Row 3: Sampled sequence
        sampled_times = torch.arange(len(selected_indices))
        axes[2, 0].plot(sampled_times, sampled_landmarks[landmark_idx, 0],
                        'bo-', markersize=8, linewidth=2)
        # Add frame indices as x-tick labels
        axes[2, 0].set_xticks(sampled_times)
        axes[2, 0].set_xticklabels(selected_indices.tolist(), rotation=45)
        axes[2, 0].set_title(f'Sampled X - {len(selected_indices)} frames')
        axes[2, 0].set_xlabel('Original Frame Index')
        axes[2, 0].set_ylabel('X Position')
        axes[2, 0].grid(True, alpha=0.3)

        axes[2, 1].plot(sampled_times, sampled_landmarks[landmark_idx, 1],
                        'ro-', markersize=8, linewidth=2)
        axes[2, 1].set_xticks(sampled_times)
        axes[2, 1].set_xticklabels(selected_indices.tolist(), rotation=45)
        axes[2, 1].set_title(f'Sampled Y - {len(selected_indices)} frames')
        axes[2, 1].set_xlabel('Original Frame Index')
        axes[2, 1].set_ylabel('Y Position')
        axes[2, 1].grid(True, alpha=0.3)

        plt.suptitle(f'Pipeline Comparison: Original ({original_landmarks.shape[2]}) → '
                     f'Extended ({extended_landmarks.shape[2]}) → Sampled ({len(selected_indices)})',
                     fontsize=16)
        plt.tight_layout()
        plt.show()

        # Visualization 8: Movement magnitude comparison
        print("\n" + "=" * 60)
        print("VISUALIZATION 8: Movement magnitude analysis")
        print("=" * 60)

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=False)

        # Calculate movement magnitudes
        if original_landmarks.shape[2] > 1:
            orig_movements = torch.diff(original_landmarks, dim=2)
            orig_magnitudes = torch.norm(orig_movements, dim=1).mean(dim=0)  # Average across landmarks
            ax1.bar(range(len(orig_magnitudes)), orig_magnitudes, color='blue', alpha=0.7)
            ax1.set_title('Original Sequence - Movement Magnitudes')
            ax1.set_ylabel('Avg Movement')
            ax1.set_xlabel('Frame Transition')
            ax1.grid(True, alpha=0.3)

        if extended_landmarks.shape[2] > 1:
            ext_movements = torch.diff(extended_landmarks, dim=2)
            ext_magnitudes = torch.norm(ext_movements, dim=1).mean(dim=0)
            ax2.bar(range(len(ext_magnitudes)), ext_magnitudes, color='green', alpha=0.7)
            ax2.axvline(x=original_landmarks.shape[2] - 2, color='red',
                        linestyle=':', alpha=0.5, label='Original end')
            ax2.set_title('Extended Sequence - Movement Magnitudes')
            ax2.set_ylabel('Avg Movement')
            ax2.set_xlabel('Frame Transition')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

        if sampled_landmarks.shape[2] > 1:
            samp_movements = torch.diff(sampled_landmarks, dim=2)
            samp_magnitudes = torch.norm(samp_movements, dim=1).mean(dim=0)
            ax3.bar(range(len(samp_magnitudes)), samp_magnitudes, color='orange', alpha=0.7)
            ax3.set_title('Sampled Sequence - Movement Magnitudes')
            ax3.set_ylabel('Avg Movement')
            ax3.set_xlabel('Frame Transition')
            ax3.grid(True, alpha=0.3)

        plt.suptitle('Movement Magnitude Analysis Through Pipeline', fontsize=16)
        plt.tight_layout()
        plt.show()

    except ImportError:
        print("MovementSampler not found. Skipping sampler-related visualizations.")
        print("To use the sampler functionality, ensure MovementSampler is available.")


    # Compute some statistics
    print("\n" + "=" * 60)
    print("STATISTICS COMPARISON")
    print("=" * 60)

    for method, extended in extended_results.items():
        # Compute average movement magnitude
        if extended.shape[2] > 1:
            movements = torch.diff(extended, dim=2)
            avg_movement = torch.mean(torch.abs(movements))
            std_movement = torch.std(movements)

            print(f"\n{method}:")
            print(f"  Average movement: {avg_movement:.4f}")
            print(f"  Std of movement: {std_movement:.4f}")

            # Check smoothness (second derivative)
            if extended.shape[2] > 2:
                second_diff = torch.diff(movements, dim=2)
                smoothness = torch.mean(torch.abs(second_diff))
                print(f"  Smoothness (lower is smoother): {smoothness:.4f}")