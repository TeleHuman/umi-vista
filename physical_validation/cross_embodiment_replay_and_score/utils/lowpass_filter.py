# Low-pass filtering utilities, two variants.

import numpy as np
from scipy.signal import butter, lfilter, lfilter_zi

class RealTimeButterworthFilter:
    """
    Real-time second-order Butterworth filter.
    Suitable for filtering one sample at a time in a control loop.
    """
    
    def __init__(self, lowcut=None, highcut=None, fs=1000, btype='low', order=2, initial_value=0.0):
        """
        Initialize the filter.
        
        Args:
        lowcut: low cutoff frequency (Hz), for low-pass and high-pass filters
        highcut: high cutoff frequency (Hz), for band-pass and band-stop filters
        fs: sampling frequency (Hz)
        btype: filter type ('low', 'high', 'band', 'stop')
        order: filter order (default: 2)
        initial_value: initial value used to reduce startup transient response
        """
        self.fs = fs
        self.btype = btype
        self.order = order
        
        # Set cutoff frequencies according to filter type
        if btype == 'low':
            self.cutoff = lowcut
        elif btype == 'high':
            self.cutoff = lowcut
        elif btype in ['band', 'stop']:
            self.cutoff = [lowcut, highcut]
        else:
            raise ValueError("Filter type must be 'low', 'high', 'band', or 'stop'")
        
        # Design filter coefficients
        self.b, self.a = butter(self.order, self.cutoff, btype=btype, fs=fs)
        
        # Optimize initial state to reduce startup transient
        self.initial_value = initial_value
        self.zi = self._compute_initial_condition(initial_value)
        
        # Whether at least one sample has been processed
        self.is_initialized = False
        
        # Keep history for manual filtering path
        self.x_hist = np.zeros(len(self.b))
        self.y_hist = np.zeros(len(self.a) - 1)
    
    def _compute_initial_condition(self, initial_value):
        """Compute optimized initial conditions."""
        # Use SciPy to compute steady-state initial conditions
        zi = lfilter_zi(self.b, self.a)
        # Scale initial conditions by the provided initial value
        zi = zi * initial_value
        return zi
    
    def reset(self, initial_value=None):
        """Reset filter state."""
        if initial_value is not None:
            self.initial_value = initial_value
        
        self.zi = self._compute_initial_condition(self.initial_value)
        self.is_initialized = False
        
        # Reset history buffers
        self.x_hist = np.zeros(len(self.b))
        self.y_hist = np.zeros(len(self.a) - 1)
    
    def filter_step(self, x_new):
        """
        Single-step filtering with optimized initialization.
        
        Args:
        x_new: new input sample
        
        Returns:
        y_new: filtered output sample
        """
        # On first run, use the optimized initial condition
        if not self.is_initialized:
            # For the first sample, pre-warm using the configured initial value
            self.is_initialized = True
        
        # Use scipy lfilter and keep state continuity
        y_new, self.zi = lfilter(self.b, self.a, [x_new], zi=self.zi)
        
        return y_new[0]
    
    def filter_with_warmup(self, data, warmup_samples=10):
        """
        Filtering with a warmup phase, useful for initial segments.
        
        Args:
        data: input data array
        warmup_samples: number of warmup samples
        
        Returns:
        filtered_data: filtered data array
        """
        if len(data) == 0:
            return np.array([])
        
        # Save current state
        original_zi = self.zi.copy()
        original_initialized = self.is_initialized
        
        # Reset to initial value
        self.reset(initial_value=data[0] if len(data) > 0 else 0.0)
        
        # Warmup phase
        warmup_data = np.full(warmup_samples, data[0])
        for sample in warmup_data:
            self.filter_step(sample)
        
        # Filter actual data
        filtered_data = []
        for sample in data:
            filtered_sample = self.filter_step(sample)
            filtered_data.append(filtered_sample)
        
        # Restore original state
        self.zi = original_zi
        self.is_initialized = original_initialized
        
        return np.array(filtered_data)
    
    def _manual_filter_with_initialization(self, x_new):
        """
        Manually implemented filtering path with optimized initialization.
        """
        if not self.is_initialized:
            # Initialize history buffers
            self.x_hist.fill(x_new)
            self.y_hist.fill(x_new)
            self.is_initialized = True
        
        # Update input history
        self.x_hist = np.roll(self.x_hist, 1)
        self.x_hist[0] = x_new
        
        # Compute output
        y_new = (np.dot(self.b, self.x_hist) - np.dot(self.a[1:], self.y_hist)) / self.a[0]
        
        # Update output history
        self.y_hist = np.roll(self.y_hist, 1)
        self.y_hist[0] = y_new
        
        return y_new
    
    def get_filter_info(self):
        """Get filter metadata."""
        info = {
            'type': f'{self.order}-order Butterworth {self.btype} filter',
            'sampling_frequency': self.fs,
            'cutoff_frequencies': self.cutoff,
            'numerator_coeffs': self.b,
            'denominator_coeffs': self.a,
            'initial_value': self.initial_value,
            'is_initialized': self.is_initialized
        }
        return info

# ---------- Low-pass + rate limit (6D) ----------
import utils.self_math as smath
class PoseFilterLimiter:
    def __init__(self, dt, cutoff_hz, max_step_xyz, max_step_rot, ws_min, ws_max):
        self.dt   = float(dt)
        self.tau  = 1.0/(2*math.pi*max(1e-3, cutoff_hz))
        self.alpha= self.dt/(self.tau + self.dt)
        self.prev = None
        self.max_xyz = float(max_step_xyz)
        self.max_rot = float(max_step_rot)
        self.ws_min  = np.array(ws_min, dtype=float)
        self.ws_max  = np.array(ws_max, dtype=float)
    def clamp_step(self, prev6, target6):
        p0 = np.array(prev6[:3]); p1 = np.array(target6[:3])
        dp = p1 - p0
        n = np.linalg.norm(dp)
        if n > self.max_xyz > 0:
            dp = dp * (self.max_xyz / n); p1 = p0 + dp
        R0 = smath.rvec_to_R(prev6[3:]); R1 = smath.rvec_to_R(target6[3:])
        dR = R1 @ R0.T
        rv = smath.R_to_rvec(dR); ang = np.linalg.norm(rv)
        if ang > self.max_rot > 0:
            rv = rv * (self.max_rot / ang)
        R1_limited = smath.rvec_to_R(rv) @ R0
        r1 = smath.R_to_rvec(R1_limited)
        return [p1[0], p1[1], p1[2], r1[0], r1[1], r1[2]]
    def lpf(self, curr6):
        if self.prev is None: self.prev = curr6; return curr6
        a = self.alpha
        out = [ (1-a)*self.prev[i] + a*curr6[i] for i in range(6) ]
        self.prev = out; return out
    def clamp_workspace(self, pose6):
        p = np.array(pose6[:3])
        p = np.minimum(np.maximum(p, self.ws_min), self.ws_max)
        return [p[0], p[1], p[2], pose6[3], pose6[4], pose6[5]]
    def process(self, target6):
        if self.prev is not None:
            target6 = self.clamp_step(self.prev, target6)
        target6 = self.lpf(target6)
        target6 = self.clamp_workspace(target6)
        self.prev = target6
        return target6

# Usage example
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from pathlib import Path
    _HERE = Path(__file__).parent
    _TXT_L = _HERE / "trajectory" / "merged_trajectory_right4.txt"
    
    # Signal containing both low- and high-frequency components
    traj_data = np.loadtxt(_TXT_L.as_posix())
    signal = traj_data[:,1]  # Use one trajectory column as test signal
    fs = 100  # Sampling frequency 100 Hz
    t = np.arange(len(signal)) / fs
    
    # Create a low-pass filter (cutoff frequency 10 Hz)
    lowpass_filter = RealTimeButterworthFilter(lowcut=10, fs=fs, btype='low')
    lowpass_filter.reset(initial_value=signal[0])
    
    # Real-time filtering simulation
    filtered_signal = []
    for i, sample in enumerate(signal):
        filtered_sample = lowpass_filter.filter_step(sample)
        filtered_signal.append(filtered_sample)
    
    filtered_signal = np.array(filtered_signal)
    
    # Plot results
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(t, signal, label='Raw signal')
    plt.plot(t, filtered_signal, label='Filtered signal', linewidth=2)
    plt.title('Real-time Butterworth filter effect')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    # Plot frequency response
    from scipy.signal import freqz
    w, h = freqz(lowpass_filter.b, lowpass_filter.a, fs=fs)
    plt.semilogx(w, 20 * np.log10(np.abs(h)))
    plt.axvline(20, color='red', linestyle='--', label='Cutoff frequency 20 Hz')
    plt.title('Filter frequency response')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (dB)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Print filter information
    print("Filter info:")
    info = lowpass_filter.get_filter_info()
    for key, value in info.items():
        print(f"{key}: {value}")