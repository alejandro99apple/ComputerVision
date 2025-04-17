import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# Parameters
fs = 256  # Sampling frequency in Hz
duration = 5  # Duration in seconds
n_channels = 19  # Number of EEG channels

# Time vector
t = np.linspace(0, duration, int(fs * duration), endpoint=False)

# Simulate EEG signals (fully random signals)
eeg_signals = []
for i in range(n_channels):
    # Fully random signal
    random_signal = np.random.normal(-20, 20, len(t))
    eeg_signals.append(random_signal)

# Convert to numpy array
eeg_signals = np.array(eeg_signals)

# Create additional signals A1 and A2 with scaled amplitudes
A1 = 20 * (np.sin(2 * np.pi * 0.5 * t) + np.cos(2 * np.pi * 0.2 * t))  # Scale to match random signals
A2 = 20 * (np.sin(2 * np.pi * 0.7 * t) + np.cos(2 * np.pi * 0.04 * t))  # Scale to match random signals

# Add A1 and A2 to the EEG signals array
eeg_signals = np.vstack([eeg_signals, A1, A2])

# Store signals in a dictionary
signals_dict = {f'Channel {i+1}': eeg_signals[i] for i in range(n_channels)}
signals_dict['A1'] = A1
signals_dict['A2'] = A2

# Plot EEG signals using the dictionary (original signals)
plt.figure(figsize=(12, 8))
offset = 200  # Increase offset to better distinguish variations
for i, (name, signal) in enumerate(signals_dict.items()):
    plt.plot(t, signal + i * offset, label=name)  # Use dictionary keys for labels

plt.title('Original Simulated EEG Signals')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (µV)')
plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1.05))
plt.tight_layout()
plt.show()

# Increase the amplitude of half of the signals (excluding A1 and A2)
half_index = n_channels // 2
for i in range(half_index):
    signals_dict[f'Channel {i+1}'] *= 2

# Plot EEG signals using the updated dictionary (modified signals)
plt.figure(figsize=(12, 8))
for i, (name, signal) in enumerate(signals_dict.items()):
    plt.plot(t, signal + i * offset, label=name)  # Use dictionary keys for labels

plt.title('Modified Simulated EEG Signals')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (µV)')
plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1.05))
plt.tight_layout()
plt.show()

# Define a Butterworth filter for the alpha band (8-12 Hz)
def butter_bandpass(lowcut, highcut, fs, order=2):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

def apply_filter(signal, lowcut, highcut, fs, order=2):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    return filtfilt(b, a, signal)

# Filter the modified signals (excluding A1 and A2)
lowcut = 8  # Alpha band lower bound
highcut = 12  # Alpha band upper bound
for i in range(n_channels):
    signals_dict[f'Channel {i+1}'] = apply_filter(signals_dict[f'Channel {i+1}'], lowcut, highcut, fs)

# Generate reference montage signals (Signals - Ref)
reference_signal = signals_dict['Channel 1']
montage_signals = {}
for name, signal in signals_dict.items():
    if True:  # Exclude the reference channel itself
        montage_signals[f'{name} - Ref'] = signal - reference_signal

# Compute the average of the last two signals in montage_signals
last_two_signals = list(montage_signals.values())[-2:]
average_signal = sum(last_two_signals) / 2

# Subtract the average signal from the first 19 channels in montage_signals
adjusted_signals = {}
for i, (name, signal) in enumerate(montage_signals.items()):
    if i < n_channels:  # Only process the first 19 channels
        adjusted_signals[name] = signal - average_signal

# Create a figure with subplots for filtered signals and adjusted montage signals
fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharey=True)

# Plot filtered signals (Alpha Band) on the left
axes[0].set_title('Filtered Simulated EEG Signals (Alpha Band)')
for i, (name, signal) in enumerate(signals_dict.items()):
    axes[0].plot(t, signal + i * offset, label=name)
axes[0].set_xlabel('Time (s)')
axes[0].set_ylabel('Amplitude (µV)')
axes[0].legend(loc='upper right', bbox_to_anchor=(1.15, 1.05))

# Plot adjusted montage signals on the right
axes[1].set_title('Adjusted Montage Signals')
for i, (name, signal) in enumerate(adjusted_signals.items()):
    axes[1].plot(t, signal + i * offset, label=name)
axes[1].set_xlabel('Time (s)')

# Adjust layout and show the combined plot
plt.tight_layout()
plt.show()
