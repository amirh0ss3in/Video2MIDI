import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# ─── Parameters ────────────────────────────────────────────────────────────────
fps = 60
start_seconds = 5
end_seconds   = 270
file_name     = "Radiohead - Let Down Piano Synthesia_timeslice.npy"

# ─── Load & slice the data ─────────────────────────────────────────────────────
# timeslice_array_full.shape == (total_frames, width, 3)
ts = np.load(file_name)
frames = ts[start_seconds*fps : end_seconds*fps]  
# now frames.shape == (num_frames, width, 3)

# ─── Grayscale conversion & horizontal profile ────────────────────────────────
# 1) to grayscale
bw_frames = frames.mean(axis=-1)         # shape: (num_frames, width)

# 2) average over time to detect keyboard layout
bw_mean   = bw_frames.mean(axis=0)       # shape: (width,)

# ─── Normalize & optionally invert ────────────────────────────────────────────
bw_mean = (bw_mean - bw_mean.min()) / (bw_mean.max() - bw_mean.min())
if bw_mean[:10].mean() > 0.5:
    bw_mean = 1 - bw_mean

# ─── Find 89 key‐edge peaks ─────────────────────────────────────────────────────
est_key_w = len(bw_mean) // 88
edges, _ = find_peaks(
    bw_mean,
    distance=est_key_w * 0.8,
)
assert len(edges) == 89, f"Found {len(edges)} edges, expected 89"

# ─── Compute 88 key centers ────────────────────────────────────────────────────
key_centers = 0.5 * (edges[:-1] + edges[1:])  # shape: (88,)

# ─── (Optional) Visualize edges & centers ─────────────────────────────────────
plt.figure(figsize=(12, 2))
plt.plot(bw_mean, label="Normalized Brightness")
plt.plot(edges, bw_mean[edges], 'k|', label="Edges")
plt.plot(key_centers, bw_mean[key_centers.astype(int)], 'ro', label="Centers")
plt.legend(loc="upper right")
plt.title("Keyboard Layout Detection")
plt.xlabel("Pixel Column")
plt.ylabel("Normalized Intensity")
plt.tight_layout()
plt.show()

# ─── Extract per-frame, per-key signals ────────────────────────────────────────
num_frames = bw_frames.shape[0]
key_signals = np.zeros((num_frames, 88), dtype=float)

for i in range(88):
    l = int(round(edges[i]))
    r = int(round(edges[i+1]))
    region = bw_frames[:, l:r]            # shape: (num_frames, key_width)
    key_signals[:, i] = region.mean(axis=1)

# key_signals.shape == (num_frames, 88)

# ─── Heatmap of key activations over time ─────────────────────────────────────

plt.figure(figsize=(12, 6))
plt.imshow(
    key_signals,
    aspect='auto',
    origin='lower',
    cmap='hot',
    interpolation='nearest'
)
plt.colorbar(label="Average Grayscale Intensity")
plt.xlabel("Key Index (0=A0 … 87=C8)")
plt.ylabel("Frame Index")
plt.title("Piano Roll Intensity Matrix")
plt.tight_layout()
plt.show()


# key_signals should be a NumPy array of shape (num_frames, 88)

# ─── Determine Key Activations (Piano Roll) ───────────────────────────────────
num_frames, num_keys = key_signals.shape
if num_keys != 88:
    print(f"Warning: Expected 88 keys, but found {num_keys}. Key type identification might be inaccurate if not starting from A0.")

# 1. Determine white/black keys based on standard piano layout
# Assuming key index 0 corresponds to A0 (MIDI note 21)
is_white_key_mask = np.zeros(num_keys, dtype=bool)
for i in range(num_keys):
    # Pitch class: 0=C, 1=C#, ..., 9=A, 10=A#, 11=B
    # MIDI note for key i = 21 + i (since A0 is 21)
    pitch_class = (21 + i) % 12
    # White keys have pitch classes corresponding to C, D, E, F, G, A, B
    if pitch_class in [0, 2, 4, 5, 7, 9, 11]: # These are C, D, E, F, G, A, B
        is_white_key_mask[i] = True

# 2. Estimate global baselines for unpressed white and black keys
# These percentiles are chosen to capture the typical unpressed state intensity.
percentile_for_dark_baseline = 10  # For black keys (normally dark)
percentile_for_bright_baseline = 90 # For white keys (normally bright)

# Aggregate signals from all black keys to find a robust dark baseline
# Ensure there are black keys to process
if np.any(~is_white_key_mask):
    black_key_signals_all = key_signals[:, ~is_white_key_mask].ravel()
    # Filter out potential NaNs or Infs if any, though unlikely with .mean()
    black_key_signals_all = black_key_signals_all[np.isfinite(black_key_signals_all)]
    if black_key_signals_all.size > 0:
        dark_baseline = np.percentile(black_key_signals_all, percentile_for_dark_baseline)
    else:
        dark_baseline = 0 # Fallback if no valid black key signals
else:
    dark_baseline = 0 # Fallback if no black keys identified

# Aggregate signals from all white keys to find a robust bright baseline
# Ensure there are white keys to process
if np.any(is_white_key_mask):
    white_key_signals_all = key_signals[:, is_white_key_mask].ravel()
    white_key_signals_all = white_key_signals_all[np.isfinite(white_key_signals_all)]
    if white_key_signals_all.size > 0:
        bright_baseline = np.percentile(white_key_signals_all, percentile_for_bright_baseline)
    else:
        bright_baseline = 255 # Fallback if no valid white key signals
else:
    bright_baseline = 255 # Fallback if no white keys identified

# print(f"Estimated Dark Baseline (for black keys): {dark_baseline:.2f}")
# print(f"Estimated Bright Baseline (for white keys): {bright_baseline:.2f}")

# 3. Calculate processed signals representing activation strength
processed_signals = np.zeros_like(key_signals)

for i in range(num_keys):
    if is_white_key_mask[i]: # White key: activation is a DECREASE from bright_baseline
        processed_signals[:, i] = bright_baseline - key_signals[:, i]
    else: # Black key: activation is an INCREASE from dark_baseline
        processed_signals[:, i] = key_signals[:, i] - dark_baseline

# 4. Clip negative values: these mean "more unpressed" than baseline or noise
processed_signals = np.maximum(0, processed_signals)

# (Optional) Visualize processed_signals before thresholding.
# This can help in choosing/validating the activation_threshold.
# On this plot, activations for ALL keys should appear as "hot" areas.
# plt.figure(figsize=(12, 6))
# plt.imshow(processed_signals, aspect='auto', origin='lower', cmap='hot', interpolation='nearest')
# plt.colorbar(label="Calculated Activation Strength")
# plt.xlabel("Key Index (0=A0 … 87=C8)")
# plt.ylabel("Frame Index")
# plt.title("Processed Key Signals (Activation Strength)")
# plt.tight_layout()
# plt.show()

# 5. Apply threshold to get the binary piano roll
# This threshold is a hyperparameter and might need tuning.
# Based on typical Synthesia color values (unpressed black ~25, unpressed white ~225, pressed ~125 on a 0-255 scale):
# - Activated black key: processed signal ~ (125 - 25) = 100
# - Activated white key: processed signal ~ (225 - 125) = 100
# A threshold like 30-50 should be reasonable.
activation_threshold = 40.0

piano_roll_binary = (processed_signals > activation_threshold).astype(np.int8)

# `piano_roll_binary` is now your (num_frames, 88) matrix where 1 indicates
# a pressed key and 0 indicates an unpressed key.

# ─── Plot Binary Piano Roll ───────────────────────────────────────────────────
# plt.figure(figsize=(12, 6))
# plt.imshow(
#     piano_roll_binary, # Shows time on Y-axis, keys on X-axis (like your original plot)
#     aspect='auto',
#     origin='lower',
#     cmap='gray_r',     # 1 (active) will be black, 0 (inactive) will be white. Use 'binary' for the reverse.
#     interpolation='nearest'
# )
# # For a binary image, a colorbar is often omitted or simplified.
# # plt.colorbar(ticks=[0, 1], format=plt.FuncFormatter(lambda val, loc: ['Inactive', 'Active'][int(val)]))
# plt.xlabel("Key Index (0=A0 … 87=C8)")
# plt.ylabel("Frame Index")
# plt.title("Binary Piano Roll (Detected Key Activations)")
# plt.tight_layout()
# plt.show()


# (Your existing code for computing key_signals and piano_roll_binary should be above this)
# key_signals: (num_frames, 88) float array of intensities
# piano_roll_binary: (num_frames, 88) int array of 0s and 1s (activations)

# ─── Visualize Overlay ───────────────────────────────────────────────────────
plt.figure(figsize=(15, 7)) # Make it a bit wider for better visibility

# 1. Plot the original key_signals heatmap as the background
plt.imshow(
    key_signals,
    aspect='auto',
    origin='lower',
    cmap='hot',  # This is the colormap from your original image
    interpolation='nearest'
)
# Add a colorbar for the background intensity map
cbar_bg = plt.colorbar(label="Average Grayscale Intensity (Background)")
# You might want to adjust cbar ticks if the range is very wide
# For example, if key_signals are 0-255:
# cbar_bg.set_ticks(np.linspace(np.min(key_signals), np.max(key_signals), 6))


# 2. Find the coordinates (frame index, key index) where piano_roll_binary is 1
active_frames_indices, active_keys_indices = np.where(piano_roll_binary == 1)

# 3. Overlay markers (e.g., small circles) for these active points
# We use plt.scatter for this. Choose a color that contrasts well with 'hot'.
# 'cyan', 'lime', or 'magenta' are often good choices.
# plt.scatter(
#     active_keys_indices,      # X-coordinates (key indices)
#     active_frames_indices,    # Y-coordinates (frame indices)
#     marker='o',               # Shape of the marker (e.g., 'o' for circle, '.' for point)
#     s=10,                     # Size of the markers (adjust as needed)
#     c='cyan',                 # Color of the markers
#     alpha=0.6,                # Transparency (0.0 to 1.0) if points are dense
#     label='Detected Activation' # Label for the legend
# )

# plt.xlabel("Key Index (0=A0 … 87=C8)")
# plt.ylabel("Frame Index")
# plt.title("Detected Activations Overlaid on Piano Roll Intensity Matrix")

# # Add a legend to identify the markers
# plt.legend(loc='upper right')

# plt.tight_layout()
# plt.show()


import mido
import numpy as np

# Make sure piano_roll_binary and fps are defined from your previous script steps
# For example:
# fps = 60 # This should be the fps value from your script
# piano_roll_binary = ... # This is your (num_frames, 88) numpy array of 0s and 1s

def piano_roll_to_midi(piano_roll_binary, fps, output_midi_path="output.mid",
                       default_velocity=90, ticks_per_beat=480, tempo_bpm=120.0):
    """
    Converts a binary piano roll to a MIDI file.

    Args:
        piano_roll_binary (np.ndarray): A 2D NumPy array (num_frames, num_keys)
                                       where 1 indicates a note is on, 0 off.
                                       Assumes 88 keys, with key_index 0 = A0 (MIDI note 21).
        fps (float): Frames per second of the source from which the piano roll was derived.
        output_midi_path (str): Path to save the generated MIDI file.
        default_velocity (int): MIDI velocity for note_on events (1-127).
        ticks_per_beat (int): MIDI ticks per beat (typically a quarter note).
        tempo_bpm (float): Tempo in beats per minute.
    """
    if not mido:
        print("Mido library not found. Please install it: pip install mido")
        return

    num_frames, num_keys = piano_roll_binary.shape
    if num_keys != 88:
        print(f"Warning: Piano roll has {num_keys} keys, but 88 were expected for standard piano. "
              f"MIDI note mapping assumes key 0 is A0.")

    # MIDI note number for A0 (the first key, index 0)
    midi_note_offset = 21

    # Calculate ticks per frame for MIDI timing
    # ticks_per_second = (ticks_per_beat * tempo_bpm) / 60
    # ticks_per_frame = ticks_per_second / fps
    ticks_per_frame = (ticks_per_beat * tempo_bpm) / (60 * fps)

    # List to store all note events with their absolute tick times
    # Each event is a dictionary: {'time_abs': absolute_tick, 'type': 'note_on'/'note_off', 'note': midi_note, 'velocity': velocity}
    note_events = []

    # Iterate through each frame and key to find note on/off events
    for frame_idx in range(num_frames):
        # Absolute time in ticks for events in the current frame
        current_frame_abs_tick = round(frame_idx * ticks_per_frame)
        for key_idx in range(num_keys):
            midi_note = key_idx + midi_note_offset
            
            current_state = piano_roll_binary[frame_idx, key_idx]
            # Determine previous state (0 if it's the first frame)
            prev_state = piano_roll_binary[frame_idx - 1, key_idx] if frame_idx > 0 else 0

            if current_state == 1 and prev_state == 0:  # Note ON event
                note_events.append({'time_abs': current_frame_abs_tick, 'type': 'note_on',
                                    'note': midi_note, 'velocity': default_velocity})
            elif current_state == 0 and prev_state == 1:  # Note OFF event
                note_events.append({'time_abs': current_frame_abs_tick, 'type': 'note_off',
                                    'note': midi_note, 'velocity': 0}) # Velocity for note_off is typically 0 or 64

    # Ensure any notes still 'on' at the very end of the roll are turned off
    # The time for these final note_off events is effectively after the last frame's content
    final_event_abs_tick = round(num_frames * ticks_per_frame)
    for key_idx in range(num_keys):
        if piano_roll_binary[num_frames - 1, key_idx] == 1: # If key was on in the last frame
            midi_note = key_idx + midi_note_offset
            note_events.append({'time_abs': final_event_abs_tick, 'type': 'note_off',
                                'note': midi_note, 'velocity': 0})

    # Sort events: primarily by absolute time, secondarily to prioritize note_off over note_on if at the same time
    def event_sort_key(event):
        type_priority = 0 if event['type'] == 'note_off' else 1 # note_off comes before note_on
        return (event['time_abs'], type_priority, event['note']) # Added note for stable sort on same time/type

    note_events.sort(key=event_sort_key)

    # Create MIDI file and track
    mid = mido.MidiFile(ticks_per_beat=ticks_per_beat)
    track = mido.MidiTrack()
    mid.tracks.append(track)

    # Add tempo information (MetaMessage) at the beginning of the track
    microseconds_per_beat = mido.bpm2tempo(tempo_bpm) # Converts BPM to microseconds per beat
    track.append(mido.MetaMessage('set_tempo', tempo=int(microseconds_per_beat), time=0))

    # Convert absolute tick times to delta times for MIDI messages
    last_event_abs_tick = 0
    for event_data in note_events:
        abs_tick = event_data['time_abs']
        delta_ticks = abs_tick - last_event_abs_tick # Time since the previous event
        
        track.append(mido.Message(event_data['type'],
                                  note=event_data['note'],
                                  velocity=event_data['velocity'],
                                  time=delta_ticks))
        last_event_abs_tick = abs_tick
        
    # Save the MIDI file
    try:
        mid.save(output_midi_path)
        print(f"MIDI file successfully saved to: {output_midi_path}")
    except Exception as e:
        print(f"Error saving MIDI file: {e}")


piano_roll_to_midi(
    piano_roll_binary,
    fps,  # Make sure this variable holds your FPS value
    output_midi_path="letdown.mid", # Choose your output filename
    default_velocity=100, # Standard velocity, can be 1-127
    tempo_bpm=90      # Common default, adjust if known
)