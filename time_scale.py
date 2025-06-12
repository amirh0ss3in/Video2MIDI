import cv2
import numpy as np
import os
from tqdm import tqdm # For the progress bar

def create_timeslice_array(video_path, y_height=0.15, save_npy_path=None, save_image_path=None):
    """
    Creates a timeslice array by stacking the middle horizontal pixel line 
    from each frame of a video. Optionally saves it as .npy and/or an image file.

    This version is optimized by pre-allocating the NumPy array if the total
    frame count is reliably reported by the video, otherwise, it falls back
    to a list-append method. Includes a tqdm progress bar.

    Args:
        video_path (str): Path to the input video file.
        save_npy_path (str, optional): Path to save the resulting NumPy array.
        save_image_path (str, optional): Path to save the resulting image.

    Returns:
        numpy.ndarray: The timeslice array, or None if an error occurred.
    """
    # Open the video file
    cap = cv2.VideoCapture(video_path) 
    
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return None

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames_reported = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"Video: {os.path.basename(video_path)}")
    print(f"Properties: {frame_width}x{frame_height}, "
          f"{total_frames_reported if total_frames_reported > 0 else 'Unknown'} frames (reported), "
          f"{fps:.2f} FPS")

    if frame_width == 0 or frame_height == 0:
        print(f"Error: Video has invalid dimensions ({frame_width}x{frame_height}).")
        cap.release()
        return None
    
    row_index = int(frame_height * (1- y_height))
    print(f"Extracting pixel line from row index: {row_index}")

    # Attempt to read the first frame to get num_channels and validate video
    ret, first_frame = cap.read()
    if not ret:
        print("Error: Could not read the first frame of the video.")
        cap.release()
        return None
    # first_frame.shape is (height, width, channels). We need channels.
    num_channels = first_frame.shape[2] 

    timeslice_array = None

    # --- Optimized Path: Pre-allocation if total_frames_reported is reliable ---
    if total_frames_reported > 0:
        print(f"Using pre-allocation method for {total_frames_reported} frames.")
        
        # Pre-allocate the NumPy array
        # Shape: (num_frames, video_width, num_channels)
        timeslice_array_prealloc = np.empty((total_frames_reported, frame_width, num_channels), dtype=np.uint8)
        
        # Place the first frame's data (already read) into the array
        timeslice_array_prealloc[0, :, :] = first_frame[row_index, :, :]
        
        frames_actually_read = 1 # Count the first frame
        
        for i in tqdm(range(1, total_frames_reported), 
                      desc="Processing frames (pre-allocated)", 
                      unit="frame", 
                      initial=1, 
                      total=total_frames_reported):
            ret_loop, frame_loop = cap.read()
            if not ret_loop:
                print(f"\nWarning: Video stream ended prematurely at frame {i} "
                      f"(expected {total_frames_reported}).")
                break # Exit loop if video ends early
            
            # Extract the middle line: frame_loop[row_index, all_columns, all_channels]
            middle_line = frame_loop[row_index, :, :]
            timeslice_array_prealloc[i, :, :] = middle_line
            frames_actually_read += 1
        
        # If fewer frames were read than reported (e.g., due to premature end)
        if frames_actually_read < total_frames_reported:
            print(f"Adjusting array size from {total_frames_reported} to {frames_actually_read} actual frames.")
            timeslice_array = timeslice_array_prealloc[:frames_actually_read, :, :]
        else:
            timeslice_array = timeslice_array_prealloc
        
        if frames_actually_read == 0: # Should not happen if first frame read was successful
             print("Error: No frames were effectively processed into the pre-allocated array.")
             cap.release()
             return None

    # --- Fallback Path: List append if total_frames_reported is not reliable ---
    else: 
        print("Warning: Total frame count not reliably reported by video. "
              "Using list-append method (potentially slower, progress bar won't show total).")
        lines_list = []
        
        # Add the first frame's line (already read)
        lines_list.append(first_frame[row_index, :, :].copy()) # .copy() is good practice here
        
        # Loop for subsequent frames
        # tqdm setup: initial=1 because first frame already processed and in list
        with tqdm(desc="Processing frames (list-append)", unit="frame", initial=1) as pbar:
            while True:
                ret_loop, frame_loop = cap.read()
                if not ret_loop:
                    break # End of video or error
                middle_line = frame_loop[row_index, :, :].copy()
                lines_list.append(middle_line)
                pbar.update(1) # Increment progress bar
        
        if not lines_list: # Should not happen if first frame read was successful
            print("Error: No frames were processed (list is empty).")
            cap.release()
            return None
        
        print(f"Stacking {len(lines_list)} collected lines...")
        timeslice_array = np.stack(lines_list, axis=0)

    # Release the video capture object
    cap.release()

    if timeslice_array is None or timeslice_array.shape[0] == 0:
        print("Error: Final timeslice array is empty or could not be created.")
        return None

    print(f"Timeslice array created successfully. Shape: {timeslice_array.shape}")

    # Save the NumPy array if path is provided
    if save_npy_path:
        try:
            np.save(save_npy_path, timeslice_array)
            print(f"Successfully saved NumPy array: {save_npy_path}")
        except Exception as e:
            print(f"Error saving NumPy array '{save_npy_path}': {e}")

    # Save the image if path is provided
    if save_image_path:
        try:
            # timeslice_array.shape[0] is number of frames (height of output image)
            if timeslice_array.shape[0] > 0:
                 cv2.imwrite(save_image_path, timeslice_array) # OpenCV expects BGR, which cap.read() provides
                 print(f"Successfully saved timeslice image: {save_image_path}")
            else:
                # This case should ideally be caught earlier if timeslice_array is None or empty
                print("Warning: Cannot save image, timeslice array has no frames.")
        except Exception as e:
            print(f"Error saving image '{save_image_path}': {e}")
            
    return timeslice_array

# --- Main execution ---
if __name__ == "__main__":
    video_file = "Radiohead - Let Down Piano Synthesia.mp4" # YOUR VIDEO FILE NAME HERE
    
    if not os.path.exists(video_file):
        print(f"Error: Video file '{video_file}' not found in the current directory: '{os.getcwd()}'.")
        print("Please make sure the script and the video are in the same folder, or provide the full path to the video file.")
    else:
        base_name_original = os.path.splitext(os.path.basename(video_file))[0]
        # Sanitize base_name for file system compatibility (replace non-alphanumeric with underscore)
        safe_base_name = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in base_name_original).strip()
        if not safe_base_name: # If original name was e.g. "!@#$%" or just spaces
            safe_base_name = "video_timeslice_output" # Default if sanitization results in empty string
        
        output_npy_file = f"{safe_base_name}_timeslice.npy"
        output_image_file = f"{safe_base_name}_timeslice.png" # Or .jpg if preferred
        
        print(f"--- Starting Timeslice Generation ---")
        print(f"Input video: {video_file}")
        print(f"Output NumPy array: {output_npy_file}")
        print(f"Output image: {output_image_file}")
        print("------------------------------------")

        # Call the function to process the video
        result_array = create_timeslice_array(video_file, 
                                              save_npy_path=output_npy_file,
                                              save_image_path=output_image_file)

        print("------------------------------------")
        if result_array is not None and result_array.shape[0] > 0:
            print("Processing complete.")
        else:
            print("Processing failed or resulted in an empty array.")
        print(f"--- End Timeslice Generation ---")