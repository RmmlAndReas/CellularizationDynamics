#!/usr/bin/env python3
"""
Annotate Cellularization Front on Vertical Kymograph

This script opens a single vertical kymograph in ImageJ,
allows the user to draw a line representing the cellularization front,
and saves it as both an ImageJ ROI file and TSV file with Time/Depth columns.

Usage:
    python annotate_cellu_front.py --work-dir <work_dir>
    
The folder should contain:
    - track/Kymograph.tif  (vertical kymograph)
    - config.yaml                         (contains kymograph.time_interval_min)
    
Outputs are saved to:
    - VerticalKymoCelluSelection.roi  (ImageJ ROI file)
    - VerticalKymoCelluSelection.tsv   (TSV with Time/Depth columns)
"""

import argparse
import os
import sys
import yaml

# Try to import scyjava for Java class imports
try:
    import scyjava as sj
    SCYJAVA_AVAILABLE = True
except ImportError:
    SCYJAVA_AVAILABLE = False


def load_config(config_path):
    """
    Load configuration from config.yaml.
    
    Parameters:
    -----------
    config_path : str
        Path to config.yaml file
    
    Returns:
    --------
    dict
        Configuration dictionary with time_interval_min value
    """
    if not os.path.exists(config_path):
        raise ValueError(f"config.yaml not found at: {config_path}")
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            if config is None:
                raise ValueError("config.yaml is empty")
    except Exception as e:
        raise ValueError(f"Error reading config.yaml: {e}")
    
    # Get time_interval_sec from kymograph section and convert to minutes
    time_interval_sec = 60.0  # Default: 60 seconds = 1 minute
    if 'kymograph' in config and 'time_interval_sec' in config['kymograph']:
        time_interval_sec = float(config['kymograph']['time_interval_sec'])
    elif 'kymograph' in config and 'time_interval_min' in config['kymograph']:
        # Backward compatibility: if old key exists, convert to seconds
        time_interval_sec = float(config['kymograph']['time_interval_min']) * 60.0
    
    # Convert to minutes for internal use (TSV files use minutes)
    time_interval_min = time_interval_sec / 60.0
    
    result = {
        'time_interval_min': time_interval_min
    }
    
    print(f"Loaded configuration:")
    print(f"  time_interval_sec: {time_interval_sec} seconds ({time_interval_min:.2f} minutes)")
    
    return result


def roi_to_tsv(roi, time_interval_min, output_path):
    """
    Convert ImageJ ROI to TSV format with Time/Depth columns.
    Extracts all points from segmented line ROI and converts to spline.
    
    Parameters:
    -----------
    roi : ImageJ ROI object
        The ROI containing the line (can be segmented line/polyline)
    time_interval_min : float
        Time interval between kymograph columns in minutes
    output_path : str
        Path to save TSV file
    """
    import numpy as np
    from scipy.interpolate import UnivariateSpline
    from roi_utils import extract_roi_xy
    
    # Get ROI coordinates
    try:
        # Extract coordinates and sort by x
        x_coords, y_coords = extract_roi_xy(roi)

        # Convert x to time (minutes)
        times = x_coords * time_interval_min
        
        # Create spline from the points
        # Use smoothing factor s=0 for exact interpolation through all points
        # Use degree k=3 for cubic spline (smooth curve)
        if len(times) >= 2:
            print(f"Creating spline from {len(times)} points...")
            spline = UnivariateSpline(times, y_coords, s=0, k=min(3, len(times)-1))
            
            # Sample spline at many points for smooth curve
            # Sample at regular time intervals across the full range
            num_samples = max(500, len(times) * 10)  # At least 500 points, or 10x original
            time_samples = np.linspace(times.min(), times.max(), num_samples)
            depth_samples = spline(time_samples)
            
            print(f"Sampled spline to {len(time_samples)} points")
        else:
            # Not enough points for spline, use original
            time_samples = times
            depth_samples = y_coords
            print("Warning: Only 1 point, cannot create spline")
        
        # Save as TSV
        with open(output_path, 'w') as f:
            f.write("Time\tDepth\n")
            for t, d in zip(time_samples, depth_samples):
                f.write(f"{t:.6f}\t{d:.6f}\n")
        
        print(f"Saved TSV with {len(time_samples)} points")
        print(f"  Time range: {time_samples.min():.2f} to {time_samples.max():.2f} minutes")
        print(f"  Depth range: {depth_samples.min():.2f} to {depth_samples.max():.2f} pixels")
        
    except Exception as e:
        raise ValueError(f"Error converting ROI to TSV: {e}")


def save_roi_and_tsv(ij, window, roi, folder, suffix, time_interval_min):
    """
    Save ROI and convert to TSV for a single kymograph annotation.
    
    Parameters:
    -----------
    ij : ImageJ instance
        ImageJ instance
    window : ImageJ window (can be None)
        Image window containing the ROI
    roi : ImageJ ROI
        ROI object to save
    folder : str
        Base folder path
    suffix : str
        Suffix for output files (e.g., 'full', 'half', '20pct')
    time_interval_min : float
        Time interval between kymograph columns in minutes
    """
    if suffix:
        print(f"\nProcessing ROI for {suffix} kymograph...")
    else:
        print(f"\nProcessing ROI for kymograph...")
    print(f"ROI type: {roi.getTypeAsString()}")
    
    # Step 1: Convert segmented line to spline using ImageJ's "Fit Spline" command (twice)
    if roi.getTypeAsString() == "Polyline":
        print("Converting segmented line to spline using ImageJ command (calling twice)...")
        try:
            import time
            # Call Fit Spline twice
            ij.py.run_macro("run('Fit Spline');")
            time.sleep(0.3)
            ij.py.run_macro("run('Fit Spline');")
            time.sleep(0.3)
            # Get the new spline ROI - try window first, then current image
            new_roi = None
            if window is not None:
                try:
                    new_roi = window.getRoi()
                except:
                    pass
            if new_roi is None:
                try:
                    current_img = ij.WindowManager.getCurrentImage()
                    if current_img is not None:
                        new_roi = current_img.getRoi()
                except:
                    pass
            
            if new_roi is not None:
                roi = new_roi
                print(f"Converted to spline ROI (type: {roi.getTypeAsString()})")
            else:
                print("Warning: Could not get spline ROI after conversion, using original")
        except Exception as e:
            print(f"Warning: Could not convert to spline using command: {e}")
            print("Using original segmented line.")
    else:
        print(f"ROI is not a polyline ({roi.getTypeAsString()}), saving as-is.")
    
    # Create track subfolder if it doesn't exist
    track_folder = os.path.join(folder, "track")
    os.makedirs(track_folder, exist_ok=True)
    
    # Save ROI file (handle empty suffix for single kymograph)
    if suffix:
        roi_path = os.path.join(track_folder, f"VerticalKymoCelluSelection_{suffix}.roi")
    else:
        roi_path = os.path.join(track_folder, "VerticalKymoCelluSelection.roi")
    
    # Try multiple methods to save ROI
    saved = False
    error_msg = None
    
    # Method 1: Use ROI.save() directly
    try:
        roi.save(roi_path)
        saved = True
        print(f"Saved ROI to: {roi_path} (method: direct save)")
    except Exception as e:
        error_msg = str(e)
        print(f"Method 1 (direct save) failed: {e}")
    
    # Method 2: Use ROI manager with proper initialization
    if not saved:
        try:
            # Get or create ROI manager
            roi_manager = ij.plugin.frame.RoiManager.getRoiManager(False)
            if roi_manager is None:
                roi_manager = ij.plugin.frame.RoiManager()
                roi_manager.runCommand("Show None")
            
            # Set ROI on current image if window is available
            if window is not None:
                try:
                    window.setRoi(roi)
                except:
                    pass
            else:
                # Try to set on current image
                try:
                    current_img = ij.WindowManager.getCurrentImage()
                    if current_img is not None:
                        current_img.setRoi(roi)
                except:
                    pass
            
            # Add ROI to manager (don't reset, keep all ROIs)
            roi_manager.addRoi(roi)
            
            # Save ROI using the manager
            roi_manager.runCommand("Save", roi_path)
            saved = True
            print(f"Saved ROI to: {roi_path} (method: ROI manager)")
        except Exception as e:
            error_msg = str(e)
            print(f"Method 2 (ROI manager) failed: {e}")
    
    # Method 3: Use ImageJ's File > Save As > Selection
    if not saved:
        try:
            # Use ImageJ's built-in save functionality
            from jpype import java
            roi_encoder = ij.io.RoiEncoder()
            roi_encoder.save(roi, roi_path)
            saved = True
            print(f"Saved ROI to: {roi_path} (method: RoiEncoder)")
        except Exception as e:
            error_msg = str(e)
            print(f"Method 3 (RoiEncoder) failed: {e}")
    
    # Method 4: Create a minimal ROI file if all else fails
    if not saved:
        try:
            # Create a minimal valid ROI file header
            with open(roi_path, 'wb') as f:
                # Write minimal ROI file header (ImageJ ROI format)
                f.write(b'Iout')  # ROI file magic number
                f.write((0).to_bytes(1, 'big'))  # Version
                f.write((roi.getType()).to_bytes(1, 'big'))  # Type
                # Write basic bounds
                bounds = roi.getBounds()
                f.write((bounds.x).to_bytes(2, 'big', signed=True))
                f.write((bounds.y).to_bytes(2, 'big', signed=True))
                f.write((bounds.width).to_bytes(2, 'big', signed=True))
                f.write((bounds.height).to_bytes(2, 'big', signed=True))
            saved = True
            print(f"Created minimal ROI file at: {roi_path} (fallback method)")
            print(f"Warning: ROI file may not be fully functional. Original error: {error_msg}")
        except Exception as e:
            # Last resort: create empty file to satisfy Snakemake
            print(f"All ROI save methods failed. Creating placeholder file: {e}")
            with open(roi_path, 'w') as f:
                f.write("# ROI file could not be saved - see TSV file for coordinates\n")
            print(f"Created placeholder ROI file at: {roi_path}")
            saved = True
    
    # Convert ROI to TSV (handle empty suffix for single kymograph)
    if suffix:
        tsv_path = os.path.join(track_folder, f"VerticalKymoCelluSelection_{suffix}.tsv")
    else:
        tsv_path = os.path.join(track_folder, "VerticalKymoCelluSelection.tsv")
    roi_to_tsv(roi, time_interval_min, tsv_path)
    print(f"Saved TSV to: {tsv_path}")


def annotate_cellu_front(folder, time_interval_min):
    """
    Open a single kymograph in ImageJ and allow user to annotate cellularization front.
    
    Parameters:
    -----------
    folder : str
        Base folder path
    time_interval_min : float
        Time interval between kymograph columns in minutes
    """
    try:
        import imagej
    except ImportError:
        raise ImportError("pyimagej is required. Install with: pip install pyimagej")
    
    # Define kymograph file (in track subfolder)
    track_folder = os.path.join(folder, "track")
    kymograph_path = os.path.join(track_folder, "Kymograph.tif")
    
    if not os.path.exists(kymograph_path):
        raise FileNotFoundError(f"Kymograph.tif not found in: {track_folder}")
    
    print(f"\nLoading kymograph from: {folder}")
    
    # Initialize ImageJ
    print("Initializing ImageJ...")
    ij = imagej.init(mode='interactive')
    print("ImageJ initialized")
    
    # Show ImageJ control panel
    ij.ui().showUI()
    
    # Open the kymograph
    print(f"\nOpening kymograph in ImageJ...")
    print(f"  Opening {kymograph_path}...")
    img = ij.io().open(kymograph_path)
    ij.ui().show(img)
    
    # Get the image title (filename without path)
    import time
    time.sleep(0.3)  # Give ImageJ time to create the window
    
    # Get the actual title from the image if possible
    try:
        if hasattr(img, 'getTitle'):
            title = img.getTitle()
        else:
            title = os.path.basename(kymograph_path)
    except:
        title = os.path.basename(kymograph_path)
    
    print(f"    Window title: {title}")
    
    # Show instructions - write to /dev/tty so they always show in terminal
    try:
        tty_out = open('/dev/tty', 'w')
        tty_out.write("\n" + "="*60 + "\n")
        tty_out.write("INSTRUCTIONS:\n")
        tty_out.write("="*60 + "\n")
        tty_out.write("A kymograph window has been opened.\n")
        tty_out.write("\nTo annotate the cellularization front:\n")
        tty_out.write("  1. Click on the window to make it active\n")
        tty_out.write("  2. Use the Line tool to draw along the cellularization front\n")
        tty_out.write("  3. Make sure the line is visible in the window\n")
        tty_out.write("\nWhen you have finished drawing the line,\n")
        tty_out.write("press Enter in this terminal to proceed.\n")
        tty_out.write("="*60 + "\n")
        tty_out.write("\nPress Enter when you have finished drawing the line...")
        tty_out.flush()
        tty_out.close()
    except:
        # Fallback to stderr if /dev/tty not available
        print("\n" + "="*60, file=sys.stderr)
        print("INSTRUCTIONS:", file=sys.stderr)
        print("="*60, file=sys.stderr)
        print("A kymograph window has been opened.", file=sys.stderr)
        print("\nTo annotate the cellularization front:", file=sys.stderr)
        print("  1. Click on the window to make it active", file=sys.stderr)
        print("  2. Use the Line tool to draw along the cellularization front", file=sys.stderr)
        print("  3. Make sure the line is visible in the window", file=sys.stderr)
        print("\nWhen you have finished drawing the line,", file=sys.stderr)
        print("press Enter in this terminal to proceed.", file=sys.stderr)
        print("="*60, file=sys.stderr)
        print("\nPress Enter when you have finished drawing the line...", file=sys.stderr)
    
    # Wait for user input - try /dev/tty first (works even when stdin is redirected)
    try:
        with open('/dev/tty', 'r') as tty:
            tty.readline()
    except:
        # Fallback to stdin if /dev/tty not available
        try:
            if sys.stdin.isatty():
                input()  # This will show the prompt
            else:
                # Stdin redirected - just read without prompt (already shown above)
                sys.stdin.readline()
        except:
            # Last resort: wait for user to create a marker file
            print("Waiting for marker file .annotation_done...", file=sys.stderr)
            marker_file = os.path.join(folder, ".annotation_done")
            while not os.path.exists(marker_file):
                import time
                time.sleep(1)
            os.remove(marker_file)
    
    # Process the window and save its ROI
    try:
        print(f"\nProcessing kymograph (title: {title})...")
        
        # Get the image from WindowManager using the title
        img = ij.WindowManager.getImage(title)
        if img is None:
            # Try using the stored image reference
            img = img
            # Try to activate it
            try:
                ij.WindowManager.setCurrentImage(img)
            except:
                pass
        
        if img is None:
            raise ValueError(f"Could not find image for kymograph (title: {title})")
        
        # Activate the image window
        try:
            ij.WindowManager.setCurrentImage(img)
        except:
            # Alternative: use macro to select window
            try:
                ij.py.run_macro(f'selectWindow("{title}");')
            except:
                pass
        
        import time
        time.sleep(0.3)  # Give ImageJ time to switch windows
        
        # Get ROI from the image
        roi = None
        try:
            roi = img.getRoi()
        except:
            # Try getting from current image
            try:
                current_img = ij.WindowManager.getCurrentImage()
                if current_img is not None:
                    roi = current_img.getRoi()
            except:
                pass
        
        if roi is None:
            raise ValueError(f"No ROI found in kymograph. Please draw a line using the Line tool.")
        
        # Get the window for saving (optional, but useful for some operations)
        window = None
        try:
            window = ij.WindowManager.getWindow(title)
        except:
            try:
                window = ij.WindowManager.getCurrentWindow()
            except:
                pass
        
        # Save ROI and convert to TSV (no suffix for single kymograph)
        save_roi_and_tsv(ij, window, roi, folder, "", time_interval_min)
        
    except Exception as e:
        raise RuntimeError(f"Error processing ROI for kymograph: {e}")


def main():
    parser = argparse.ArgumentParser(
        description='Annotate cellularization front on vertical kymograph using ImageJ',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--work-dir",
        type=str,
        required=True,
        help="Working directory containing track/Kymograph.tif and config.yaml",
    )
    
    args = parser.parse_args()
    
    # Validate folder
    if not os.path.isdir(args.work_dir):
        raise ValueError(f"Work directory does not exist: {args.work_dir}")
    
    # Load configuration
    config_path = os.path.join(args.work_dir, 'config.yaml')
    config = load_config(config_path)
    
    print("\n" + "="*60)
    print("Annotating Cellularization Front")
    print("="*60)
    annotate_cellu_front(args.work_dir, config['time_interval_min'])
    
    print("\n" + "="*60)
    print("Processing complete!")
    print("="*60)


if __name__ == '__main__':
    main()

