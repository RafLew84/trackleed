# stal/io/results_exporter.py

import os
import pandas as pd
from typing import List, Dict
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from trackleed.core.data_models import LeedFile, Track
from trackleed.io.image_reader import load_tiff

def _save_full_data_csv(tracks: List[Track], all_files_map: Dict[int, LeedFile], directory_path: str):
    """Write the full per-spot numerical results, including radius metrics."""
    all_results = []
    for track in tracks:
        if not hasattr(track, 'analysis_results'): continue
        for frame_id, fit_results in sorted(track.analysis_results.items()):
            if not fit_results: continue
            leed_file = all_files_map.get(frame_id)
            if not leed_file: continue
            row = {
                'track_id': track.track_id,
                'frame_id': frame_id,
                'temperature': leed_file.temperature,
                'energy': leed_file.energy,
                'coverage': leed_file.coverage,
                **fit_results
            }
            row.pop('covariance_matrix', None)
            all_results.append(row)

    if not all_results: return
    df = pd.DataFrame(all_results)
    preferred_order = [
        'track_id', 'frame_id', 'temperature', 'energy', 'coverage',
        'global_center_x', 'center_x_error', 
        'global_center_y', 'center_y_error',
        'radius', 'radius_error',
        'volume', 'volume_error', 
        'amplitude', 'amplitude_error',
        'fwhm_x', 'fwhm_x_error', 
        'fwhm_y', 'fwhm_y_error', 
        'asymmetry', 'r_squared'
    ]
    existing_columns = [col for col in preferred_order if col in df.columns]
    df = df[existing_columns]
    output_path = os.path.join(directory_path, 'full_analysis_per_spot.csv')
    df.to_csv(output_path, index=False, float_format='%.4f')

def _save_center_data_csv(all_files_map: Dict[int, LeedFile], directory_path: str):
    """Write pattern-center information for every processed frame."""
    center_data = []
    for frame_id, leed_file in sorted(all_files_map.items()):
        if leed_file.pattern_center:
            center_data.append(
                {
                    'frame_id': frame_id,
                    'temperature': leed_file.temperature,
                    'energy': leed_file.energy,
                    'coverage': leed_file.coverage,
                    **leed_file.pattern_center
                }
            )
    if not center_data: return
    df = pd.DataFrame(center_data)
    output_path = os.path.join(directory_path, 'pattern_center_data.csv')
    df.to_csv(output_path, index=False, float_format='%.4f')

def _save_average_radius_csv(all_files_map: Dict[int, LeedFile], directory_path: str):
    """Write frame-level average radius summaries."""
    avg_data = []
    for frame_id, leed_file in sorted(all_files_map.items()):
        if hasattr(leed_file, 'post_analysis_results') and leed_file.post_analysis_results:
            row = {
                'frame_id': frame_id,
                'temperature': leed_file.temperature,
                'energy': leed_file.energy,
                'coverage': leed_file.coverage,
                **leed_file.post_analysis_results
            }
            avg_data.append(row)
    if not avg_data: return
    df = pd.DataFrame(avg_data)
    output_path = os.path.join(directory_path, 'average_radius_per_frame.csv')
    df.to_csv(output_path, index=False, float_format='%.4f')

def _save_track_legend_to_csv(tracks: List[Track], directory_path: str):
    """Creates and saves a legend file mapping track IDs to their starting points."""
    legend_data = []
    for track in tracks:
        if not track.spots_by_image_id: continue
        # Find the first frame for this track
        start_frame_id = min(track.spots_by_image_id.keys())
        start_spot = track.spots_by_image_id[start_frame_id]
        legend_data.append({
            'track_id': track.track_id,
            'start_frame_id': start_frame_id,
            'initial_x_pos': start_spot.roi_rect.center().x(),
            'initial_y_pos': start_spot.roi_rect.center().y()
        })
    if not legend_data: return
    
    df = pd.DataFrame(legend_data)
    output_path = os.path.join(directory_path, 'track_legend.csv')
    df.to_csv(output_path, index=False)
    print(f"Track legend saved to: {output_path}")

def _save_visualization_images(tracks: List[Track], all_files_map: Dict[int, LeedFile], directory_path: str):
    """Saves images with track overlays for every 5th frame."""
    images_dir = os.path.join(directory_path, 'track_visualizations')
    os.makedirs(images_dir, exist_ok=True)

    all_frame_ids = sorted(list(all_files_map.keys()))
    colors = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF', '#FFA500', '#800080']

    for i, frame_id in enumerate(all_frame_ids):
        if i % 5 != 0: continue # Process only every 5th frame

        image_data = load_tiff(all_files_map[frame_id].filepath)
        if image_data is None: continue

        # Convert to a drawable RGB image
        if image_data.ndim == 2: # Grayscale to RGB
            image_data = np.stack((image_data,)*3, axis=-1)
        image_data = (255 * (image_data - image_data.min()) / (image_data.max() - image_data.min())).astype(np.uint8)
        
        img = Image.fromarray(image_data).convert("RGB")
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("arial.ttf", 15)
        except IOError:
            font = ImageFont.load_default()

        # Draw all tracks
        for track_idx, track in enumerate(tracks):
            color = colors[track.track_id % len(colors)]
            
            # Draw the trace up to the current frame
            points = [spot.roi_rect.center() for fid, spot in sorted(track.spots_by_image_id.items()) if fid <= frame_id]
            if len(points) > 1:
                draw.line([(p.x(), p.y()) for p in points], fill=color, width=1)
            
            # Draw a marker and ID at the current position
            if frame_id in track.spots_by_image_id:
                spot = track.spots_by_image_id[frame_id]
                center = spot.roi_rect.center()
                bbox = [center.x()-3, center.y()-3, center.x()+3, center.y()+3]
                draw.ellipse(bbox, outline=color, width=2)
                draw.text((center.x()+5, center.y()), str(track.track_id), fill=color, font=font)

        # Save the image
        output_path = os.path.join(images_dir, f'frame_{frame_id:04d}.png')
        img.save(output_path)
    
    print(f"Visualization images saved to: {images_dir}")

def export_full_results(
    tracks: List[Track],
    all_files_map: Dict[int, LeedFile],
    directory_path: str
):
    """
    Main exporter function that saves all analysis results.
    """
    # Save the main numerical data
    _save_full_data_csv(tracks, all_files_map, directory_path)
    _save_center_data_csv(all_files_map, directory_path)
    _save_average_radius_csv(all_files_map, directory_path)
    
    # Save the track legend
    _save_track_legend_to_csv(tracks, directory_path)
    
    # Save visualization images
    _save_visualization_images(tracks, all_files_map, directory_path)
