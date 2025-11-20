# STAL – Spot Tracking & Analysis for LEED

Trackleed is a PyQt6 desktop application for loading LEED image stacks, detecting diffraction spots, and tracking them over time.

## Spot Detection Backends

The **Analysis Parameters → Find Spots** panel lets you choose between:

- **Classic** – the original white-top-hat + peak finder. Tune WTH radius, blur sigma, peak distance, and ROI size.
- **Deep Learning** – runs a trained segmentation model (U-Net) from the `segmentation_models/` folder. Drop your `.pt` checkpoints there, click **Refresh**, and adjust confidence / area / dilation parameters to map the segmentation mask into ROIs.

You can switch backends at any time; settings are preserved per backend.
