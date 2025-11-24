# stal/gui/tracking_dialog.py

from typing import List
import numpy as np
from PyQt6.QtCore import Qt, QTimer, QPoint, QPointF, QRect, QThread
from PyQt6.QtGui import QImage, QPixmap, QActionGroup, QAction
from PyQt6.QtWidgets import (QDialog, QWidget, QHBoxLayout, QVBoxLayout, QSlider, QSplitter,
                             QLabel, QTableWidget, QAbstractItemView, QTableWidgetItem, QDialogButtonBox,
                             QHeaderView, QPushButton, QMessageBox, QToolBar, QMenu, QMessageBox, QProgressDialog)

from trackleed.core.data_models import LeedFile, Track, SpotROI
from trackleed.gui.leed_image_viewer import LeedImageViewer
from trackleed.io.image_reader import load_tiff
from trackleed.analysis.tracking import retrack_single_spot_optical_flow
from trackleed.logic.generic_worker import GenericWorker


class TrackingDialog(QDialog):
    """
    A dedicated dialog for visualizing, analyzing, and editing tracking results.
    """
    def __init__(
        self,
        tracking_data: List[Track],
        all_files: List[LeedFile],
        masks: List,
        roi_size: int,
        lk_win_size: int,
        lk_max_level: int,
        parent=None,
        save_session_callback=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Tracking Results Visualizer")
        self.setMinimumSize(1200, 800)

        current_flags=self.windowFlags()
        self.setWindowFlags(current_flags | Qt.WindowType.WindowMinimizeButtonHint | Qt.WindowType.WindowMaximizeButtonHint)

        self.worker_thread = None 
        self.worker = None 

        self.tracks = tracking_data
        self.file_map = {f.id: f for f in all_files if f.id is not None}
        self.masks = masks
        self.roi_size = roi_size
        self.lk_win_size = lk_win_size  
        self.lk_max_level = lk_max_level
        self.save_session_callback = save_session_callback

        self.sorted_frame_ids = sorted(list(self.file_map.keys()))
        
        main_layout = QVBoxLayout(self)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)

        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        self.image_viewer = LeedImageViewer()
        self.image_viewer.track_point_context_menu_requested.connect(self.on_track_point_context_menu)
        self.image_viewer.point_for_track_added_at.connect(self.handle_add_or_correct_point)
        
        self.controls_layout = QHBoxLayout()
        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_label = QLabel("Frame: -/-")
        self.controls_layout.addWidget(self.frame_slider)
        self.controls_layout.addWidget(self.frame_label)
        
        left_layout.addWidget(self.image_viewer)
        left_layout.addLayout(self.controls_layout)
        splitter.addWidget(left_widget)

        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        self.tracks_table = QTableWidget()
        self.plot_widget = QLabel("Plot Area Placeholder")
        self.plot_widget.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.add_track_button = QPushButton("Add New Track")
        self.add_track_button.clicked.connect(self.enter_add_new_track_mode)

        self.delete_track_button = QPushButton("Delete Selected Track")
        self.delete_track_button.clicked.connect(self.delete_selected_track)
        right_layout.addWidget(QLabel("Found Tracks:"))
        right_layout.addWidget(self.tracks_table)
        right_layout.addWidget(self.delete_track_button)
        right_layout.insertWidget(3, self.add_track_button)
        right_layout.addWidget(QLabel("Track Analysis:"))
        right_layout.addWidget(self.plot_widget)
        splitter.addWidget(right_widget)
        
        splitter.setSizes([800, 400])

        self.animation_timer = QTimer(self)
        self.animation_timer.timeout.connect(self.advance_frame)

        self.button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self.save_session_button = self.button_box.addButton(
            "Save Session", QDialogButtonBox.ButtonRole.ActionRole
        )
        self.save_session_button.clicked.connect(self._on_save_session)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        main_layout.addWidget(self.button_box)

        self._init_ui_elements()
        self.populate_tracks_table()
        
        if self.sorted_frame_ids:
            # Show the first frame by its index (0)
            self.on_slider_value_changed(0)
            self.show_frame(self.sorted_frame_ids[0], fit=True)
            self.frame_slider.setValue(0)
        
        toolbar = QToolBar("Interaction Modes")
        main_layout.addWidget(toolbar) # Add the toolbar as a regular widget to the layout

        mode_group = QActionGroup(self)
        mode_group.setExclusive(True)

        pan_mode_action = QAction("Pan/View", self)
        pan_mode_action.setCheckable(True)
        pan_mode_action.setChecked(True)
        pan_mode_action.triggered.connect(lambda: self.image_viewer.set_mode('PAN'))
        toolbar.addAction(pan_mode_action)
        mode_group.addAction(pan_mode_action)

        edit_mode_action = QAction("Edit Track", self)
        edit_mode_action.setCheckable(True)
        edit_mode_action.triggered.connect(lambda: self.image_viewer.set_mode('EDIT_TRACK'))
        toolbar.addAction(edit_mode_action)
        mode_group.addAction(edit_mode_action)

        add_correct_mode_action = QAction("Add/Correct Point", self)
        add_correct_mode_action.setCheckable(True)
        add_correct_mode_action.triggered.connect(lambda: self.image_viewer.set_mode('ADD_CORRECT_POINT'))
        toolbar.addAction(add_correct_mode_action)
        mode_group.addAction(add_correct_mode_action)

        add_new_track_mode_action = QAction("Add New Track", self)
        add_new_track_mode_action.setCheckable(True)
        add_new_track_mode_action.triggered.connect(self.enter_add_new_track_mode)
        toolbar.addAction(add_new_track_mode_action)
        mode_group.addAction(add_new_track_mode_action)

        self.image_viewer.new_track_started_at.connect(self.handle_add_new_track)

    def _on_save_session(self):
        """Trigger saving the current session from inside the dialog."""
        if self.save_session_callback:
            self.save_session_callback(self.tracks)
        else:
            QMessageBox.information(self, "Save Session", "Save session is not available in this context.")


    def _init_ui_elements(self):
        """Initializes the UI widgets like table and slider."""
        if self.sorted_frame_ids:
            self.frame_slider.setMinimum(0)
            self.frame_slider.setMaximum(len(self.sorted_frame_ids) - 1)
        self.frame_slider.valueChanged.connect(self.on_slider_value_changed)

        self.tracks_table.setColumnCount(4)
        self.tracks_table.setHorizontalHeaderLabels(["Track ID", "Start Frame", "End Frame", "Length"])
        self.tracks_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.tracks_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.tracks_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.tracks_table.itemSelectionChanged.connect(self.on_track_selected)

        self.play_button = QPushButton("Play")
        self.play_button.setCheckable(True)
        self.play_button.clicked.connect(self.toggle_animation)
        self.controls_layout.insertWidget(0, self.play_button) 

    def enter_add_new_track_mode(self):
        """Switches the viewer to the mode for adding a new track."""
        self.image_viewer.set_mode('ADD_NEW_TRACK')

    def handle_add_new_track(self, scene_pos: QPointF):
        """Creates a new track and initiates tracking from the user's click."""
        next_track_id = max([t.track_id for t in self.tracks] + [0]) + 1
        
        current_frame_id = self.sorted_frame_ids[self.frame_slider.value()]
        
        roi_size = self.roi_size
        roi_half = roi_size // 2
        new_rect = QRect(int(scene_pos.x() - roi_half), int(scene_pos.y() - roi_half), roi_size, roi_size)
        
        new_spot = SpotROI(
            id=next_track_id,
            roi_rect=new_rect,
            center_x=scene_pos.x(),
            center_y=scene_pos.y(),
            intensity=0,
            source='manual_new'
        )
        
        new_track = Track(
            track_id=next_track_id, 
            spots_by_image_id={current_frame_id: new_spot}
        )
        
        self.tracks.append(new_track)
        
        self.populate_tracks_table()
        for row in range(self.tracks_table.rowCount()):
            if int(self.tracks_table.item(row, 0).text()) == next_track_id:
                self.tracks_table.selectRow(row)
                break
        self.show_frame(current_frame_id)
        
        print(f"Created new track {next_track_id} starting at frame {current_frame_id}.")


    def on_track_point_context_menu(self, spot_id: int, global_pos: QPoint):
        """Creates and shows the context menu for editing a track point."""
        current_row = self.tracks_table.currentRow()
        if current_row < 0: return

        menu = QMenu(self)
        
        delete_action = menu.addAction("Delete this point from track")
        
        menu.addSeparator()
        retrack_action = menu.addAction("Re-track from this point")
        
        action = menu.exec(global_pos)
        
        if action == delete_action:
            self.delete_point_from_track(spot_id)
        elif action == retrack_action:
            self.initiate_retrack(spot_id)

    def initiate_retrack(self, spot_id: int):
        """Prepares and launches the re-tracking worker."""
        current_row = self.tracks_table.currentRow()
        track_id_item = self.tracks_table.item(current_row, 0)
        selected_track_id = int(track_id_item.text())
        track_to_retrack = next((t for t in self.tracks if t.track_id == selected_track_id), None)
        
        current_frame_id = self.sorted_frame_ids[self.frame_slider.value()]
        
        if not track_to_retrack or current_frame_id not in track_to_retrack.spots_by_image_id:
            QMessageBox.warning(self, "Error", "Cannot start re-tracking. The selected track does not exist on the current frame.")
            return

        params = {
            'lk_win_size': self.lk_win_size,
            'lk_max_level': self.lk_max_level,
            'roi_size': self.roi_size 
        }

        worker = GenericWorker(
            retrack_single_spot_optical_flow,
            track_to_retrack,
            current_frame_id,
            self.file_map,
            self.sorted_frame_ids,
            params
        )
        self.launch_retrack_worker(worker)

    def launch_retrack_worker(self, worker: GenericWorker):
        """Runs the re-tracking worker with a progress dialog."""
        self.progress_dialog = QProgressDialog("Re-tracking spot...", "Cancel", 0, 100, self)
        self.progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
        self.progress_dialog.show()

        self.worker_thread = QThread()
        self.worker = worker
        self.worker.moveToThread(self.worker_thread)

        self.progress_dialog.canceled.connect(self.worker_thread.requestInterruption)
        self.worker.progress.connect(self.progress_dialog.setValue)
        self.worker.finished.connect(self.on_retrack_finished)
        self.worker.error.connect(self.on_retrack_error)

        self.worker.finished.connect(self.progress_dialog.accept)
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)

        self.worker_thread.started.connect(self.worker.run)
        self.worker_thread.start()

    def on_retrack_finished(self, updated_track: Track):
        """Handles the successful completion of the re-tracking task."""
        print(f"Re-tracking for track {updated_track.track_id} finished.")
        
        for i, track in enumerate(self.tracks):
            if track.track_id == updated_track.track_id:
                self.tracks[i] = updated_track
                break
        
        self.populate_tracks_table()
        current_frame_id = self.sorted_frame_ids[self.frame_slider.value()]
        self.show_frame(current_frame_id)
        
        QMessageBox.information(self, "Success", "Re-tracking completed successfully.")

    def on_retrack_error(self, error_message: str):
        """Handles errors from the re-tracking worker."""
        QMessageBox.critical(self, "Re-tracking Error", f"An error occurred:\n{error_message}")
    
    def handle_add_or_correct_point(self, scene_pos: QPointF):
        """Slot to handle adding a new point to or correcting an existing point in a track."""
        current_row = self.tracks_table.currentRow()
        if current_row < 0:
            QMessageBox.warning(self, "No Track Selected", "Please select a track from the table first.")
            return
        track_id_item = self.tracks_table.item(current_row, 0)
        if not track_id_item: return
        
        selected_track_id = int(track_id_item.text())
        selected_track = next((t for t in self.tracks if t.track_id == selected_track_id), None)
        if not selected_track: return

        current_frame_id = self.sorted_frame_ids[self.frame_slider.value()]

        # TODO: Pass roi_size from MainWindow
        roi_size = self.roi_size 
        roi_half = roi_size // 2
        new_rect = QRect(int(scene_pos.x() - roi_half), int(scene_pos.y() - roi_half), roi_size, roi_size)
        
        all_spot_ids_on_frame = {s.id for t in self.tracks for s in t.spots_by_image_id.values() if s.id is not None}
        next_spot_id = max(all_spot_ids_on_frame | {0}) + 1

        new_spot = SpotROI(
            id=next_spot_id,
            roi_rect=new_rect,
            center_x=scene_pos.x(),
            center_y=scene_pos.y(),
            intensity=0,
            source='manual_corr'
        )
        
        selected_track.spots_by_image_id[current_frame_id] = new_spot
        
        print(f"Updated track {selected_track.track_id} on frame {current_frame_id} with new spot {new_spot.id}")
        
        self.populate_tracks_table()
        self.show_frame(current_frame_id)

    def delete_point_from_track(self, spot_id_to_delete: int):
        """Finds the track containing the spot and removes the spot from it."""
        print(f"[Dialog] Attempting to delete spot with ID: {spot_id_to_delete}")

        slider_index = self.frame_slider.value()
        if not (0 <= slider_index < len(self.sorted_frame_ids)):
            print("[Dialog] ERROR: Invalid slider index. Aborting deletion.")
            return
        current_frame_id = self.sorted_frame_ids[slider_index]
        print(f"[Dialog] Current Frame ID is: {current_frame_id}")
        
        track_to_modify = None
        
        # Find the track that contains this spot on this frame
        for track in self.tracks:
            # Check if the track exists on the current frame
            if current_frame_id in track.spots_by_image_id:
                # Check if the spot on that frame has the ID we're looking for
                if track.spots_by_image_id[current_frame_id].id == spot_id_to_delete:
                    track_to_modify = track
                    print(f"[Dialog] Found spot in Track ID: {track.track_id}")
                    break
        
        if track_to_modify:
            print(f"[Dialog] Proceeding with deletion from track {track_to_modify.track_id}...")
            # Remove the spot entry for the current frame from the track's dictionary
            del track_to_modify.spots_by_image_id[current_frame_id]
            
            # If the track is now empty, remove it completely
            if not track_to_modify.spots_by_image_id:
                print(f"[Dialog] Track {track_to_modify.track_id} is now empty. Removing track.")
                self.tracks.remove(track_to_modify)

            # Refresh all UI elements
            print("[Dialog] Refreshing UI...")
            self.populate_tracks_table()
            self.show_frame(current_frame_id)
            self._update_highlights()
            print("[Dialog] Deletion and UI refresh complete.")
        else:
            print(f"[Dialog] ERROR: Could not find a track on frame {current_frame_id} containing spot with ID {spot_id_to_delete}.")

    
    def toggle_animation(self, checked: bool):
        """Starts or stops the frame-by-frame animation."""
        if checked:
            self.play_button.setText("Pause")
            self.animation_timer.start(100) # Czas w ms, 100ms = 10 klatek/s
        else:
            self.play_button.setText("Play")
            self.animation_timer.stop()

    def advance_frame(self):
        """Moves the slider to the next frame, looping at the end."""
        current_val = self.frame_slider.value()
        max_val = self.frame_slider.maximum()
        next_val = current_val + 1
        if next_val > max_val:
            next_val = self.frame_slider.minimum()
        self.frame_slider.setValue(next_val)

    def delete_selected_track(self):
        """Deletes the currently selected track from the table and the data model."""
        selected_items = self.tracks_table.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "No Selection", "Please select a track in the table to delete.")
            return

        track_id_to_delete = int(self.tracks_table.item(selected_items[0].row(), 0).text())

        reply = QMessageBox.question(self, "Confirm Deletion",
                                     f"Are you sure you want to delete Track ID: {track_id_to_delete}?",
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                     QMessageBox.StandardButton.No)

        if reply == QMessageBox.StandardButton.Yes:
            # Remove track from the main list
            self.tracks = [t for t in self.tracks if t.track_id != track_id_to_delete]
            # Refresh the UI
            self.populate_tracks_table()
            self.image_viewer.show_track_path(None) # Clear any drawn path

    def populate_tracks_table(self):
        """Fills the table with data from the tracks."""
        self.tracks_table.setRowCount(len(self.tracks))
        for row, track in enumerate(self.tracks):
            frame_ids = track.spots_by_image_id.keys()
            start_frame = min(frame_ids) if frame_ids else -1
            end_frame = max(frame_ids) if frame_ids else -1
            self.tracks_table.setItem(row, 0, QTableWidgetItem(str(track.track_id)))
            self.tracks_table.setItem(row, 1, QTableWidgetItem(str(start_frame)))
            self.tracks_table.setItem(row, 2, QTableWidgetItem(str(end_frame)))
            self.tracks_table.setItem(row, 3, QTableWidgetItem(str(len(frame_ids))))

    def on_slider_value_changed(self, index: int):
        """
        NEW: Slot that translates a slider index into a frame ID and calls show_frame.
        """
        if not self.sorted_frame_ids or not (0 <= index < len(self.sorted_frame_ids)):
            return
        
        # Translate index to the actual frame ID (time value)
        frame_id = self.sorted_frame_ids[index]
        self.show_frame(frame_id)
        
        # Update the label to be more informative
        self.frame_label.setText(f"Frame {index + 1}/{len(self.sorted_frame_ids)} (ID: {frame_id})")

    def on_track_selected(self):
        """Handles the table selection change by updating all highlights."""
        self._update_highlights()

    def _update_highlights(self):
        """
        Helper method to apply highlights to the viewer based on the current table selection.
        This should be called whenever the view or selection changes.
        """
        current_row = self.tracks_table.currentRow()
        if current_row < 0:
            self.image_viewer.highlight_roi(None)
            self.image_viewer.show_track_path(None)
            return

        id_item = self.tracks_table.item(current_row, 0)
        if not id_item: return
        track_id = int(id_item.text())
        
        selected_track = next((t for t in self.tracks if t.track_id == track_id), None)
        
        self.image_viewer.show_track_path(selected_track)
        
        if selected_track:
            # 1. Get the current SLIDER INDEX
            slider_index = self.frame_slider.value()

            # 2. Translate the INDEX to the actual FRAME ID
            if not (0 <= slider_index < len(self.sorted_frame_ids)):
                return
            current_frame_id = self.sorted_frame_ids[slider_index]
            
            # 3. Now use the correct frame_id to find the spot
            if current_frame_id in selected_track.spots_by_image_id:
                spot_id_to_highlight = selected_track.spots_by_image_id[current_frame_id].id
                self.image_viewer.highlight_roi(spot_id_to_highlight)
            else:
                self.image_viewer.highlight_roi(None)
        else:
            self.image_viewer.highlight_roi(None)

    def show_frame(self, frame_id: int, fit: bool = False):
        """Loads and displays the image and overlays for a given frame ID."""
        leed_file = self.file_map.get(frame_id)
        if not leed_file: return

        image_data = load_tiff(leed_file.filepath)
        pixmap = self._convert_numpy_to_pixmap(image_data)
        self.image_viewer.set_image(pixmap, fit=fit)

        spots_in_frame = []
        for track in self.tracks:
            if frame_id in track.spots_by_image_id:
                spots_in_frame.append(track.spots_by_image_id[frame_id])

        self.image_viewer.update_masks(self.masks)
        self.image_viewer.update_rois(spots_in_frame)

        self._update_highlights()

    def _convert_numpy_to_pixmap(self, image_data):
        """Helper to convert a NumPy array to a QPixmap."""
        if image_data is None or image_data.size == 0: return QPixmap()
        if image_data.dtype != np.uint8:
             min_val, max_val = image_data.min(), image_data.max()
             if max_val > min_val: image_data = (255 * (image_data - min_val) / (max_val - min_val)).astype(np.uint8)
             else: image_data = np.zeros_like(image_data, dtype=np.uint8)
        height, width = image_data.shape[:2]; bytes_per_line = width
        q_image = QImage(image_data.data, width, height, bytes_per_line, QImage.Format.Format_Grayscale8)
        return QPixmap.fromImage(q_image)

    def get_updated_tracks(self) -> List[Track]:
        """Returns the final, possibly edited, list of tracks."""
        return self.tracks
