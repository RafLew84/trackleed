# stal/main.py

import sys
from PyQt6.QtWidgets import QApplication

# Import our custom MainWindow class from the corresponding module
from trackleed.gui.main_window import MainWindow

def main():
    """
    Main function to initialize and run the application.
    """
    # Create the application instance. sys.argv allows passing command-line arguments.
    app = QApplication(sys.argv)

    # Create an instance of our main window
    window = MainWindow()

    # Show the window on the screen
    window.show()

    # Start the application's event loop and handle a clean exit
    sys.exit(app.exec())

# Standard boilerplate to run the main function when the script is executed directly
if __name__ == '__main__':
    main()
