import sys
import os
import traceback

def error_message_detail(error, error_detail: sys):
    """Formats the error message with file name, line number, and error description."""
    _, _, exc_tb = error_detail.exc_info()
    if exc_tb:
        file_name = exc_tb.tb_frame.f_code.co_filename
        line_number = exc_tb.tb_lineno
    else:
        file_name = "Unknown"
        line_number = "Unknown"

    error_message = (
        f"Error occurred in Python script: [{file_name}] "
        f"at line number [{line_number}]. "
        f"Error message: [{str(error)}]"
    )
    return error_message

class ProjectException(Exception):
    """Custom Exception class for the project."""
    def __init__(self, error_message, error_detail: sys = sys):
        super().__init__(error_message)  # Initialize the base Exception class
        self.error_message = error_message_detail(error_message, error_detail)

    def __str__(self):
        return self.error_message
