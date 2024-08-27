import sys
from src.logger import logging
def error_message_details(error, error_details:sys):
    _, _, tb = sys.exc_info()
    file_name  = tb.tb_frame.f_code.co_filename
    error_message = 'error in file: ' + file_name + 'Line no: ' + str(tb.tb_lineno) + 'Error: ' + str(error)
    return error_message

class CustomException(Exception):

    def __init__(self, error_message, error_details:sys):
        super().__init__(error_message)
        self.error_message = error_message_details(error_message, error_details)

    def __str__(self) -> str:
        return self.error_message



