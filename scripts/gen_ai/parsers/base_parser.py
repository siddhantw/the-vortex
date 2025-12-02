# Base parser class for all requirement parsers
class BaseParser:
    def parse(self, file_path: str) -> dict:
        raise NotImplementedError("parse() must be implemented by subclasses.")
    def get_file_type(self, file_path: str) -> str:
        """
        Get the file type based on the file extension.
        :param file_path:
        :return:
        """
        file_type = file_path.split('.')[-1].lower()
        return file_type
    def get_file_name(self, file_path: str) -> str:
        """
        Get the file name without the extension.
        :param file_path:
        :return:
        """
        file_name = file_path.split('/')[-1].split('.')[0]
        return file_name
    def get_file_extension(self, file_path: str) -> str:
        """
        Get the file extension.
        :param file_path:
        :return:
        """
        file_extension = file_path.split('.')[-1].lower()
        return file_extension
    def get_file_size(self, file_path: str) -> int:
        """
        Get the file size in bytes.
        :param file_path:
        :return:
        """
        import os
        file_size = os.path.getsize(file_path)
        return file_size
