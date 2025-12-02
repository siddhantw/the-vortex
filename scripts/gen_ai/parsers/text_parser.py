from .base_parser import BaseParser

class TextParser(BaseParser):
    def parse(self, file_path: str) -> dict:
        with open(file_path, 'r') as f:
            content = f.read()
        # Handle empty file case
        if not content.strip():
            return {"raw_text": "[Empty file]"}
        # Handle non-UTF-8 encoding
        try:
            content.encode('utf-8')
        except UnicodeEncodeError:
            return {"raw_text": "[Encoding error: non-UTF-8 content]"}
        # Handle other potential errors
        except Exception as e:
            return {"raw_text": f"[Text parsing error: {e}]"}
        # Return the parsed content
        return {"raw_text": content}

