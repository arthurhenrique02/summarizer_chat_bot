import typing

from fastapi import File, UploadFile

ALLOWED_FILE_EXTENSIONS = [".txt", ".pdf", ".docx", ".md"]


def check_file_extension(
    filename: str, allowed_extensions: list[str] = ALLOWED_FILE_EXTENSIONS
) -> bool:
    """Check if the file has an allowed extension."""
    return any(filename.lower().endswith(ext) for ext in allowed_extensions)


def read_file_content(file: typing.Annotated[UploadFile, File()]) -> str:
    """Read the content of an uploaded file."""
    content = file.file.read()
    return content.decode("utf-8", errors="ignore")
