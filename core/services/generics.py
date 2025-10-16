import typing
from abc import ABC, abstractmethod

from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


class ADocumentProcessor(ABC):
    def separate_text_into_documents(self, text: str) -> typing.List[Document]:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=50,
            length_function=len,
        )
        return [
            Document(page_content=chunk)
            for chunk in text_splitter.split_text(text)
            if chunk
        ]


class AHuggingFaceBot(ABC):
    prompt_template: PromptTemplate
    name: str
    model: str
    llm: typing.Any

    def __init__(self, name: str, prompt: PromptTemplate | None, model: str):
        if prompt is not None and not isinstance(prompt, PromptTemplate):
            raise ValueError("Prompt must be an instance of PromptTemplate")
        self.name = name
        self.prompt_template = prompt
        self.model = model

    @abstractmethod
    def task(self, *args, **kwargs): ...

    @abstractmethod
    def start_llm(self): ...
