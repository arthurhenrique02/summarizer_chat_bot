import typing
from abc import ABC, abstractmethod

from langchain.prompts import PromptTemplate
from langchain_huggingface.embeddings import HuggingFaceEmbeddings


class IHuggingFaceBot(ABC):
    prompt_template: PromptTemplate
    name: str
    model: str
    llm: typing.Any

    def __init__(self, name: str, prompt: PromptTemplate, model: str):
        self.name = name
        self.prompt_template = prompt
        self.model = model

    @abstractmethod
    def task(self): ...

    @abstractmethod
    def start_llm(self): ...


class EmbeddingBot(IHuggingFaceBot):
    def __init__(self, name: str, prompt: PromptTemplate, model: str):
        super().__init__(name, prompt, model)
        self.llm = self.start_llm()

    def task(self):
        # TODO implement embedding task
        return "embedding"

    def start_llm(self):
        return HuggingFaceEmbeddings(model_name=self.model)


# TODO CREATE SUMMARIZATION BOT
