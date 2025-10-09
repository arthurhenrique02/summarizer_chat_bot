import typing

from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from langchain_community.document_transformers import EmbeddingsClusteringFilter
from langchain_core.documents import Document
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_huggingface.llms import HuggingFacePipeline
from transformers import AutoTokenizer, pipeline

from services.utils import ADocumentProcessor, AHuggingFaceBot


class EmbeddingBot(AHuggingFaceBot, ADocumentProcessor):
    def __init__(self, name: str, prompt: PromptTemplate, model: str):
        super().__init__(name, prompt, model)
        self.llm = self.start_llm()

    def task(self, text: str, *args, **kwargs):
        docs = self.separate_text_into_documents(text)

        # Embed and cluster text by using k-means clustering
        filter_cluster = EmbeddingsClusteringFilter(
            embeddings=self.llm, num_clusters=len(docs)
        )

        return filter_cluster.transform_documents(documents=docs)

    def start_llm(self):
        return HuggingFaceEmbeddings(
            model_name=self.model,
            model_kwargs={
                "device": "cuda",
            },
            encode_kwargs={"normalize_embeddings": True},
        )


class SummarizationBot(AHuggingFaceBot, ADocumentProcessor):
    llm: HuggingFacePipeline

    def __init__(self, name: str, prompt: PromptTemplate, model: str):
        super().__init__(name, prompt, model)
        self.llm = self.start_llm()

    def task(self, documents: typing.List[Document], *args, **kwargs):
        return load_summarize_chain(
            self.llm,
            chain_type="map_reduce",
            token_max=1000,
        ).invoke({"input_documents": documents})

    def start_llm(self) -> HuggingFacePipeline:
        return HuggingFacePipeline(
            pipeline=pipeline(
                "summarization",
                model=self.model,
                tokenizer=AutoTokenizer.from_pretrained(
                    self.model, from_tf=False, from_flax=False, use_fast=True
                ),
                device_map="cpu",
                framework="pt",
                max_new_tokens=1000,
            ),
        )
