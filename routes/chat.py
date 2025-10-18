import os
import typing

from dotenv import load_dotenv
from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import JSONResponse
from langchain_core.prompts import PromptTemplate

from core.services.bot import EmbeddingBot, SummarizationBot
from core.services.utils import read_file_content

load_dotenv()

blueprint_name = "chat"

router = APIRouter(
    prefix=f"/{blueprint_name}",
    tags=[blueprint_name],
    responses={404: {"description": "Not found"}},
)


embedding_bot = EmbeddingBot(
    name="embedding_bot",
    prompt=None,
    model=os.getenv("EMBEDDING_MODEL"),
)

summarization_bot = SummarizationBot(
    name="summarization_bot",
    prompt=PromptTemplate(
        input_variables=["text"],
        template=(
            "Summarize the following text in a concise, detailed and clear manner,"
            " giving max relevant information you could."
            " Text:\n\n'{text}'\n\n"
        ),
    ),
    model="google/pegasus-large",
    combine_prompt=PromptTemplate(
        input_variables=["text"],
        template=(
            "Given the following summaries, create a final summary that is concise, detailed and clear,"
            " giving max relevant information you could,"
            " returning only the summary, a string text without any additional information and any meta-text."
            " Summaries:\n\n'{text}'\n\n"
        ),
    ),
)


@router.post("/summarize")
async def summarize(
    chat_message: typing.Annotated[str, Form()],
    file: typing.Annotated[UploadFile | None, File()] = None,
):
    f_content = read_file_content(file) if file else ""
    chat_message += "\n\n" + f_content
    embedded_documents = embedding_bot.task(chat_message)
    summary = summarization_bot.task(embedded_documents)

    return JSONResponse(content={"summary": summary}, status_code=200)
