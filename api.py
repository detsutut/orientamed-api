import os
import time

from dotenv import find_dotenv, load_dotenv
from fastapi import FastAPI
from typing import Union
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field
import yaml
import logging
from logging.handlers import RotatingFileHandler
import uvicorn
import gradio as gr

from core.data_models import LLMResponse

############# .ENV ####################

if find_dotenv():
    load_dotenv()

if find_dotenv("secrets.env"):
    load_dotenv("secrets.env")
elif find_dotenv("core/secrets.env"):
    load_dotenv("core/secrets.env")
else:
    raise FileNotFoundError("No secrets.env file found.")

############# LOCAL MODULES ####################

from rag import rag_invoke
from utils.login import verify_token, login, get_role, log_usage, check_ban, check_daily_token_limit, set_softban
from utils.stats import get_usage_statistics
from gui import gradio_gui

############# SETTINGS ##################
with open(os.getenv("API_SETTINGS_PATH")) as stream:
    api_config = yaml.safe_load(stream)
with open(os.getenv("CORE_SETTINGS_PATH")) as stream:
    rag_config = yaml.safe_load(stream)


############# LOGGER ##################
# Create a custom logger
logger = logging.getLogger("app")
logger.setLevel(logging.DEBUG if api_config.get("debug",False) else logging.INFO)  # Capture all logs

# Console handler - prints everything
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG if api_config.get("debug",False) else logging.INFO)
console_formatter = logging.Formatter("%(asctime)s - [%(levelname)s] - %(name)s - %(message)s")
console_handler.setFormatter(console_formatter)

# File handler - saves only ERROR and above
file_handler = RotatingFileHandler("logs.log", maxBytes=api_config.get("logs").get("max-size-mb",1)*1000000, backupCount=api_config.get("logs").get("num-backup",1))
file_handler.setLevel(logging.INFO)
file_formatter = logging.Formatter("%(asctime)s - [%(levelname)s] - %(name)s - %(message)s")
file_handler.setFormatter(file_formatter)

# Add handlers to the logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)

############ DATA MODELS ################

class Credentials(BaseModel):
    username: str = Field(description="Username")
    password: str = Field(description="Password")

class GenerateQueryParams(BaseModel):
    user_input: str = Field(description="Query to generate answer")
    history: list[dict] = Field(default=[],
                                description="History as a list of dictionaries with openai-style 'role' and 'content' keys")
    additional_context: Union[str, None] = Field(default=None, description="Additional context to use for the query")
    augment_query: bool = Field(default=False, description="Augmented query")
    retrieve_only: bool = Field(default=False, description="Retrieve only")
    use_graph: bool = Field(default=True, description="Use graph")
    use_embeddings: bool = Field(default=True, description="Use embeddings")
    reranker: str = Field(default="RRF", description="Reranker type. Options: RRF or top_k")
    check_consistency: bool = Field(default=False, description="Check answer consistency with graph")
    max_refs: int = Field(default=5, description="Max retrieved references to use to answer")
    pre_translate: bool = Field(default=False, description="Use preliminary LLM translation in concept extraction or delegate it to the concept extractor")


############### API ######################
app = FastAPI(title="OrientaMed",
              contact={"name": "Tommaso Buonocore",
                       "url": "https://github.com/detsutut",
                       "email": "buonocore.tms@gmail.com"})

@app.get("/")
def read_root():
    return {}


@app.get('/favicon.ico', include_in_schema=False)
async def favicon():
    return FileResponse("favicon.ico")


@app.get("/auth/check", response_model=dict)
async def check():
    return {
        "status": "ok",
    }

@app.post("/auth/check", response_model=dict)
async def check(access_token: str):
    payload = verify_token(access_token)
    return {
        "status": "ok",
        "logged": True if payload else None,
    }

@app.post("/auth/login", response_model=dict)
async def log(credentials: Credentials):
    token = login(credentials.username, credentials.password)
    if token:
        return JSONResponse(content={
            "status": "ok",
            "access-token": token,
            "role": get_role(credentials.username)
        })
    else:
        return JSONResponse(content={
            "status": "error",
            "access-token": None,
            "error": "Invalid username or password"
        }, status_code=401)

@app.post("/stats", response_model=dict)
async def stats(access_token: str):
    user = verify_token(access_token)
    if not user:
        return JSONResponse(content={"error": "Invalid token."}, status_code=401)
    try:
        statistics = get_usage_statistics()
        if statistics:
            return JSONResponse(content=statistics, status_code=200)
        else:
            return JSONResponse(content={"error": "No data found."}, status_code=401)
    except Exception as e:
        logger.error(e)
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/generate", response_model=dict)
def generate(query: GenerateQueryParams, access_token: str):
    user = verify_token(access_token)
    if not user:
        return JSONResponse(content={"error": "Invalid token."}, status_code=401)
    if not query.user_input:
        return JSONResponse(content={"error": "Please provide a text."}, status_code=400)
    user_role = get_role(user)
    if user_role != "admin" and check_ban(user):
        return JSONResponse(content={"error": "User is banned."}, status_code=403)
    try:
        start_time = time.time()
        response: LLMResponse = rag_invoke(query=query.user_input,
                              history=query.history,
                              additional_context=query.additional_context,
                              query_aug=query.augment_query,
                              retrieve_only=query.retrieve_only,
                              use_graph=query.use_graph,
                              use_embeddings=query.use_embeddings,
                              reranker=query.reranker,
                              pre_translate=query.pre_translate,
                              max_refs=query.max_refs,
                              check_consistency=query.check_consistency)
        duration_ms = int((time.time() - start_time) * 1000)
        log_usage(username=user,
                  token_in=response.consumed_tokens.input,
                  token_out=response.consumed_tokens.output,
                  duration_ms=duration_ms,
                  session_id=access_token.split(".")[-1])
        if user_role != "admin":
            over = check_daily_token_limit(username=user, role=user_role)
            if over:
                logger.warning("User has exceeded the daily token limit.")
                set_softban(username=user)
        return JSONResponse(content=response.model_dump(), status_code=200)
    except Exception as e:
        logger.error(e)
        return JSONResponse(content={"error": str(e)}, status_code=500)

app = gr.mount_gradio_app(app, gradio_gui.gui, path="/debug/gui")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)