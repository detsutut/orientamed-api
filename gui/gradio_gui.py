import time

import gradio as gr
from boto3 import Session
import logging

from core.data_models import LLMResponse
from utils.login import login, verify_token, log_usage, get_role, check_ban, check_daily_token_limit, set_softban
from utils.stats import get_usage_statistics
from core.utils import get_mfa_response
from rag import update_rag, rag_invoke, rag_schema

logger = logging.getLogger('app.' + __name__)


def user_login(user: str, pw: str, mfa_token: str | None = None):
    token = login(user, pw)
    if token and mfa_token:
        mfa_response = get_mfa_response(mfa_token)
        if mfa_response:
            logger.info("Establishing session with AWS bedrock service...")
            session = Session(aws_access_key_id=mfa_response['Credentials']['AccessKeyId'],
                              aws_secret_access_key=mfa_response['Credentials']['SecretAccessKey'],
                              aws_session_token=mfa_response['Credentials']['SessionToken'])
            update_rag(session)
        else:
            logger.error(
                "Impossible to establish a session with AWS bedrock service. Check your MFA token and try again. If the problem persists, contact the developer.")
    return token

def get_stats(token):
    user = verify_token(token)
    if not user:
        gr.Warning("Invalid token",  duration=10)
        return None
    statistics = get_usage_statistics()
    return statistics

def get_img(token):
    user = verify_token(token)
    if not user:
        gr.Warning("Invalid token",  duration=10)
        return None
    return rag_schema()

def reply(user_input, emb, graph, qa, ro, reranker, pre_translate,max_refs,check_consistency, token, r: gr.Request) -> LLMResponse:
    user = verify_token(token)
    if user:
        user_role = get_role(user)
        if user_role!="admin" and check_ban(user):
            gr.Warning("User is banned",  duration=10)
            return None
        start_time = time.time()
        response: LLMResponse = rag_invoke(query=user_input,
                                             query_aug=qa,
                                             use_graph=graph,
                                             retrieve_only=ro,
                                             reranker=reranker,
                                             pre_translate=pre_translate,
                                             max_refs=max_refs,
                                             check_consistency=check_consistency,
                                             use_embeddings=emb)
        duration_ms = int((time.time() - start_time) * 1000)
        log_usage(username=user,
                  token_in=response.consumed_tokens.input,
                  token_out=response.consumed_tokens.output,
                  duration_ms=duration_ms,
                  session_id=token.split(".")[-1],
                  ip_address=r.client.host)
        if user_role != "admin":
            over = check_daily_token_limit(username=user, role=user_role)
            if over:
                logger.warning("User has exceeded the daily token limit.")
                set_softban(username=user)
        return response.model_dump()
    return None


with gr.Blocks(title="Debug App") as gui:
    with gr.Group("Login"):
        with gr.Row():
            user = gr.Text(label="Username")
            pw = gr.Text(label="Password", type="password")
            token = gr.Text(label="Token")
            mfa_token = gr.Text(label="MFA Token (Local Deployment Only)", type="password")
        login_btn = gr.Button("Login")
    with gr.Group("Input"):
        user_input = gr.Textbox(label="Input message")
        with gr.Accordion("Options"):
            with gr.Row():
                emb = gr.Checkbox(label="Embedding-based Retrieval", value=True)
                graph = gr.Checkbox(label="Graph-based Retrieval", value=True)
                ro = gr.Checkbox(label="Retrieval Only", value=False)
                qa = gr.Checkbox(label="Query Augmentation", value=False)
                pre_translate = gr.Checkbox(label="Pre-Translate", value=True)
                check_consistency = gr.Checkbox(label="Consistency Check", value=True)
            with gr.Row():
                reranker = gr.Dropdown(label="Reranker", value="RRF", choices=["RRF", "top_k"])
                max_refs = gr.Slider(label="Max References", value=5, minimum=1, maximum=20, step=1)
        stats_btn = gr.Button("Get Stats", variant="secondary")
        submit_btn = gr.Button("Send Message", variant="primary")
    with gr.Group():
        response = gr.JSON(label="Output")
    with gr.Group():
        schema = gr.Image(label="Output")
        schema_btn = gr.Button("Load", variant="primary")

    schema_btn.click(get_img, inputs=[token], outputs=[schema])
    login_btn.click(user_login, inputs=[user, pw, mfa_token], outputs=[token])
    stats_btn.click(get_stats, inputs=[token], outputs=[response])
    pw.submit(user_login, inputs=[user, pw, mfa_token], outputs=[token])
    submit_btn.click(reply, inputs=[user_input, emb, graph, qa, ro, reranker, pre_translate, max_refs, check_consistency, token], outputs=[response])
