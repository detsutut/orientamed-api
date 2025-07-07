import time

import gradio as gr
from boto3 import Session
import logging

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

def reply(user_input, emb, graph, qa, ro, token, r: gr.Request):
    user = verify_token(token)
    if user:
        user_role = get_role(user)
        if user_role!="admin" and check_ban(user):
            gr.Warning("User is banned",  duration=10)
            return None
        start_time = time.time()
        request = rag_invoke(query=user_input,
                             query_aug=qa,
                             use_graph=graph,
                             retrieve_only=ro,
                             use_embeddings=emb)
        duration_ms = int((time.time() - start_time) * 1000)
        if request:
            log_usage(username=user,
                      token_in=request.get("input_tokens_count", 0),
                      token_out=request.get("output_tokens_count", 0),
                      duration_ms=duration_ms,
                      session_id=token.split(".")[-1],
                      ip_address=r.client.host)
            if user_role != "admin":
                over = check_daily_token_limit(username=user, role=user_role)
                if over:
                    logger.warning("User has exceeded the daily token limit.")
                    set_softban(username=user)
            return request
    return None


with gr.Blocks(title="Debug App") as gui:
    with gr.Row():
        user = gr.Text(label="Username")
        pw = gr.Text(label="Password", type="password")
        token = gr.Text(label="Token")
        mfa_token = gr.Text(label="MFA Token (Local Deployment Only)", type="password")
        login_btn = gr.Button("Login")
    with gr.Group():
        with gr.Row():
            user_input = gr.Textbox(label="Input")
            stats_btn = gr.Button("Stats", variant="secondary")
        with gr.Row():
            emb = gr.Checkbox(label="Embeddings", value=True)
            graph = gr.Checkbox(label="Graphs", value=True)
            qa = gr.Checkbox(label="Query Aug", value=False)
            ro = gr.Checkbox(label="Retrieve Only", value=False)
        with gr.Row():
            submit_btn = gr.Button("Send", variant="primary")
    with gr.Group():
        response = gr.JSON(label="Output")
    with gr.Group():
        schema = gr.Image(label="Output")
        schema_btn = gr.Button("Load", variant="primary")

    schema_btn.click(get_img, inputs=[token], outputs=[schema])
    login_btn.click(user_login, inputs=[user, pw, mfa_token], outputs=[token])
    stats_btn.click(get_stats, inputs=[token], outputs=[response])
    pw.submit(user_login, inputs=[user, pw, mfa_token], outputs=[token])
    submit_btn.click(reply, inputs=[user_input, emb, graph, qa, ro, token], outputs=[response])
