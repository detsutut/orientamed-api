import json

import gradio as gr
import requests


def login(user, pw, mfa_token):
    response = requests.post("http://localhost:8000/auth/login",
                             data=json.dumps({"username": user, "password": pw}),
                             headers={"ContentType": "application/json"})
    if response.status_code == 200:
        token = response.json()["access-token"]
        if mfa_token:
            response = requests.post("http://localhost:8000/auth/init",
                          data={"access_token": token, "mfa_token": mfa_token})
            return token
    else:
        token = None
    return token


def reply(user_input, emb, graph, qa, ro, token):
    url = "http://localhost:8000/generate"
    data = {'user_input': user_input,
            'history': [],
            "additional_context": "",
            "augment_query": qa,
            "use_graph": graph,
            "retrieve_only": ro,
            "use_embeddings": emb
            }
    params = {'access_token': token}
    res = requests.post(url,
                        data=json.dumps(data),
                        params=params,
                        headers={"ContentType": "application/json"})
    return res.json()


with gr.Blocks(title="Debug App") as app:
    with gr.Row():
        user = gr.Text(label="Username")
        pw = gr.Text(label="Password", type="password")
        token = gr.Text(label="Token")
        mfa_token = gr.Text(label="MFA Token (Local Deployment Only)", type="password")
        login_btn = gr.Button("Login")
    with gr.Group():
        with gr.Row():
            user_input = gr.Textbox(label="Input")
        with gr.Row():
            emb = gr.Checkbox(label="Embeddings", value=True)
            graph = gr.Checkbox(label="Graphs", value=True)
            qa = gr.Checkbox(label="Query Aug", value=False)
            ro = gr.Checkbox(label="Retrieve Only", value=False)
        with gr.Row():
            submit_btn = gr.Button("Send", variant="primary")
    with gr.Group():
        response = gr.JSON(label="Output")

    login_btn.click(login, inputs=[user, pw, mfa_token], outputs=[token])
    pw.submit(login, inputs=[user, pw, mfa_token], outputs=[token])
    submit_btn.click(reply, inputs=[user_input, emb, graph, qa, ro, token], outputs=[response])
