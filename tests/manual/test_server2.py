import gradio as gr
def echo(x): return x + "!"
with gr.Blocks() as demo:
    txt = gr.Textbox()
    gr.Button().click(fn=echo, inputs=txt, outputs=txt, api_name="predict")
demo.launch(server_port=7862, server_name="127.0.0.1")
