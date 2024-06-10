import gradio as gr
import base64
import main

def image_to_base64(path):
    with open(path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded_string

img_path = 'img/initial_img.png'
base64_string = image_to_base64(img_path)

with open("style.css", "r") as file:
    css = file.read().replace('MARKER_FOR_BASE64', base64_string)

def open_main_interface():
    return gr.update(visible=False), gr.update(visible=True)

def initial_interface():
    with gr.Blocks(css=css) as demo:
        initial_section = gr.Column(visible=True)
        main_section = gr.Column(visible=False)
        
        with initial_section:
            gr.Markdown("<div style='height: 465px;'>&nbsp;</div>")
            with gr.Row():
                button = gr.Button("START")
                button.click(open_main_interface, inputs=[], outputs=[initial_section, main_section])
        
        with main_section:
            main.main_interface()

        return demo

if __name__ == "__main__":
    initial_interface().launch()
