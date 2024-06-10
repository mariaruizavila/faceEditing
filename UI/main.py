import gradio as gr
from PIL import Image, ImageDraw
import sys
import os
import torch
import torch.utils.data as data
import argparse
from torchvision.utils import make_grid
from torchvision.transforms import ToPILImage
import yaml
import clip
import torchvision

sys.path.append('../')
import models
from models.interface_trainer import Trainer as InterFaceTrainer
from models.tedi_trainer import Trainer as TediTrainer
from models.styleclip_trainer import Trainer as StyleCLIPTrainer
from models.multi_trainer import Trainer as MultiTrainer
from models.stylegan2.model import Generator


from constants import ATTR_TO_NUM, LOG_DIR

torch.autograd.set_detect_anomaly(True)
Image.MAX_IMAGE_PIXELS = None
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    DEVICE = torch.device('cuda')
else:
    os.environ["TORCH_CUDA_ARCH_LIST"] = "Turing"
    DEVICE = torch.device('cpu')
    
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="new_train", help="Path to the config file.")
parser.add_argument("--label_file", type=str, default="./data/celebahq_anno.npy", help="label file path")
parser.add_argument("--stylegan_model_path", type=str,default=".models/pixel2style2pixel/pretrained_models/psp_ffhq_encode.pt",help="stylegan model path")
parser.add_argument("--classifier_model_path", type=str, default="./models/latent_classifier_epoch_20.pth", help="pretrained attribute classifier")
parser.add_argument("--log_path", type=str, default=".models/logs/", help="log file path")
parser.add_argument("--out", type=str, default="./prepared_data/output", help="Name of the out folder")
opts = parser.parse_args()

config = yaml.safe_load(open('./configs/' + "performance" + '.yaml', 'r'))
attrs = config['attr'].split(',')
attr_num = [ATTR_TO_NUM[a] for a in attrs]

log_dir = os.path.join(LOG_DIR, "performance") + '/'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

def get_trainer(model=1, log_dir=log_dir, attr_num=attr_num, attrs=attrs):
    if model == 1: #multi
        trainer = MultiTrainer(attr_num, attrs, opts.label_file)
        trainer.load_model_multi(log_dir, model)
    elif model == 2: #interfaceGAN
        trainer = InterFaceTrainer(attr_num, attrs, opts.label_file)
    elif model == 3: #tediGAN
        trainer = TediTrainer(attr_num, attrs, opts.label_file)
    else: #styleCLIP
        trainer = StyleCLIPTrainer(attr_num, attrs, opts.label_file)
    trainer.initialize(opts.stylegan_model_path, opts.classifier_model_path)
    trainer.to(DEVICE)
    return trainer

def load_image(image_path):
    return Image.open(image_path)

def modify_image_interface(img, atts):
    coeff = torch.tensor(atts, dtype=torch.float32)
    trainer = get_trainer(2)
    w_0 = transform_img(img)
    w_1 = trainer.T_net(
        w_0.view(w_0.size(0), -1), coeff.unsqueeze(0).to(DEVICE), scaling=1
    )
    w_1 = w_1.view(w_0.size())
    result = ToPILImage()(make_grid(w_1.detach().cpu(), normalize=True, scale_each=True, range=(-1, 1), padding=0))
    torchvision.utils.save_image(result.detach().cpu(), os.path.join(opts.results_dir, "result.png"), normalize=True, scale_each=True, range=(-1, 1))
    return result

def modify_image_tedi(img, description):
    trainer = trainer(3)
    text = torch.cat([clip.tokenize(description)]).cuda()
    g_ema = Generator(1024, 512, 8)
    g_ema.eval()
    g_ema = g_ema.cuda()
    initial_latent_code = transform_img(img)
    _, latent_code = g_ema([initial_latent_code], return_latents=True, truncation=0.7)
    latent = latent_code.detach().clone()
    latent = trainer.adapt_latent_with_text(latent, text)
    result, _ = g_ema([latent], input_is_latent=True, randomize_noise=False)
    torchvision.utils.save_image(result.detach().cpu(), os.path.join(opts.results_dir, "result.png"), normalize=True, scale_each=True, range=(-1, 1))
    return result

def modify_image_styleclip(img, description):
    trainer = get_trainer(4)
    text = torch.cat([clip.tokenize(description)]).cuda()
    g_ema = Generator(1024, 512, 8)
    g_ema.eval()
    g_ema = g_ema.cuda()
    latent_code_init_not_trunc = transform_img(img)
    _, latent_code_init = g_ema([latent_code_init_not_trunc], return_latents=True,
                                truncation=0.7)
    latent = trainer.adjust_latent_with_text(latent, text)
    result, _ = g_ema([latent, text], input_is_latent=True, randomize_noise=False)
    torchvision.utils.save_image(result.detach().cpu(), os.path.join(opts.results_dir, "result.png"), normalize=True, scale_each=True, range=(-1, 1))
    return result

def modify_image_multi(img, atts):
    coeff = torch.tensor(atts, dtype=torch.float32)
    trainer = get_trainer(1)
    w_0 = transform_img(img)
    w_1 = trainer.T_net(
        w_0.view(w_0.size(0), -1), coeff.unsqueeze(0).to(DEVICE), scaling=1
    )
    w_1 = w_1.view(w_0.size())
    result = ToPILImage()(make_grid(w_1.detach().cpu(), normalize=True, scale_each=True, range=(-1, 1), padding=0))
    torchvision.utils.save_image(result.detach().cpu(), os.path.join(opts.results_dir, "result.png"), normalize=True, scale_each=True, range=(-1, 1))
    return result

def transfomrm_text_input(atts):
    text = ""
    coeff = torch.tensor(atts, dtype=torch.float32)
    for i in range (0, len(coeff)):
        if len(text) != 0:
            text += " and "
        if coeff[i] == 0:
            break
        if coeff[i] == 1:
            text += "add "
        elif coeff[i] == 2:
            text += "add a lot of "
        elif coeff[i] == -1:
            text += "decrease "
        elif coeff[i] == -2:
            text += "remove "
        if i == 0:
            text += "bald"
        elif i == 1:
            text += "male"
        elif i == 2:
            text += "smiling"
        elif i == 3:
            text += "big lips"
        elif i == 4:
            text += "young"
        elif i == 5:
            text += "no beard"
        elif i == 6:
            text += "big nose"
        elif i == 7:
            text += "pale skin"
        elif i == 8:
            text += "narrow eyes"
        elif i == 9:
            text += "attractive"
        elif i == 10:
            text += "bangs"
        elif i == 11:
            text += "lipstick"
        elif i == 12:
            text += "eyeglasses"
        elif i == 13:
            text += "high cheekbones"
        elif i == 14:
            text += "gray hair"
        elif i == 15:
            text += "blond hair"
        elif i == 16:
            text += "bushy eyebrows"
        elif i == 17:
            text += "chubby"
        elif i == 18:
            text += "wavy hair"
        elif i == 19:
            text += "arched eyebrows"
            
    return text

def modify_images_comparison(img, atts1, atts2):
    atts = atts1 + atts2
    text = transfomrm_text_input(atts)
    r_interface = modify_image_interface(img, atts)
    r_multi = modify_image_multi(img, atts)
    r_tedi = modify_image_tedi(img, text)
    r_styleclip = modify_image_tedi(img, text)
    return r_interface, r_tedi, r_styleclip, r_multi

def transform_img(img):
    if isinstance(img, Image.Image):
        img_tensor = model.encoder.preprocess(img).unsqueeze(0).cuda()  
    else:
        raise TypeError("La imagen de entrada debe ser un objeto PIL.Image")

    with torch.no_grad():
        latent_code = model.encoder.encode(img_tensor)
    
    return latent_code

def main_interface():
    with gr.Blocks(css="style.css") as demo:
        with gr.Column(elem_classes="main-background"):
            gr.Markdown("<h1>Welcome to the Face Editor!</h1>")
            description = ''
            with gr.Tabs():
                with gr.Tab("InterFaceGAN"):
                    with gr.Tabs():
                        with gr.Tab("Usage"):
                            with gr.Row():
                                with gr.Column(scale=1):
                                    atts = [
                                        gr.Slider(minimum=-2, maximum=2, label="Bald", step=1, value=0),
                                        gr.Slider(minimum=-2, maximum=2, label="Male", step=1, value=0),
                                        gr.Slider(minimum=-2, maximum=2, label="Smiling", step=1, value=0),
                                        gr.Slider(minimum=-2, maximum=2, label="Big Lips", step=1, value=0),
                                        gr.Slider(minimum=-2, maximum=2, label="Young", step=1, value=0),
                                        gr.Slider(minimum=-2, maximum=2, label="No Beard", step=1, value=0),
                                        gr.Slider(minimum=-2, maximum=2, label="Big Nose", step=1, value=0),
                                        gr.Slider(minimum=-2, maximum=2, label="Pale Skin", step=1, value=0),
                                        gr.Slider(minimum=-2, maximum=2, label="Narrow Eyes", step=1, value=0),
                                        gr.Slider(minimum=-2, maximum=2, label="Bangs", step=1, value=0),
                                        gr.Slider(minimum=-2, maximum=2, label="Attractive", step=1, value=0),
                                        gr.Slider(minimum=-2, maximum=2, label="Lipstick", step=1, value=0),
                                        gr.Slider(minimum=-2, maximum=2, label="Eyeglasses", step=1, value=0),
                                        gr.Slider(minimum=-2, maximum=2, label="High Cheekbones", step=1, value=0),
                                        gr.Slider(minimum=-2, maximum=2, label="Gray Hair", step=1, value=0),
                                        gr.Slider(minimum=-2, maximum=2, label="Blond Hair", step=1, value=0),
                                        gr.Slider(minimum=-2, maximum=2, label="Bushy Eyebrows", step=1, value=0),
                                        gr.Slider(minimum=-2, maximum=2, label="Chubby", step=1, value=0),
                                        gr.Slider(minimum=-2, maximum=2, label="Wavy Hair", step=1, value=0),
                                        gr.Slider(minimum=-2, maximum=2, label="Arched Eyebrows", step=1, value=0),
                                    ]

                                with gr.Column(scale=1):
                                    img = gr.Image(type="pil", label="Input")
                                    boton = gr.Button("Modify image")
                                    result = gr.Image(type="pil", label="Output", interactive=False)

                                boton.click(
                                    fn=modify_image_interface,
                                    inputs=[img, atts],
                                    outputs=result
                                )

                        with gr.Tab("Information"):
                            gr.Markdown("<h2>Overview</h2>\
                                        <hr>\
                                        Interface GAN is a sophisticated deep learning model created in 2020 that utilizes the capabilities \
                                        of GANs to enable detailed and high-fidelity image editing. The model is designed to \
                                        manipulate latent codes, which are the underlying representations of images in the GAN's \
                                        latent space. By adjusting these codes, users can make precise edits to facial features, \
                                        expressions, and other attributes in a highly intuitive manner.<br><br>\
                                        <h2>Instructions</h2> \
                                        <hr> \
                                            &nbsp;&nbsp;&nbsp;&nbsp;1. Select 'InterFaceGAN'. <br>\
                                            &nbsp;&nbsp;&nbsp;&nbsp;2. Select 'Usage'. <br>\
                                            &nbsp;&nbsp;&nbsp;&nbsp;3. Upload an image. There are three options: upload it directly from your \
                                            device, take a photo or paste an image. <br>\
                                            &nbsp;&nbsp;&nbsp;&nbsp;4. Choose the attributes to modify by moving the sliders, by moving them to the \
                                            right you will increase the selected attribute and by moving the slider to the left you will decrease it. <br>\
                                            &nbsp;&nbsp;&nbsp;&nbsp;5. Click 'Modify Image'.<br><br>\
                                        <h2>More Information</h2> \
                                        <hr> \
                                        ")
                                        
                            gr.HTML('<a href="https://arxiv.org/pdf/2005.09635" target="_blank">Read the Paper</a>')
                            gr.HTML('<a href="https://github.com/genforce/interfacegan" target="_blank">GitHub Repository</a>')

                with gr.Tab("Multi Attribute Latent Transformer"):
                    with gr.Tabs():
                        with gr.Tab("Usage"):
                            with gr.Row():
                                with gr.Column(scale=1):
                                    atts = [
                                        gr.Slider(minimum=-2, maximum=2, label="Bald", step=1, value=0),
                                        gr.Slider(minimum=-2, maximum=2, label="Male", step=1, value=0),
                                        gr.Slider(minimum=-2, maximum=2, label="Smiling", step=1, value=0),
                                        gr.Slider(minimum=-2, maximum=2, label="Big Lips", step=1, value=0),
                                        gr.Slider(minimum=-2, maximum=2, label="Young", step=1, value=0),
                                        gr.Slider(minimum=-2, maximum=2, label="No Beard", step=1, value=0),
                                        gr.Slider(minimum=-2, maximum=2, label="Big Nose", step=1, value=0),
                                        gr.Slider(minimum=-2, maximum=2, label="Pale Skin", step=1, value=0),
                                        gr.Slider(minimum=-2, maximum=2, label="Narrow Eyes", step=1, value=0),
                                        gr.Slider(minimum=-2, maximum=2, label="Bangs", step=1, value=0),
                                        gr.Slider(minimum=-2, maximum=2, label="Attractive", step=1, value=0),
                                        gr.Slider(minimum=-2, maximum=2, label="Lipstick", step=1, value=0),
                                        gr.Slider(minimum=-2, maximum=2, label="Eyeglasses", step=1, value=0),
                                        gr.Slider(minimum=-2, maximum=2, label="High Cheekbones", step=1, value=0),
                                        gr.Slider(minimum=-2, maximum=2, label="Gray Hair", step=1, value=0),
                                        gr.Slider(minimum=-2, maximum=2, label="Blond Hair", step=1, value=0),
                                        gr.Slider(minimum=-2, maximum=2, label="Bushy Eyebrows", step=1, value=0),
                                        gr.Slider(minimum=-2, maximum=2, label="Chubby", step=1, value=0),
                                        gr.Slider(minimum=-2, maximum=2, label="Wavy Hair", step=1, value=0),
                                        gr.Slider(minimum=-2, maximum=2, label="Arched Eyebrows", step=1, value=0),
                                    ]

                                with gr.Column(scale=1):
                                    img = gr.Image(type="pil", label="Input")
                                    boton = gr.Button("Modify Image")
                                    result = gr.Image(type="pil", label="Output", interactive=False)
                                
                                boton.click(
                                    fn=modify_image_tedi,
                                    inputs=[img, atts],
                                    outputs=result
                                )

                        with gr.Tab("Information"):
                            gr.Markdown("<h2>Overview</h2>\
                                        <hr>\
                                        One of the latest contributions to facial attribute manipulation is an unified transformer \
                                        capable of learning multi-attribute transformations. This method, developed by Adria \
                                        Carrasquilla in 2023, builds on the Single Latent Transformer method and overcomes \
                                        its sequential nature by forcing the original image to go through different models, each one \
                                        transforming one attribute at a time.\
                                        <br><br>\
                                        <h2>Instructions</h2> \
                                        <hr> \
                                            &nbsp;&nbsp;&nbsp;&nbsp;1. Select 'Multi Attribute Latent Transformer'. <br>\
                                            &nbsp;&nbsp;&nbsp;&nbsp;2. Select 'Usage'. <br>\
                                            &nbsp;&nbsp;&nbsp;&nbsp;3. Upload an image. There are three options: upload it directly from your \
                                            device, take a photo or paste an image. <br>\
                                            &nbsp;&nbsp;&nbsp;&nbsp;4. Choose the attributes to modify by moving the sliders, by moving them to the \
                                            right you will increase the selected attribute and by moving the slider to the left you will decrease it. <br>\
                                            &nbsp;&nbsp;&nbsp;&nbsp;5. Click 'Modify Image'.<br><br>\
                                        <h2>More Information</h2> \
                                        <hr> \
                                        ")
                                        
                            gr.HTML('<a href="https://github.com/genforce/interfacegan" target="_blank">GitHub Repository</a>')

                with gr.Tab("TediGAN"):
                    with gr.Tabs():
                        with gr.Tab("Usage"):
                            with gr.Row():
                                img = gr.Image(type="pil", label="Input")
                            with gr.Row():
                                description = gr.Textbox(label="Description", placeholder="Write description for new image")
                            with gr.Row():
                                boton = gr.Button("Modify Image")
                            with gr.Row():
                                result = gr.Image(type="pil", label="Output", interactive=False)

                        boton.click(
                            fn=modify_image_tedi,
                            inputs=[img, description],
                            outputs=result
                        )
                    
                        with gr.Tab("Information"):
                            gr.Markdown("<h2>Overview</h2>\
                                        <hr>\
                                        TediGAN is a 2020 model that seamlessly combines the capabilities of Generative \
                                        Adversarial Networks (GANs) with textual descriptions to enable detailed and flexible image \
                                        editing. By integrating GANs with natural language processing, TediGAN identifies the relevant \
                                        features in the latent space of a pre-trained GAN based on user-provided text. It then manipulates \
                                        these features to apply changes such as altering facial expressions, adding accessories, or changing \
                                        hairstyles, all guided by the specific textual instructions. <br><br>\
                                        <h2>Instructions</h2> \
                                        <hr> \
                                            &nbsp;&nbsp;&nbsp;&nbsp;1. Select 'TediGAN'. <br>\
                                            &nbsp;&nbsp;&nbsp;&nbsp;2. Select 'Usage'. <br>\
                                            &nbsp;&nbsp;&nbsp;&nbsp;3. Upload an image. There are three options: upload it directly from your \
                                            device, take a photo or paste an image. <br>\
                                            &nbsp;&nbsp;&nbsp;&nbsp;4. Enter a detailed description that specifies the changes you want to make on the face.<br>\
                                            &nbsp;&nbsp;&nbsp;&nbsp;5. Click 'Modify Image'.<br><br>\
                                        <h2>More Information</h2> \
                                        <hr> \
                                        ")
                                        
                            gr.HTML('<a href="https://arxiv.org/pdf/2012.03308" target="_blank">Read the Paper</a>')
                            gr.HTML('<a href="https://github.com/IIGROUP/TediGAN" target="_blank">GitHub Repository</a>')

                with gr.Tab("StyleCLIP"):
                    with gr.Tabs():
                        with gr.Tab("Usage"):
                            with gr.Row():
                                img = gr.Image(type="pil", label="Input")
                            with gr.Row():
                                description = gr.Textbox(label="Description", placeholder="Write description for new image")
                            with gr.Row():
                                boton = gr.Button("Modify Image")
                            with gr.Row():
                                result = gr.Image(type="pil", label="Output")
                                
                            boton.click(
                                fn=modify_image_styleclip,
                                inputs=[img, description],
                                outputs=result
                            )

                        with gr.Tab("Information"):
                            gr.Markdown("<h2>Overview</h2>\
                                        <hr>\
                                        StyleCLIP is an advanced image editing model created in 2021 that leverages the robust language-image \
                                        understanding of CLIP (Contrastive Language-Image Pre-Training) alongside the powerful image \
                                        generation capabilities of StyleGAN. This combination allows users to make precise modifications \
                                        to images through natural language descriptions. StyleCLIP works by mapping the text descriptions \
                                        to the corresponding latent space directions in StyleGAN, adjusting the latent codes based on the\
                                        provided text, and generating edited images that accurately reflect the specified changes, such as \
                                        altering styles, colors, or specific attributes like 'change the expression to happy'.<br><br>\
                                        <h2>Instructions</h2> \
                                        <hr> \
                                            &nbsp;&nbsp;&nbsp;&nbsp;1. Select 'StyleCLIP'. <br>\
                                            &nbsp;&nbsp;&nbsp;&nbsp;2. Select 'Usage'. <br>\
                                            &nbsp;&nbsp;&nbsp;&nbsp;3. Upload an image. There are three options: upload it directly from your \
                                            device, take a photo or paste an image. <br>\
                                            &nbsp;&nbsp;&nbsp;&nbsp;4. Enter a detailed description that specifies the changes you want to make on the face. <br>\
                                            &nbsp;&nbsp;&nbsp;&nbsp;5. Click 'Modify Image'.<br><br>\
                                        <h2>More Information</h2> \
                                        <hr> \
                                        ")
                                        
                            gr.HTML('<a href="https://arxiv.org/pdf/2103.17249" target="_blank">Read the Paper</a>')
                            gr.HTML('<a href="https://github.com/orpatashnik/StyleCLIP" target="_blank">GitHub Repository</a>')
                    
                with gr.Tab("Comparison"):
                    with gr.Row():
                        img = gr.Image(type="pil", label="Input")
                    with gr.Row():
                        with gr.Column():
                            atts1 = [
                                gr.Slider(minimum=-2, maximum=2, label="Bald", step=1, value=0),
                                gr.Slider(minimum=-2, maximum=2, label="Male", step=1, value=0),
                                gr.Slider(minimum=-2, maximum=2, label="Smiling", step=1, value=0),
                                gr.Slider(minimum=-2, maximum=2, label="Big Lips", step=1, value=0),
                                gr.Slider(minimum=-2, maximum=2, label="Young", step=1, value=0),
                                gr.Slider(minimum=-2, maximum=2, label="No Beard", step=1, value=0),
                                gr.Slider(minimum=-2, maximum=2, label="Big Nose", step=1, value=0),
                                gr.Slider(minimum=-2, maximum=2, label="Pale Skin", step=1, value=0),
                                gr.Slider(minimum=-2, maximum=2, label="Narrow Eyes", step=1, value=0),
                                gr.Slider(minimum=-2, maximum=2, label="Bangs", step=1, value=0)]
                        with gr.Column():
                            atts2 = [
                                gr.Slider(minimum=-2, maximum=2, label="Attractive", step=1, value=0),
                                gr.Slider(minimum=-2, maximum=2, label="Lipstick", step=1, value=0),
                                gr.Slider(minimum=-2, maximum=2, label="Eyeglasses", step=1, value=0),
                                gr.Slider(minimum=-2, maximum=2, label="High Cheekbones", step=1, value=0),
                                gr.Slider(minimum=-2, maximum=2, label="Gray Hair", step=1, value=0),
                                gr.Slider(minimum=-2, maximum=2, label="Blond Hair", step=1, value=0),
                                gr.Slider(minimum=-2, maximum=2, label="Bushy Eyebrows", step=1, value=0),
                                gr.Slider(minimum=-2, maximum=2, label="Chubby", step=1, value=0),
                                gr.Slider(minimum=-2, maximum=2, label="Wavy Hair", step=1, value=0),
                                gr.Slider(minimum=-2, maximum=2, label="Arched Eyebrows", step=1, value=0)
                            ]
                    with gr.Row():
                        boton = gr.Button("Modify Image")
                    with gr.Row():
                        result_interfaceGAN = gr.Image(type="pil", label="InterFaceGAN", interactive=False)
                        result_multiTransformer = gr.Image(type="pil", label="MultiTransformer", interactive=False)
                        result_tediGAN = gr.Image(type="pil", label="TediGAN", interactive=False)
                        result_styleCLIP = gr.Image(type="pil", label="StyleCLIP", interactive=False)   
                        
                    boton.click(
                        fn=modify_images_comparison,
                        inputs=[img, atts1, atts2],
                        outputs=[result_interfaceGAN, result_multiTransformer, result_tediGAN, result_styleCLIP]
                    )

        return demo

if __name__ == "__main__":
    main_interface().launch()
