import gradio as gr
from PIL import Image
from style_transfer import load_image, run_style_transfer, save_output
import time

def transfer_style(content_img, style_img, style_strength):
    content_tensor = load_image(content_img)
    style_tensor = load_image(style_img)

    style_weight = style_strength * 5000 
    content_weight = 1

    output = run_style_transfer(content_tensor, style_tensor, steps=6000, style_weight=style_weight, content_weight=content_weight, learning_rate=0.003)
    output_path = save_output(output, path="output.jpg")
    
    return Image.open(output_path), output_path

# ─────────────────────────────── UI Setup ───────────────────────────────

with gr.Blocks(theme=gr.themes.Soft(), css="""
  .gr-button { background: linear-gradient(to right, #4e54c8, #8f94fb); color: white; font-weight: bold; }
  .gr-slider input { color: #fff !important; }
  .gr-markdown h1 { font-size: 2rem; font-weight: bold; }
  body { background-color: #121212; color: #fff; }
""") as demo:

    gr.Markdown("# 🎨 AI Style Transfer App")
    gr.Markdown("Upload your **content** and **style** images on the left. Adjust the **style strength** on the slider, then click **Stylize** to blend them!")

    with gr.Row():
        with gr.Column(scale=1):
            content_input = gr.Image(label="🖼️ Content Image")
            style_input = gr.Image(label="🎨 Style Image")
            strength_slider = gr.Slider(minimum=1, maximum=1000, step=10, value=100, label="Style Weight")
            stylize_button = gr.Button("Stylize")

        with gr.Column(scale=1):
            styled_output = gr.Image(label="✨ Stylized Output")
            download_button = gr.File(label="⬇️ Download Image")

    stylize_button.click(
        transfer_style,
        inputs=[content_input, style_input, strength_slider],
        outputs=[styled_output, download_button]
    )

demo.launch()
