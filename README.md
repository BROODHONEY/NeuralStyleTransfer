#Neural Style Transfer App

Welcome to my **Neural Style Transfer App** — a Gradio-powered tool where you can blend the **content of one image** with the **artistic style of another** in seconds, right from your browser.

Try it live on Hugging Face 👉 [Click here](https://huggingface.co/spaces/BroodHoney/NeuralStyleTransfer)
- Upload your **content** and **style** images
- Adjust **style influence** with a slider
- Download the stylized image

---

## What Does It Do?

It uses a deep learning model based on **VGG19** to generate an output image that looks like:

### Content Image:
![TAL-florence-ITCITIES0924-6aada9377bca4896a4338ab8950cca39](https://github.com/user-attachments/assets/1e61875f-ae0b-489e-af15-91065b80e238)
### Style Image:
![istockphoto-543465140-612x612](https://github.com/user-attachments/assets/dca2761f-3bd2-46d3-a5d8-e3d0c3acfaf1)
### Output Image:
![output](https://github.com/user-attachments/assets/efe0709c-5993-401a-83df-74d49396d522)

---

## 🛠️ How It Works

1. Extracts features from both content and style images using VGG19.
2. Computes a content loss (how different the output is from the content).
3. Computes a style loss (difference in textures/brush strokes using Gram matrices).
4. Minimizes the total loss to generate a stylized image.

---

## Running Locally (Recommended)

If you want to run it locally:

```bash
git clone https://github.com/your-username/NeuralStyleTransfer.git
cd NeuralStyleTransfer

# Create and activate env
conda create -n style-transfer python=3.10
# Or if you wnat to run it using GPU
conda create -n style-transfer pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

conda activate style-transfer

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
```
