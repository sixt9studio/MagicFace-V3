import torch
import spaces
from diffusers import StableDiffusionPipeline, DDIMScheduler, AutoencoderKL
from transformers import AutoFeatureExtractor
from ip_adapter.ip_adapter_faceid import IPAdapterFaceID, IPAdapterFaceIDPlus
from huggingface_hub import hf_hub_download
from insightface.app import FaceAnalysis
from insightface.utils import face_align
import gradio as gr
import cv2
import os
import uuid
from datetime import datetime

# Model paths
base_model_path = "SG161222/Realistic_Vision_V4.0_noVAE"
vae_model_path = "stabilityai/sd-vae-ft-mse"
image_encoder_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
ip_ckpt = hf_hub_download(repo_id="h94/IP-Adapter-FaceID", filename="ip-adapter-faceid_sd15.bin", repo_type="model")
ip_plus_ckpt = hf_hub_download(repo_id="h94/IP-Adapter-FaceID", filename="ip-adapter-faceid-plusv2_sd15.bin", repo_type="model")

device = "cuda"

# Initialize the noise scheduler
noise_scheduler = DDIMScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    clip_sample=False,
    set_alpha_to_one=False,
    steps_offset=1,
)

# Load models
vae = AutoencoderKL.from_pretrained(vae_model_path).to(dtype=torch.float16)
pipe = StableDiffusionPipeline.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    scheduler=noise_scheduler,
    vae=vae
).to(device)

ip_model = IPAdapterFaceID(pipe, ip_ckpt, device)
ip_model_plus = IPAdapterFaceIDPlus(pipe, image_encoder_path, ip_plus_ckpt, device)

# Initialize FaceAnalysis
app = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

cv2.setNumThreads(1)

STYLE_PRESETS = [
    {
        "title": "Mona Lisa",
        "prompt": "A mesmerizing portrait in the style of Leonardo da Vinci's Mona Lisa, renaissance oil painting, soft sfumato technique, mysterious smile, Florentine background, museum quality, masterpiece",
        "preview": "🎨"
    },
    {
        "title": "Iron Hero",
        "prompt": "Hyper realistic portrait as a high-tech superhero, wearing advanced metallic suit, arc reactor glow, inside high-tech lab, dramatic lighting, cinematic composition",
        "preview": "🦾"
    },
    {
        "title": "Ancient Egyptian",
        "prompt": "Portrait as an ancient Egyptian pharaoh, wearing golden headdress and royal regalia, hieroglyphics background, dramatic desert lighting, archaeological discovery style",
        "preview": "👑"
    },
    {
        "title": "Sherlock Holmes",
        "prompt": "Victorian era detective portrait, wearing deerstalker hat and cape, holding magnifying glass, foggy London background, mysterious atmosphere, detailed illustration",
        "preview": "🔍"
    },
    {
        "title": "Star Wars Jedi",
        "prompt": "Epic portrait as a Jedi Master, wearing traditional robes, holding lightsaber, temple background, force aura effect, cinematic lighting, movie poster quality",
        "preview": "⚔️"
    },
    {
        "title": "Van Gogh Style",
        "prompt": "Self-portrait in the style of Vincent van Gogh, bold brushstrokes, vibrant colors, post-impressionist style, emotional intensity, starry background",
        "preview": "🎨"
    },
    {
        "title": "Greek God",
        "prompt": "Mythological portrait as an Olympian deity, wearing flowing robes, golden laurel wreath, Mount Olympus background, godly aura, classical Greek art style",
        "preview": "⚡"
    },
    {
        "title": "Medieval Knight",
        "prompt": "Noble knight portrait, wearing ornate plate armor, holding sword and shield, castle background, heraldic designs, medieval manuscript style",
        "preview": "🛡️"
    },
    {
        "title": "Matrix Hero",
        "prompt": "Cyberpunk portrait in digital reality, wearing black trench coat and sunglasses, green code rain effect, dystopian atmosphere, cinematic style",
        "preview": "🕶️"
    },
    {
        "title": "Pirate Captain",
        "prompt": "Swashbuckling pirate captain portrait, wearing tricorn hat and colonial coat, ship's deck background, dramatic sea storm, golden age of piracy style",
        "preview": "🏴‍☠️"
    }
]

# Updated CSS for improved readability and scrolling
css = '''
/* Allow body to scroll freely */
html, body {
    margin: 0;
    padding: 0;
    background: #f0f2f5;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    color: #333333;
    overflow-y: scroll;
}

/* Outer container can grow but allow scrolling */
#component-0 {
    width: 100%;
    box-sizing: border-box;
    padding: 20px;
}

/* Main content container with good contrast and spacing */
.container {
    background-color: #ffffff;
    color: #333333;
    border-radius: 10px;
    padding: 30px;
    margin: 0 auto 40px auto; /* Margin bottom to ensure space for scrolling */
    box-shadow: 0 8px 16px rgba(0, 0, 0, 0.15);
    max-width: 1400px;
}

/* Header styling with higher contrast text on dark background */
.header {
    text-align: center;
    margin-bottom: 2rem;
    background: #003366;
    padding: 2rem;
    border-radius: 10px;
    color: #ffffff;
}

/* Preset grid styling */
.preset-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
    gap: 1rem;
    margin: 1rem 0;
}

/* Preset cards: clear borders, high contrast text */
.preset-card {
    background: #ffffff;
    padding: 1rem;
    border-radius: 8px;
    cursor: pointer;
    transition: all 0.3s ease;
    border: 2px solid #003366;
    text-align: center;
    color: #003366;
    font-weight: bold;
}

.preset-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 6px 18px rgba(0, 0, 0, 0.2);
    background: #e6f0ff;
}

/* Larger emoji styling */
.preset-emoji {
    font-size: 2.5rem;
    margin-bottom: 0.5rem;
}

/* Input container with a lighter background for contrast */
.input-container {
    background: #e6f0ff;
    color: #003366;
    padding: 1.5rem;
    border-radius: 8px;
    margin-bottom: 1rem;
    border: 1px solid #003366;
}

/* Output gallery with a clear border and white background */
.output-gallery {
    border: 2px solid #003366;
    border-radius: 8px;
    padding: 10px;
    background: #ffffff;
}

/* Ensure any footer is hidden */
footer { display: none !important; }
'''

@spaces.GPU(enable_queue=True)
def generate_image(images, gender, prompt, progress=gr.Progress(track_tqdm=True)):
    if not prompt:
        prompt = f"Professional portrait of a {gender.lower()}"
    
    # Add specific keywords to ensure single person
    prompt = f"{prompt}, single person, solo portrait, one person only, centered composition"
    
    # Add negative prompt to prevent multiple people
    negative_prompt = (
        "multiple people, group photo, crowd, double portrait, triple portrait, "
        "many faces, multiple faces, two faces, three faces, multiple views, collage, photo grid"
    )
    
    faceid_all_embeds = []
    first_iteration = True
    preserve_face_structure = True
    face_strength = 2.1
    likeness_strength = 0.7

    for image in images:
        face = cv2.imread(image)
        faces = app.get(face)
        if not faces:
            continue
        faceid_embed = torch.from_numpy(faces[0].normed_embedding).unsqueeze(0)
        faceid_all_embeds.append(faceid_embed)

        # For the first face, keep a reference image aligned
        if first_iteration and preserve_face_structure:
            face_image = face_align.norm_crop(face, landmark=faces[0].kps, image_size=224)
            first_iteration = False

    if not faceid_all_embeds:
        return None

    # Average embedding across all provided images
    average_embedding = torch.mean(torch.stack(faceid_all_embeds, dim=0), dim=0)

    # Generate the new image using IP-Adapter FaceID Plus
    image = ip_model_plus.generate(
        prompt=prompt,
        negative_prompt=negative_prompt,
        faceid_embeds=average_embedding,
        scale=likeness_strength, 
        face_image=face_image, 
        shortcut=True, 
        s_scale=face_strength, 
        width=512, 
        height=768,  
        num_inference_steps=100,
        guidance_scale=7.5  
    )
    return image

def create_preset_click_handler(idx, prompt_input):
    def handler():
        return {"value": STYLE_PRESETS[idx]["prompt"]}
    return handler

with gr.Blocks(css=css) as demo:
    # You could add a visitor badge or other element here if desired
    # For now, we omit it to focus on the scrolling and contrast fixes


    
    with gr.Column(elem_classes="container"):
        with gr.Column(elem_classes="header"):
            gr.HTML("<h1 style='color:white;'>✨ MagicFace V3</h1>")
            gr.HTML("<h3 style='color:white;'>Transform Your Face Into Legendary Characters! https://discord.gg/openfreeai </h3>")
        
        with gr.Row():
            with gr.Column(scale=1):
                images_input = gr.Files(
                    label="📸 Upload Your Face Photos",
                    file_types=["image"],
                    elem_classes="input-container"
                )
                gender_input = gr.Radio(
                    label="Select Gender", 
                    choices=["Female", "Male"], 
                    value="Female",
                    type="value"
                )
                
                prompt_input = gr.Textbox(
                    label="🎨 Custom Prompt",
                    placeholder="Describe your desired transformation in detail...",
                    lines=3
                )
                
                with gr.Column(elem_classes="preset-container"):
                    gr.Markdown("### 🎭 Magic Transformations")
                    preset_grid = []
                    for idx, preset in enumerate(STYLE_PRESETS):
                        preset_button = gr.Button(
                            f"{preset['preview']} {preset['title']}",
                            elem_classes="preset-card"
                        )
                        preset_button.click(
                            fn=create_preset_click_handler(idx, prompt_input),
                            inputs=[],
                            outputs=[prompt_input]
                        )
                        preset_grid.append(preset_button)
                
                generate_button = gr.Button("🚀 Generate Magic", variant="primary")

            with gr.Column(scale=1):
                output_gallery = gr.Gallery(
                    label="Magic Gallery",
                    elem_classes="output-gallery",
                    columns=2
                )

        with gr.Accordion("📖 Quick Guide", open=False):
            gr.Markdown("""
                ### How to Use MagicFace V3
                1. Upload one or more face photos
                2. Select your gender
                3. Choose a magical transformation or write your own prompt
                4. Click 'Generate Magic'

                ### Pro Tips
                - Upload multiple angles of your face for better results
                - Try combining different historical or fictional characters
                - Feel free to modify the preset prompts
                - Click on generated images to view them in full size
                
                Scroll to see more content if your screen is small. Enjoy!
            """)

    generate_button.click(
        fn=generate_image,
        inputs=[images_input, gender_input, prompt_input],
        outputs=output_gallery
    )

demo.queue()
demo.launch()
