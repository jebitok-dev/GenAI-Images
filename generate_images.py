from diffusers import StableDiffusionPipeline
import torch

# Load the pre-trained model
model_id = 'CompVis/stable-diffusion-v1-4'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

pipe = StableDiffusionPipeline.from_pretrained(model_id)
pipe.to(device)

# Generate images

prompts = [
    "A diverse group of young millennials and Gen Zs in graduation caps and gowns, standing in front of a modest suburban home. Their middle-class or low-income parents are beside them, smiling proudly. The scene is filled with joy and hope, with a sunny day and a clear blue sky.",
    "The same group of graduates, now dressed in business casual attire, sitting in front of computers with worried expressions. Job rejection emails fill the screens, and newspapers with headlines about high unemployment rates are scattered on a desk. The room is dimly lit, reflecting their anxiety and uncertainty.",
    "The graduates, now with renewed determination, are gathered around a computer in a cozy, well-lit room. They are exploring online resources about software development, digital marketing, machine learning, and UI/UX design. A banner in the background reads 'Tech BootCamp'. The mood is one of discovery and hope.",
    "The group, now confidently working in a modern, open-plan office environment. They are collaborating on projects, smiling, and presenting their work. Certificates of completion from BootCamps and tech courses are displayed on the walls. The scene is vibrant and professional, reflecting their success.",
    "The successful tech professionals are back in their community, hosting a workshop for younger students in a bright, welcoming classroom. They are teaching and mentoring, with laptops open, showing the possibilities of a career in tech. The young students look inspired and eager to learn. The atmosphere is encouraging and positive."
]

for i, prompt in enumerate(prompts): 
    image = pipe(prompt).images[0]
    image.save(f'image_{i + 1}.png')
    print(f'Image image_{i + 1}.png generated successfully!')