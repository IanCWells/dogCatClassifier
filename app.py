import gradio as gr
from fastai.vision.all import *

def is_cat(x): return x[0].isUpper()

# Load your trained model
learn = load_learner('model.pkl')

categories = ['Dog', 'Cat']  # Use a list instead of a set
def classify_image(img):
    pred, idx, probs = learn.predict(img)
    return dict(zip(categories, map(float, probs)))

# Updated Gradio components
image = gr.Image(height = (192), width = (192))  # Use gr.Image for image input
label = gr.Label()  # Instantiate gr.Label correctly
examples = ['dog.jpg', 'cat.jpg', 'dunno.jpg']

# Create the interface
intf = gr.Interface(fn=classify_image, inputs=image, outputs=label, examples=examples)

intf.launch(inline=False)