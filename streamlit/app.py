import cv2
import numpy as np
import streamlit as st
from streamlit_drawable_canvas import st_canvas

# from src.predict import predict
import torch
from torchvision import transforms
from src.net import Net


MODEL_INPUT_SIZE = 28
CANVAS_SIZE = MODEL_INPUT_SIZE * 8
CHECKPOINT_PATH = "mnist_cnn.pt"


@st.cache
def load_model(ckpt_path, device):
    net = Net().to(device)
    net.eval()
    net.load_state_dict(torch.load(ckpt_path, map_location=torch.device(device)))
    return net


def predict(img_numpy, net, device="cpu"):
    transform = transforms.Compose([transforms.ToTensor()])
    img_torch = transform(img_numpy.astype(np.float32))
    img_torch = img_torch.unsqueeze(dim=0).to(device)

    log_preds = net(img_torch)
    return log_preds


def main():
    st.write("Draw any number (0-9) here")
    canvas_res = st_canvas(
        stroke_width=20,
        stroke_color="white",
        background_color="black",
        width=CANVAS_SIZE,
        height=CANVAS_SIZE,
        drawing_mode="freedraw",
        key="canvas",
        display_toolbar=True,
    )

    net = load_model(CHECKPOINT_PATH, "cpu")

    if canvas_res.image_data is not None:
        # Scale down image to the model input size
        img = cv2.resize(
            canvas_res.image_data.astype("uint8"), (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE)
        )
        # Rescaled image upwards to show
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # img_rescaled = cv2.resize(img, (CANVAS_SIZE, CANVAS_SIZE), interpolation=cv2.INTER_NEAREST)

        # st.write("Downscaled model input:")
        # st.image(img_rescaled)

        pred = predict(np.array(img), net).detach().numpy()
        st.write(
            f"The predicted digit is {pred.argmax()}. The model is {np.exp(pred.max()) * 100:.1f}% sure."
        )


if __name__ == "__main__":
    main()
