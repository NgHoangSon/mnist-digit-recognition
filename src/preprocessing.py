import torch
from PIL import Image, ImageOps
import numpy as np
import torchvision.transforms as transforms

def preprocess_image(image):
    image=image.convert("L")

    arr=np.array(image)

    mask=arr<100
    ys,xs=np.where(mask)

    if len(xs)==0:
        raise ValueError("không tìm thấy nét chữ số")

    cropped=image.crop((
        xs.min(),
        ys.min(),
        xs.max()+1,
        ys.max()+1
    ))

    w,h=cropped.size
    side=max(w,h)+20
    canvas=Image.new("L",(side,side),255)
    canvas.paste(
        cropped,
        ((side-w)//2,(side-h)//2)
    )

    resized=canvas.resize(
        (28,28),
        Image.Resampling.LANCZOS
    )

    inverted=ImageOps.invert(resized)

    transform=transforms.ToTensor()
    x=transform(inverted)

    x=x.unsqueeze(0)

    return x