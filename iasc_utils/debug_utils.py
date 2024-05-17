from PIL import Image, ImageDraw
import torch

def save_image(image, bbox, bbox2, name="test.png"):
    bbox = bbox.unsqueeze(0).tolist()[0][0]
    bbox = [int(i) for i in bbox]
    bbox = [(bbox[0], bbox[1]), (bbox[2], bbox[1]), (bbox[2], bbox[3]), (bbox[0], bbox[3])]

    bbox2 = bbox2.unsqueeze(0).tolist()[0][0]
    bbox2 = [int(i) for i in bbox2]
    bbox2 = [(bbox2[0], bbox2[1]), (bbox2[2], bbox2[1]), (bbox2[2], bbox2[3]), (bbox2[0], bbox2[3])]


    image = ((image+1)*0.5).mul(255).add_(0.5).clamp_(0, 255).to(torch.uint8)
    image = image.squeeze(0).permute(1,2,0).cpu().numpy()
    image = Image.fromarray(image)
    draw = ImageDraw.Draw(image)

    draw.polygon(bbox, outline='red', width=5)
    draw.polygon(bbox2, outline='blue', width=5)

    image.save(name)