from PIL import Image

def im_concat(*imgs):
    dst = Image.new('RGB', (sum(im.width for im in imgs), imgs[0].height))
    for i, im in enumerate(imgs):
        dst.paste(im, (i * imgs[0].width, 0))
    return dst