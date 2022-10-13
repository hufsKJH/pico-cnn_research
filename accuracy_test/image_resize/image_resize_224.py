import os
import glob
from PIL import Image

from tqdm import tqdm

files = glob.glob('./test_images/*')

for f in tqdm(files):
#     print(files)
    title, ext = os.path.splitext(f)
    if ext in ['.jpg', '.png', '.jpeg']:
        img = Image.open(f)
        img_resize = img.resize((224, 224))
        img_resize.save(f)