"""Draw the app's existing screen mark at Windows icon resolutions."""
from pathlib import Path
from PIL import Image, ImageDraw

image = Image.new('RGBA', (256, 256))
d = ImageDraw.Draw(image)
d.rounded_rectangle((0, 0, 255, 255), radius=64, fill='#174b39')
d.rounded_rectangle((48, 60, 208, 176), radius=24, outline='#f5faef', width=12)
d.line((128, 176, 128, 212), fill='#f5faef', width=12)
d.line((96, 212, 160, 212), fill='#f5faef', width=12)
d.line((88, 116, 116, 88), fill='#f5faef', width=12)
d.line((140, 148, 168, 120), fill='#f5faef', width=12)
image.save(Path(__file__).with_name('isharescreen.ico'), sizes=[(16,16),(24,24),(32,32),(48,48),(64,64),(128,128),(256,256)])
