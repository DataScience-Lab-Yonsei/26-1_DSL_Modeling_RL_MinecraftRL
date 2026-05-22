from pathlib import Path
import numpy as np
from PIL import Image
from train_craftground_raycast_curriculum import annotate_frame

src = Path('/home/hj/dsl/modeling/Hyunjin/artifacts/raycast_hunt/s84_169_20260325_continue/r02_hitfocus_recovery/gifs/r02_hitfocus_recovery_epoch_006_ep_01.gif')
out = src.with_name(src.stem + '_overlay_preview.gif')

im = Image.open(src)
frames = []
for i in range(getattr(im, 'n_frames', 1)):
    im.seek(i)
    arr = np.array(im.convert('RGB'))
    annotated = annotate_frame(
        arr,
        [
            'r02 e006 ep1',
            'hp=18.0 shots=77 hits=10 kills=1',
            'dealt=33.0 taken=11.0 step=560',
        ],
    )
    frames.append(Image.fromarray(annotated))

frames[0].save(out, save_all=True, append_images=frames[1:], duration=100, loop=0)
print(out)
