import os
import numpy as np
import imageio.v2 as imageio
import OpenEXR
import Imath

obj_name = "barrel_stove"

root = "/home/elghobashy/relight/baseline/nvdiffrec/data/bareel_stove/textures"
def read_exr_channel(path, channel='R'):
    exr = OpenEXR.InputFile(path)
    print(f"📄 Channels in {path}:", exr.header()['channels'].keys())
    dw = exr.header()['dataWindow']
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1
    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    data = exr.channel(channel, pt)
    return np.frombuffer(data, dtype=np.float32).reshape((height, width))

def convert_kd(path_in, path_out):
    kd = imageio.imread(path_in)
    imageio.imwrite(path_out, kd)
    print("✅ Kd saved to", path_out)

def convert_ks(kd_path, metal_path, path_out):
    kd = imageio.imread(kd_path).astype(np.float32) / 255.0
    
    metal = read_exr_channel(metal_path, 'Y')[..., None]  # H x W x 1
    ks = (1.0 - metal) * 0.04 + metal * kd
    ks = np.clip(ks * 255.0, 0, 255).astype(np.uint8)
    imageio.imwrite(path_out, ks)
    print("✅ Ks saved to", path_out)

def convert_normal(path_in, path_out):
    r = read_exr_channel(path_in, 'R')
    g = read_exr_channel(path_in, 'G')
    b = read_exr_channel(path_in, 'B')
    normal = np.stack([r, g, b], axis=-1)
    # Convert from [-1,1] to [0,1]
    normal = np.clip(normal * 0.5 + 0.5, 0, 1)
    normal = (normal * 255).astype(np.uint8)
    imageio.imwrite(path_out, normal)
    print("✅ Normal saved to", path_out)

# Run conversions
convert_kd(
    os.path.join(root, f"{obj_name}_diff_4k.jpg"),
    os.path.join(root,"kd.png"),
)
convert_ks(
    os.path.join(root, f"{obj_name}_diff_4k.jpg"),
    os.path.join(root, f"{obj_name}_metal_4k.exr"),
    os.path.join(root, "ks.png")
)

convert_normal(
    os.path.join(root, f"{obj_name}_nor_gl_4k.exr"),
    os.path.join(root, "norm.png")
)
