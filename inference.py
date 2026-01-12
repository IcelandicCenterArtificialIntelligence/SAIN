import argparse
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os
import sys
from collections import OrderedDict

sys.path.append(os.getcwd())

try:
    from model.SAIN import SAIN
except ImportError:
    print("Error importando SAIN. Asegúrate de estar en la raíz del repositorio.")
    sys.exit(1)

# --- CLASE DE ARGUMENTOS ---
class MockArgs:
    def __init__(self):
        self.cuda = True
        self.num_gpu = 1
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.resume_flownet = ''
        self.resume = False
        self.phase = 'test'
        self.dataset = 'std12k'
        self.joinType = 'concat'
        self.model = 'SAIN'
        self.nbr_frame = 2
        self.crop_size = 192
        self.batch_size = 1
        self.test_batch_size = 1
        self.num_workers = 1
        self.loss = '0.7*L1+0.3*LPIPS'
        self.lr = 0.0002
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.start_epoch = 0
        self.max_epoch = 50
        self.pretrained = None
        self.checkpoint_dir = 'ckp'
        self.result_dir = 'result'
        self.log_iter = 100

def load_image(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"No se encuentra la imagen: {path}")
    img = Image.open(path).convert('RGB')
    return img

def save_image(tensor, path):
    if isinstance(tensor, (list, tuple)):
        tensor = tensor[0]
    img = tensor.squeeze(0).cpu().clamp(0, 1)
    tf = transforms.ToPILImage()
    img_pil = tf(img)
    img_pil.save(path)
    print(f"✅ Interpolación guardada en: {path}")

def run_inference(args):
    print(f"🔧 Configurando modelo SAIN en {args.max_size}px max...")

    # 1. Configuración del Modelo
    model_args = MockArgs()
    try:
        model = SAIN(model_args)
    except Exception as e:
        print(f"❌ Error inicializando SAIN: {e}")
        return

    # 2. Cargar Checkpoint
    if not os.path.exists(args.checkpoint):
        print(f"❌ Error: No existe el checkpoint: {args.checkpoint}")
        return

    checkpoint = torch.load(args.checkpoint, map_location=model_args.device)
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint

    # Limpiar prefijos 'module.'
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_state_dict[k.replace("module.", "")] = v

    # Cargar pesos
    try:
        model.load_state_dict(new_state_dict, strict=True)
    except RuntimeError:
        print(f"⚠️  Carga estricta falló. Usando carga flexible...")
        model.load_state_dict(new_state_dict, strict=False)

    model.to(model_args.device)
    model.eval()

    # 3. Procesamiento de Imágenes (Downscale + Padding)
    print(f"🖼️  Procesando entrada...")
    img1_pil = load_image(args.img1)
    img2_pil = load_image(args.img2)

    orig_w, orig_h = img1_pil.size

    # Calcular factor de escala (Downscaling) para evitar OOM
    scale_factor = 1.0
    if max(orig_w, orig_h) > args.max_size:
        scale_factor = args.max_size / max(orig_w, orig_h)
        new_w_scaled = int(orig_w * scale_factor)
        new_h_scaled = int(orig_h * scale_factor)
        print(f"📉 Downscaling: {orig_w}x{orig_h} -> {new_w_scaled}x{new_h_scaled} (Factor: {scale_factor:.2f})")

        img1_pil = img1_pil.resize((new_w_scaled, new_h_scaled), Image.BICUBIC)
        img2_pil = img2_pil.resize((new_w_scaled, new_h_scaled), Image.BICUBIC)
    else:
        new_w_scaled, new_h_scaled = orig_w, orig_h

    # Convertir a Tensor
    tf = transforms.ToTensor()
    I0 = tf(img1_pil).unsqueeze(0).to(model_args.device)
    I1 = tf(img2_pil).unsqueeze(0).to(model_args.device)

    # Calcular Padding necesario (múltiplos de 64)
    # Esto evita deformar la imagen (aspect ratio se mantiene, se añade borde negro)
    align = 64
    pad_w = (align - (new_w_scaled % align)) % align
    pad_h = (align - (new_h_scaled % align)) % align

    # Aplicar Padding (Left, Right, Top, Bottom)
    if pad_w > 0 or pad_h > 0:
        print(f"⬜ Añadiendo padding: W+{pad_w}, H+{pad_h}")
        I0 = F.pad(I0, (0, pad_w, 0, pad_h), mode='replicate')
        I1 = F.pad(I1, (0, pad_w, 0, pad_h), mode='replicate')

    # 4. Inputs Dummy
    B, _, H_pad, W_pad = I0.shape
    dummy_points = torch.zeros(B, 1, H_pad, W_pad).to(model_args.device)
    flow_tensor = torch.zeros(B, 2, H_pad, W_pad).to(model_args.device)
    dummy_region_flow = [flow_tensor, flow_tensor]

    # 5. Inferencia
    print("🚀 Generando frame...")
    with torch.no_grad():
        output = model(I0, I1, dummy_points, dummy_region_flow)

        # Gestión de lista/tupla de salida
        if isinstance(output, (list, tuple)):
            output = output[0]

    # 6. Post-procesado (Crop + Upscale)
    print("✨ Restaurando resolución original...")

    # a) Quitar el padding
    # Si pad_w es 0, el slice es hasta el final, si no, restamos el pad
    crop_w = W_pad - pad_w
    crop_h = H_pad - pad_h
    output = output[:, :, :crop_h, :crop_w]

    # b) Upscaling a resolución original exacta
    # Usamos interpolate bilinear/bicubic para suavizar
    output = F.interpolate(output, size=(orig_h, orig_w), mode='bicubic', align_corners=False)

    save_image(output, args.output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--img1', type=str, required=True)
    parser.add_argument('--img2', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, default='ckp/checkpoints/model_best.pth')
    parser.add_argument('--output', type=str, default='result_test.png')
    # Nuevo argumento para controlar la VRAM
    parser.add_argument('--max_size', type=int, default=960, help='Resolución máxima del lado más largo para evitar OOM (default: 960)')

    args, unknown = parser.parse_known_args()
    run_inference(args)
