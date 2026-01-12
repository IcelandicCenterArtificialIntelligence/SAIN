import argparse
import torch
from torchvision import transforms
from PIL import Image
import os
import sys
from collections import OrderedDict

# Añadimos el directorio actual al path
sys.path.append(os.getcwd())

try:
    from model.SAIN import SAIN
except ImportError:
    print("Error importando SAIN. Asegúrate de estar en la raíz del repositorio.")
    sys.exit(1)

# --- CLASE PARA IMITAR LOS ARGUMENTOS ---
class MockArgs:
    def __init__(self):
        # Hardware
        self.cuda = True
        self.num_gpu = 1
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Parámetros vacíos para evitar errores de carga
        self.resume_flownet = ''
        self.resume = False

        # Configuración
        self.phase = 'test'
        self.dataset = 'std12k'
        self.joinType = 'concat'
        self.model = 'SAIN'

        # Estructura
        self.nbr_frame = 2
        self.crop_size = 192
        self.batch_size = 1
        self.test_batch_size = 1
        self.num_workers = 1

        # Dummies
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

def preprocess(img, device):
    tf = transforms.Compose([
        transforms.ToTensor(),
    ])
    return tf(img).unsqueeze(0).to(device)

def save_image(tensor, path):
    if isinstance(tensor, (list, tuple)):
        tensor = tensor[0]

    img = tensor.squeeze(0).cpu()
    img = torch.clamp(img, 0, 1)
    tf = transforms.ToPILImage()
    img_pil = tf(img)
    img_pil.save(path)
    print(f"✅ Interpolación guardada exitosamente en: {path}")

def run_inference(args):
    print(f"🔧 Configurando modelo SAIN...")

    # 1. Inicializar
    model_args = MockArgs()
    print(f"   Device seleccionado: {model_args.device}")

    try:
        model = SAIN(model_args)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"❌ Error inicializando SAIN: {e}")
        return

    # 2. Cargar Checkpoint
    print(f"📂 Cargando pesos desde {args.checkpoint}...")
    if not os.path.exists(args.checkpoint):
        print(f"❌ Error: No existe el archivo de pesos: {args.checkpoint}")
        return

    checkpoint = torch.load(args.checkpoint, map_location=model_args.device)

    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # Limpiar prefijos 'module.' (DataParallel)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace("module.", "")
        new_state_dict[name] = v

    try:
        model.load_state_dict(new_state_dict, strict=True)
        print("✅ Pesos cargados correctamente (Strict mode)")
    except RuntimeError:
        print(f"⚠️  Carga estricta falló. Intentando carga flexible...")
        model.load_state_dict(new_state_dict, strict=False)

    model.to(model_args.device)
    model.eval()

    # 3. Imágenes
    print(f"🖼️  Leyendo imágenes...")
    img1 = load_image(args.img1)
    img2 = load_image(args.img2)

    w, h = img1.size
    align_factor = 64
    new_w = (w // align_factor) * align_factor
    new_h = (h // align_factor) * align_factor

    if new_w != w or new_h != h:
        print(f"⚠️  Ajustando tamaño: {w}x{h} -> {new_w}x{new_h}")
        img1 = img1.resize((new_w, new_h), Image.BICUBIC)
        img2 = img2.resize((new_w, new_h), Image.BICUBIC)

    I0 = preprocess(img1, model_args.device)
    I1 = preprocess(img2, model_args.device)

    # 4. Generar Inputs Dummy (CORREGIDO FINAL)
    print("🧩 Generando guías dummy (Trazos y Flujo Bi-direccional)...")
    B, C, H, W = I0.shape

    # 1. Points: Debe ser de 1 canal (Grayscale/Mask)
    dummy_points = torch.zeros(B, 1, H, W).to(model_args.device)

    # 2. Region Flow: Debe ser una LISTA de 2 tensores (Forward y Backward)
    # Cada flujo tiene 2 canales (dx, dy)
    flow_tensor = torch.zeros(B, 2, H, W).to(model_args.device)
    dummy_region_flow = [flow_tensor, flow_tensor]  # <--- EL ARREGLO: Lista de 2 elementos

    # 5. Inferencia
    print("🚀 Generando frame intermedio...")
    with torch.no_grad():
        # Pasamos los 4 argumentos
        output = model(I0, I1, dummy_points, dummy_region_flow)

    # 6. Guardar
    save_image(output, args.output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--img1', type=str, required=True)
    parser.add_argument('--img2', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, default='ckp/checkpoints/model_best.pth')
    parser.add_argument('--output', type=str, default='result_test.png')

    args, unknown = parser.parse_known_args()
    run_inference(args)
