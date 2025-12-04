import cv2
import numpy as np
import torch
import torch.nn.functional as F
import random
from google.colab.patches import cv2_imshow

# --- 1. Configurações ---
PATCH_SIZE = 20      # Tamanho do bloco
OVERLAP_SIZE = 10    # Sobreposição
ALPHA = 0.8          # 0.1 foca na costura (textura), 0.9 foca na forma (rosto)
TOLERANCE = 0.1       # Tolerância para aleatoriedade (10%)
STEP = PATCH_SIZE - OVERLAP_SIZE

# Verifica GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 Processando em: {device}")

# --- 2. Funções Auxiliares (GPU) ---

def create_patch_bank(texture_tensor, patch_size, step):
    """
    Cria um 'banco' com TODOS os patches possíveis da textura de uma vez.
    Isso permite comparar tudo simultaneamente na GPU.
    Retorna: Tensor (N_patches, patch_size, patch_size, 3)
    """
    # texture_tensor entra como (H, W, C)
    # PyTorch unfold precisa de (Batch, Channel, H, W)
    t = texture_tensor.permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)

    # Extrai patches usando unfold
    # patches terá shape (1, C * patch_h * patch_w, N_blocks)
    kc, kh, kw = 3, patch_size, patch_size
    patches = F.unfold(t, kernel_size=(kh, kw), stride=step)

    # Reorganiza para (N_blocks, patch_h, patch_w, C)
    patches = patches.permute(0, 2, 1).squeeze(0)
    patches = patches.view(-1, 3, kh, kw).permute(0, 2, 3, 1)

    return patches

def get_best_patch_gpu(patches_bank, target_patch, overlap_mask):
    """
    Calcula o erro de TODOS os patches do banco contra o alvo de uma só vez.
    """
    # 1. Erro de Correspondência (Forma do Alvo)
    # (N, P, P, 3) - (1, P, P, 3) -> Broadcasting automático
    diff_corr = patches_bank - target_patch
    err_corr = (diff_corr ** 2).sum(dim=(1, 2, 3)) # Soma erros de todos pixels

    # 2. Erro de Sobreposição (Costura)
    # A máscara define onde olhar. Se mask for 0, o erro é zerado.
    diff_overlap = diff_corr * overlap_mask
    err_overlap = (diff_overlap ** 2).sum(dim=(1, 2, 3))

    # 3. Erro Total
    total_error = ALPHA * err_corr + (1 - ALPHA) * err_overlap

    # 4. Seleção com Aleatoriedade (para evitar grid repetitivo)
    min_val = torch.min(total_error)

    # Evita erro zero absoluto
    if min_val == 0:
        threshold = 0.00001
    else:
        threshold = min_val * (1 + TOLERANCE)

    # Pega índices onde erro < threshold
    candidates_indices = torch.nonzero(total_error <= threshold).squeeze()

    # Se houver apenas um candidato ou tensor vazio, trata o caso
    if candidates_indices.dim() == 0:
        idx = candidates_indices.item()
    else:
        # Escolhe um índice aleatório da lista de tensores
        random_idx = torch.randint(0, len(candidates_indices), (1,)).item()
        idx = candidates_indices[random_idx].item()

    return patches_bank[idx]

def calculate_min_cut_cpu(diff_map):
    """
    Min-Cut (Programação Dinâmica) é sequencial e difícil de paralelizar.
    É mais rápido/fácil fazer rapidinho na CPU com NumPy para patches pequenos.
    """
    h, w = diff_map.shape
    E = diff_map.copy()

    # Acumula custos
    for r in range(1, h):
        for c in range(w):
            prev_cost = E[r-1, c]
            if c > 0: prev_cost = min(prev_cost, E[r-1, c-1])
            if c < w - 1: prev_cost = min(prev_cost, E[r-1, c+1])
            E[r, c] += prev_cost

    # Backtracking
    mask = np.zeros_like(E, dtype=bool)
    c = np.argmin(E[-1])
    for r in range(h - 1, -1, -1):
        mask[r, c:] = True # Lado direito do corte
        if r > 0:
            start = max(0, c-1)
            end = min(w, c+2)
            c = start + np.argmin(E[r-1, start:end])

    return mask

# --- 3. Função Principal ---

def quilt_transfer_gpu(texture_img, target_img):

    # 1. Prepara Tensors e move para GPU
    tex_t = torch.tensor(texture_img, dtype=torch.float32).to(device)
    tgt_t = torch.tensor(target_img, dtype=torch.float32).to(device)

    output_H, output_W, C = tgt_t.shape

    # Ajusta tamanho para múltiplo do STEP
    output_H = output_H - (output_H - PATCH_SIZE) % STEP
    output_W = output_W - (output_W - PATCH_SIZE) % STEP
    tgt_t = tgt_t[:output_H, :output_W, :]

    # Canvas de saída
    output = torch.zeros_like(tgt_t)

    # 2. Cria o Banco de Patches (O segredo da velocidade!)
    print("⏳ Criando banco de patches na VRAM...")
    patches_bank = create_patch_bank(tex_t, PATCH_SIZE, STEP) # Step menor = mais qualidade, mas mais memória
    print(f"✅ Banco criado com {len(patches_bank)} patches.")

    # 3. Iteração Principal
    rows = range(0, output_H - PATCH_SIZE + 1, STEP)
    cols = range(0, output_W - PATCH_SIZE + 1, STEP)

    total_steps = len(rows) * len(cols)
    curr_step = 0

    for r in rows:
        for c in cols:
            is_first_row = (r == 0)
            is_first_col = (c == 0)

            # Recortes atuais
            target_patch = tgt_t[r:r+PATCH_SIZE, c:c+PATCH_SIZE]

            # --- Criação da Máscara de Overlap (Na GPU) ---
            # 1.0 onde deve calcular overlap, 0.0 onde não importa
            overlap_mask = torch.zeros((PATCH_SIZE, PATCH_SIZE, 1), device=device)
            current_output_patch = output[r:r+PATCH_SIZE, c:c+PATCH_SIZE]

            if not is_first_row:
                overlap_mask[0:OVERLAP_SIZE, :, :] = 1.0
            if not is_first_col:
                overlap_mask[:, 0:OVERLAP_SIZE, :] = 1.0

            # --- Busca do Melhor Patch (Rápido!) ---
            best_patch = get_best_patch_gpu(patches_bank, target_patch, overlap_mask)

            # --- Min-Cut (Híbrido CPU/GPU) ---
            # Se for o primeiro bloco, só copia
            final_patch = best_patch.clone()

            if not is_first_row or not is_first_col:
                # Convertemos para CPU numpy apenas o necessário para o corte complexo
                best_np = best_patch.cpu().numpy()
                curr_np = current_output_patch.cpu().numpy()

                # Corte Vertical (Esquerda)
                if not is_first_col:
                    diff = ((best_np[:, :OVERLAP_SIZE] - curr_np[:, :OVERLAP_SIZE])**2).sum(axis=2)
                    mask_v = calculate_min_cut_cpu(diff) # Retorna mascara booleana
                    mask_v = np.repeat(mask_v[:, :, np.newaxis], 3, axis=2)

                    # Aplica corte na região de overlap
                    final_patch[:, :OVERLAP_SIZE] = torch.tensor(
                        curr_np[:, :OVERLAP_SIZE] * (1-mask_v) + best_np[:, :OVERLAP_SIZE] * mask_v,
                        device=device
                    )

                # Corte Horizontal (Topo)
                # Nota: Para simplificar, aplicamos sobre o resultado anterior
                # O ideal seria combinar os dois cortes, mas sequencial funciona bem visualmente
                if not is_first_row:
                    curr_top_np = output[r:r+OVERLAP_SIZE, c:c+PATCH_SIZE].cpu().numpy() # Lê do output já gravado
                    best_top_np = final_patch[:OVERLAP_SIZE, :].cpu().numpy()

                    diff = ((best_top_np - curr_top_np)**2).sum(axis=2).T # Transpor para usar mesma logica
                    mask_h = calculate_min_cut_cpu(diff).T
                    mask_h = np.repeat(mask_h[:, :, np.newaxis], 3, axis=2)

                    final_patch[:OVERLAP_SIZE, :] = torch.tensor(
                        curr_top_np * (1-mask_h) + best_top_np * mask_h,
                        device=device
                    )

            # Grava no canvas de saída
            output[r:r+PATCH_SIZE, c:c+PATCH_SIZE] = final_patch

            curr_step += 1
            if curr_step % 50 == 0:
                print(f"Processando: {curr_step}/{total_steps}", end='\r')

    return output.cpu().numpy()

# --- 4. Execução ---
try:
    texture_filename = 'grama.jpg' # Use suas imagens
    target_filename = 'kenji3.png'

    texture_img = cv2.imread(texture_filename)
    target_img = cv2.imread(target_filename)

    if texture_img is None or target_img is None:
        raise ValueError("Imagens não encontradas!")

    # Normaliza
    texture_img = texture_img.astype(np.float32) / 255.0
    target_img = target_img.astype(np.float32) / 255.0

    # Roda Quilting
    result = quilt_transfer_gpu(texture_img, target_img)

    # Exibe
    result_display = (np.clip(result, 0, 1) * 255).astype(np.uint8)
    print("\n--- ✅ Concluído ---")
    cv2_imshow(result_display)
    cv2.imwrite('resultado_gpu.png', result_display)

except Exception as e:
    print(f"\n❌ Erro: {e}")
