import cv2
import numpy as np
import torch
import torch.nn.functional as F
import random
from google.colab.patches import cv2_imshow

#Configurações

# Tamanho do bloco
PATCH_SIZE = 32
#overlap
OVERLAP_SIZE = int(PATCH_SIZE/6)
#0 = foco textura, 1 = foco forma
ALPHA = 0.8
#aleatoriedade fixa em 10%
TOLERANCE = 0.1

# Step menor = mais qualidade, mas mais memória
STEP = PATCH_SIZE - OVERLAP_SIZE

# Verifica GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Processando em: {device}")

#Funções Auxiliares para paralelizar (GPU)

def create_patch_bank(texture_tensor, patch_size, step):

    t = texture_tensor.permute(2, 0, 1).unsqueeze(0)
    kc, kh, kw = 3, patch_size, patch_size
    patches = F.unfold(t, kernel_size=(kh, kw), stride=step)

    patches = patches.permute(0, 2, 1).squeeze(0)
    patches = patches.view(-1, 3, kh, kw).permute(0, 2, 3, 1)

    return patches

#calcular todos os caminhos de uma vez
def get_best_patch_gpu(patches_bank, target_patch, overlap_mask):

    diff_corr = patches_bank - target_patch
    err_corr = (diff_corr ** 2).sum(dim=(1, 2, 3))

    diff_overlap = diff_corr * overlap_mask
    err_overlap = (diff_overlap ** 2).sum(dim=(1, 2, 3))

    # erro Total
    total_error = ALPHA * err_corr + (1 - ALPHA) * err_overlap

    # seleção aleatoria
    min_val = torch.min(total_error)

    # evitar 0 nas divisões
    if min_val == 0:
        threshold = 0.00001
    else:
        threshold = min_val * (1 + TOLERANCE)

    # selecionar adequados
    candidates_indices = torch.nonzero(total_error <= threshold).squeeze()

    # tratar o caso de 0 ou mais de um candidato
    if candidates_indices.dim() == 0:
        idx = candidates_indices.item()
    else:
        # escolhe um aleatório da lista de tensores
        random_idx = torch.randint(0, len(candidates_indices), (1,)).item()
        idx = candidates_indices[random_idx].item()

    return patches_bank[idx]

#minimum error boundary cut
def calculate_min_cut_cpu(diff_map):

    h, w = diff_map.shape
    E = diff_map.copy()

    # acmula custos
    for r in range(1, h):
        for c in range(w):
            prev_cost = E[r-1, c]
            if c > 0: prev_cost = min(prev_cost, E[r-1, c-1])
            if c < w - 1: prev_cost = min(prev_cost, E[r-1, c+1])
            E[r, c] += prev_cost

    # caminho de volta
    mask = np.zeros_like(E, dtype=bool)
    c = np.argmin(E[-1])
    for r in range(h - 1, -1, -1):
        mask[r, c:] = True
        if r > 0:
            start = max(0, c-1)
            end = min(w, c+2)
            c = start + np.argmin(E[r-1, start:end])

    return mask

# func principal
def quilt_transfer_gpu(texture_img, target_img):

    # manda tensors pra gpu
    tex_t = torch.tensor(texture_img, dtype=torch.float32).to(device)
    tgt_t = torch.tensor(target_img, dtype=torch.float32).to(device)

    output_H, output_W, C = tgt_t.shape

    output_H = output_H - (output_H - PATCH_SIZE) % STEP
    output_W = output_W - (output_W - PATCH_SIZE) % STEP
    tgt_t = tgt_t[:output_H, :output_W, :]

    # saída
    output = torch.zeros_like(tgt_t)

    patches_bank = create_patch_bank(tex_t, PATCH_SIZE, STEP)

    rows = range(0, output_H - PATCH_SIZE + 1, STEP)
    cols = range(0, output_W - PATCH_SIZE + 1, STEP)

    total_steps = len(rows) * len(cols)
    curr_step = 0

    for r in rows:
        for c in cols:
            is_first_row = (r == 0)
            is_first_col = (c == 0)

            target_patch = tgt_t[r:r+PATCH_SIZE, c:c+PATCH_SIZE]

            #criacao da mascara de overlap
          
            overlap_mask = torch.zeros((PATCH_SIZE, PATCH_SIZE, 1), device=device)
            current_output_patch = output[r:r+PATCH_SIZE, c:c+PATCH_SIZE]

            if not is_first_row:
                overlap_mask[0:OVERLAP_SIZE, :, :] = 1.0
            if not is_first_col:
                overlap_mask[:, 0:OVERLAP_SIZE, :] = 1.0

            best_patch = get_best_patch_gpu(patches_bank, target_patch, overlap_mask)

            final_patch = best_patch.clone()

            if not is_first_row or not is_first_col:
             
             #precisa de numpy
                best_np = best_patch.cpu().numpy()
                curr_np = current_output_patch.cpu().numpy()

                if not is_first_col:
                    diff = ((best_np[:, :OVERLAP_SIZE] - curr_np[:, :OVERLAP_SIZE])**2).sum(axis=2)
                    mask_v = calculate_min_cut_cpu(diff)
                    mask_v = np.repeat(mask_v[:, :, np.newaxis], 3, axis=2)

                    final_patch[:, :OVERLAP_SIZE] = torch.tensor(
                        curr_np[:, :OVERLAP_SIZE] * (1-mask_v) + best_np[:, :OVERLAP_SIZE] * mask_v,
                        device=device
                    )

                if not is_first_row:
                    curr_top_np = output[r:r+OVERLAP_SIZE, c:c+PATCH_SIZE].cpu().numpy()
                    best_top_np = final_patch[:OVERLAP_SIZE, :].cpu().numpy()

                    diff = ((best_top_np - curr_top_np)**2).sum(axis=2).T 
                    mask_h = calculate_min_cut_cpu(diff).T
                    mask_h = np.repeat(mask_h[:, :, np.newaxis], 3, axis=2)

                    final_patch[:OVERLAP_SIZE, :] = torch.tensor(
                        curr_top_np * (1-mask_h) + best_top_np * mask_h,
                        device=device
                    )

            # saída
            output[r:r+PATCH_SIZE, c:c+PATCH_SIZE] = final_patch

            curr_step += 1
            if curr_step % 50 == 0:
                print(f"Processando: {curr_step}/{total_steps}", end='\r')

    return output.cpu().numpy()

# executar
try:
    texture_filename = 'areia.jpeg'
    target_filename = 'kenji.jpeg'

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
    print("\n Concluído!!! ---")
    cv2_imshow(result_display)
    cv2.imwrite('resultado_gpu.png', result_display)

except Exception as e:
    print(f"\n Erro {e}")
