import numpy as np
from PIL import Image
import random
import cv2
import os

# Configurações - varia conforme imagem
TEXTURE_PATH = "grama.jpg"
OUTPUT_PATH = "resultado.png"
OUT_SIZE = (1024, 1024)         
PATCH_SIZE = 128                
OVERLAP = int(PATCH_SIZE/6)                   
TOLERANCE = 0.1                 

#carrega e salva, cria ruído para rodar independentemente
def load_texture(path):
    if not os.path.exists(path):
        print(f"Arquivo {path} não encontrado. Gerando ruído.")
        return np.random.rand(200, 200, 3).astype(np.float32)
    img = Image.open(path).convert("RGB")
    return np.array(img) / 255.0

def save_image(img_array, path):
    img_array = np.clip(img_array, 0, 1)
    img = Image.fromarray((img_array * 255).astype(np.uint8))
    img.save(path)
    print(f"Salvo em: {path}")

# MINIMUM ERROR BOUNDARY CUT
def compute_min_cut_path(error_surface):
    H, W = error_surface.shape
    E = error_surface.copy()
    path_trace = np.zeros((H, W), dtype=np.int32)
    
    # calcular custos acumulados
    for i in range(1, H):
        for j in range(W):
            # limites pra não sair do array (j-1, j, j+1)
            prev_x_start = max(0, j - 1)
            prev_x_end = min(W, j + 2)
            
            # Pega os vizinhos
            prev_row_slice = E[i-1, prev_x_start:prev_x_end]
            
            #valor mínimo entre os vizinhos
            min_val = np.min(prev_row_slice)
            
            min_idx_rel = np.argmin(prev_row_slice)
            min_idx_abs = prev_x_start + min_idx_rel
            
            # Aaumular erro
            E[i, j] += min_val
            path_trace[i, j] = min_idx_abs

    # Reconstruir caminho
    path = np.zeros(H, dtype=np.int32)
    path[-1] = np.argmin(E[-1])
    
    for i in range(H - 2, -1, -1):
        path[i] = path_trace[i + 1, path[i+1]]
        
    return path

#corte de erro mín
def create_cut_mask(overlap_region_bg, overlap_region_new, axis=0):
    if axis == 1:
        overlap_region_bg = np.transpose(overlap_region_bg, (1, 0, 2))
        overlap_region_new = np.transpose(overlap_region_new, (1, 0, 2))

    #calcular o erro quadrático entre as duas sobreposições
    diff = overlap_region_bg - overlap_region_new
    error_map = np.sum(diff ** 2, axis=2)

    # encontra o caminho de menor erro
    path = compute_min_cut_path(error_map)

    #cria a máscara baseada no caminho
    H, W = error_map.shape
    mask = np.zeros((H, W), dtype=np.float32)
    
    for i in range(H):
        # Tudo à direita do caminho vira 1 (substitui fundo)
        # Tudo à esquerda vira 0 (mantém o fundo)
        mask[i, path[i]:] = 1.0

    if axis == 1:
        mask = np.transpose(mask, (1, 0))

    return mask


#suavizar para melhor resultds usando pirâmides laplacianas -- IA usada para aprimorar resultados
def laplacian_blend_with_cut(img_base, img_new, mask, levels=4):
    mask_blurred = cv2.GaussianBlur(mask, (5, 5), 2) # máscara suave
    
    # inicialização das pirâmides
    GA = img_base.astype(np.float32)
    GB = img_new.astype(np.float32)
    GM = mask_blurred.astype(np.float32) 
    
    #Construir Pirâmides Gaussianas
    gpA = [GA]
    gpB = [GB]
    gpM = [GM]
    
    for i in range(levels):
        GA = cv2.pyrDown(GA)
        GB = cv2.pyrDown(GB)
        GM = cv2.pyrDown(GM)
        gpA.append(GA)
        gpB.append(GB)
        gpM.append(GM)

    lpA = [gpA[levels]] 
    lpB = [gpB[levels]]
    
    for i in range(levels, 0, -1):
        GE_A = cv2.pyrUp(gpA[i])
        GE_B = cv2.pyrUp(gpB[i])
        
        #ajuste de tamanho se a divisão ímpar
        h, w = gpA[i-1].shape[:2]
        GE_A = cv2.resize(GE_A, (w, h)) 
        GE_B = cv2.resize(GE_B, (w, h))
        
        # Laplaciana = Original - suavizada
        L_A = gpA[i-1] - GE_A
        L_B = gpB[i-1] - GE_B
        lpA.append(L_A)
        lpB.append(L_B)
        
    #mesclagem as pirâmides usando a máscara
    LS = []
    gpM = gpM[::-1] 
    
    for la, lb, mask_level in zip(lpA, lpB, gpM):
        h, w = la.shape[:2]
        mask_level = cv2.resize(mask_level, (w, h))
        mask_3d = mask_level[:, :, np.newaxis] 
        ls = la * (1.0 - mask_3d) + lb * mask_3d
        LS.append(ls)

    # 
    ls_reconstruct = LS[0]
    for i in range(1, len(LS)):
        ls_reconstruct = cv2.pyrUp(ls_reconstruct)
        h, w = LS[i].shape[:2]
        ls_reconstruct = cv2.resize(ls_reconstruct, (w, h))
        ls_reconstruct += LS[i]

    return np.clip(ls_reconstruct, 0, 1)

#melhor caminho
def find_best_patch(texture, overlap_top, overlap_left, patch_size):
    H, W, _ = texture.shape
    search_img = (texture * 255).astype(np.uint8)
    
    total_error = np.zeros((H - patch_size, W - patch_size), dtype=np.float32)

    if overlap_top is not None:
        t_top = (overlap_top * 255).astype(np.uint8)
        ssd_top = cv2.matchTemplate(search_img, t_top, cv2.TM_SQDIFF)
        total_error += ssd_top[:H-patch_size, :W-patch_size]

    if overlap_left is not None:
        t_left = (overlap_left * 255).astype(np.uint8)
        ssd_left = cv2.matchTemplate(search_img, t_left, cv2.TM_SQDIFF)
        total_error += ssd_left[:H-patch_size, :W-patch_size]

    # Seleção com tolerância para evitar repetição
    min_val, _, _, _ = cv2.minMaxLoc(total_error)
    limit = max(min_val * (1 + TOLERANCE), min_val + 1.0)
    
    y_locs, x_locs = np.where(total_error <= limit)
    
    if len(y_locs) == 0:
        return random.randint(0, H-patch_size), random.randint(0, W-patch_size)
        
    idx = random.randint(0, len(y_locs) - 1)
    return y_locs[idx], x_locs[idx]

#loop principal
def synthesis(texture, out_size, patch_size, overlap):
    H_out, W_out = out_size
    output = np.zeros((H_out, W_out, 3), dtype=np.float32)
    step = patch_size - overlap
    num_y = (H_out - overlap) // step
    num_x = (W_out - overlap) // step

    for i in range(num_y):
        for j in range(num_x):
            y_out = i * step
            x_out = j * step
            
            #definir overlap
            ov_top = None; ov_left = None
            if i > 0: ov_top = output[y_out : y_out + overlap, x_out : x_out + patch_size]
            if j > 0: ov_left = output[y_out : y_out + patch_size, x_out : x_out + overlap]

            #melhor caminho
            y_src, x_src = find_best_patch(texture, ov_top, ov_left, patch_size)
            patch = texture[y_src : y_src + patch_size, x_src : x_src + patch_size]

            #máscara de corte
            mask = np.ones((patch_size, patch_size), dtype=np.float32)

            if i > 0:
                #novo patch com corte horizontal
                patch_ov_top = patch[:overlap, :]
                cut_mask_top = create_cut_mask(ov_top, patch_ov_top, axis=1)
                mask[:overlap, :] *= cut_mask_top

            if j > 0:
                #novo patch com corte vertical
                patch_ov_left = patch[:, :overlap]
                cut_mask_left = create_cut_mask(ov_left, patch_ov_left, axis=0)
                mask[:, :overlap] *= cut_mask_left

        
            region_curr = output[y_out : y_out + patch_size, x_out : x_out + patch_size]
            
            if i==0 and j==0:
                output[y_out:y_out+patch_size, x_out:x_out+patch_size] = patch
            else:
                # Usa Laplacian Blend para aplicar a máscara de corte suavemente
                blended = laplacian_blend_with_cut(region_curr, patch, mask, levels=4)
                output[y_out:y_out+patch_size, x_out:x_out+patch_size] = blended

            print(f"Patch {i},{j} ok", end='\r')
            
    return output


if __name__ == "__main__":
    if not os.path.exists(TEXTURE_PATH):
        #se não hover imagem, encerra e sai
        print(f"ERRO: O arquivo de textura não foi encontrado no caminho: {TEXTURE_PATH}")
        sys.exit(1)

    texture = load_texture(TEXTURE_PATH)
    res = synthesis(texture, OUT_SIZE, PATCH_SIZE, OVERLAP)
    save_image(res, OUTPUT_PATH)
