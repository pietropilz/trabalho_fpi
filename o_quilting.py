import numpy as np
from PIL import Image
import random
import cv2
import os

# --- Configurações ---
TEXTURE_PATH = "c.jpg"          
OUTPUT_PATH = "resultado_min_cut.png"
OUT_SIZE = (1024, 1024)         
PATCH_SIZE = 128                
OVERLAP = 32                    
TOLERANCE = 0.1                 

# ------------------------------------------------------------------
# 1. UTILITÁRIOS
# ------------------------------------------------------------------
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

# ------------------------------------------------------------------
# 2. MINIMUM ERROR BOUNDARY CUT (O CÉREBRO NOVO)
# ------------------------------------------------------------------

def compute_min_cut_path(error_surface):
    """
    Usa Programação Dinâmica para encontrar o caminho vertical de menor custo
    através de uma superfície de erro.
    Retorna uma lista de índices x para cada linha y.
    """
    H, W = error_surface.shape
    
    # Matriz de energia acumulada
    E = error_surface.copy()
    
    # Matriz para rastrear o caminho (backtracking)
    path_trace = np.zeros((H, W), dtype=np.int32)
    
    # 1. Forward Pass (Calcula custos acumulados)
    for i in range(1, H):
        for j in range(W):
            # Define limites para não sair do array (j-1, j, j+1)
            prev_x_start = max(0, j - 1)
            prev_x_end = min(W, j + 2)
            
            # Pega os vizinhos da linha anterior
            prev_row_slice = E[i-1, prev_x_start:prev_x_end]
            
            # Encontra o valor mínimo entre os vizinhos
            min_val = np.min(prev_row_slice)
            
            # Índice relativo do mínimo (0, 1 ou 2)
            min_idx_rel = np.argmin(prev_row_slice)
            
            # Índice absoluto na matriz
            min_idx_abs = prev_x_start + min_idx_rel
            
            # Acumula erro
            E[i, j] += min_val
            path_trace[i, j] = min_idx_abs

    # 2. Backward Pass (Reconstrói o caminho)
    path = np.zeros(H, dtype=np.int32)
    
    # Começa pelo pixel de menor erro na última linha
    path[-1] = np.argmin(E[-1])
    
    for i in range(H - 2, -1, -1):
        # Olha para o trace para saber de onde veio
        path[i] = path_trace[i + 1, path[i+1]]
        
    return path

def create_cut_mask(overlap_region_bg, overlap_region_new, axis=0):
    """
    Gera uma máscara binária (0 ou 1) baseada no corte de erro mínimo.
    axis=0: Corte Vertical (para overlap esquerdo)
    axis=1: Corte Horizontal (para overlap superior)
    """
    # Se for corte horizontal, transpomos as imagens para tratar como vertical
    if axis == 1:
        overlap_region_bg = np.transpose(overlap_region_bg, (1, 0, 2))
        overlap_region_new = np.transpose(overlap_region_new, (1, 0, 2))

    # 1. Calcula o Erro Quadrático (SSD) entre as duas sobreposições
    # Somamos os canais de cor para ter um mapa de erro 2D
    diff = overlap_region_bg - overlap_region_new
    error_map = np.sum(diff ** 2, axis=2)

    # 2. Encontra o caminho de menor erro
    path = compute_min_cut_path(error_map)

    # 3. Cria a máscara baseada no caminho
    H, W = error_map.shape
    mask = np.zeros((H, W), dtype=np.float32)
    
    for i in range(H):
        # Tudo à direita do caminho (incluindo o caminho) vira 1 (usa a nova imagem)
        # Tudo à esquerda vira 0 (mantém a imagem de fundo)
        mask[i, path[i]:] = 1.0

    # Des-transpõe se necessário
    if axis == 1:
        mask = np.transpose(mask, (1, 0))

    return mask

# ------------------------------------------------------------------
# 3. BLENDING (COM SUAVIZAÇÃO DA MÁSCARA DE CORTE)
# ------------------------------------------------------------------

def laplacian_blend_with_cut(img_base, img_new, mask, levels=4):
    """
    Aplica o corte de menor erro (já contido em 'mask'), suaviza a borda binária,
    e depois usa pirâmides Laplacianas para fundir as frequências.
    
    img_base: Região existente no output.
    img_new: Novo patch (src).
    mask: Máscara binária (0/1) gerada pela lógica de Minimum Error Cut.
    levels: Número de níveis da pirâmide.
    """
    # 1. Suavização da Máscara (após o Min Cut)
    # Isso é crucial para evitar serrilhados (aliasing) no corte.
    mask_blurred = cv2.GaussianBlur(mask, (5, 5), 2)
    
    # 2. Inicialização das Pirâmides
    # Garante float32
    GA = img_base.astype(np.float32)
    GB = img_new.astype(np.float32)
    GM = mask_blurred.astype(np.float32) # Usa a máscara suave
    
    # 1. Construir Pirâmides Gaussianas
    gpA = [GA]
    gpB = [GB]
    gpM = [GM] # <--- CORREÇÃO: Inicialização de gpM
    
    for i in range(levels):
        # Downsample
        GA = cv2.pyrDown(GA)
        GB = cv2.pyrDown(GB)
        GM = cv2.pyrDown(GM)
        gpA.append(GA)
        gpB.append(GB)
        gpM.append(GM)
        
    # 2. Construir Pirâmides Laplacianas
    # O topo da pirâmide é a última Gaussiana
    lpA = [gpA[levels]] 
    lpB = [gpB[levels]]
    
    for i in range(levels, 0, -1):
        # Upsample da imagem menor
        GE_A = cv2.pyrUp(gpA[i])
        GE_B = cv2.pyrUp(gpB[i])
        
        # Ajuste de tamanho se a divisão ímpar causou diferença de 1px
        h, w = gpA[i-1].shape[:2]
        GE_A = cv2.resize(GE_A, (w, h)) 
        GE_B = cv2.resize(GE_B, (w, h))
        
        # Laplaciana = Original - Upsampled(Smoothed)
        L_A = gpA[i-1] - GE_A
        L_B = gpB[i-1] - GE_B
        lpA.append(L_A)
        lpB.append(L_B)
        
    # 3. Mesclar as Pirâmides usando a Máscara
    LS = []
    # Inverter gpM para alinhar com lpA/lpB (do topo para baixo)
    gpM = gpM[::-1] # <--- CORREÇÃO: gpM agora está inicializada
    
    for la, lb, mask_level in zip(lpA, lpB, gpM):
        h, w = la.shape[:2]
        mask_level = cv2.resize(mask_level, (w, h))
        
        # 🐛 CORREÇÃO CRÍTICA (ValueError): Expande a máscara para 3 canais
        mask_3d = mask_level[:, :, np.newaxis] 
        
        # Equação de Blending: L = LA * (1-M) + LB * M
        # LB * M: Usa o patch NOVO onde a máscara é 1
        # LA * (1-M): Usa o patch BASE onde a máscara é 0
        ls = la * (1.0 - mask_3d) + lb * mask_3d
        
        LS.append(ls)

    # 4. Reconstruir Imagem Final
    ls_reconstruct = LS[0]
    for i in range(1, len(LS)):
        ls_reconstruct = cv2.pyrUp(ls_reconstruct)
        h, w = LS[i].shape[:2]
        ls_reconstruct = cv2.resize(ls_reconstruct, (w, h))
        ls_reconstruct += LS[i]

    return np.clip(ls_reconstruct, 0, 1)

# ------------------------------------------------------------------
# 5. LOOP PRINCIPAL
# ------------------------------------------------------------------

def synthesis(texture, out_size, patch_size, overlap):
    H_out, W_out = out_size
    output = np.zeros((H_out, W_out, 3), dtype=np.float32)
    step = patch_size - overlap
    num_y = (H_out - overlap) // step
    num_x = (W_out - overlap) // step
    
    print(f"Sintetizando {num_y}x{num_x} patches com Minimum Error Boundary Cut...")

    for i in range(num_y):
        for j in range(num_x):
            y_out = i * step
            x_out = j * step
            
            # --- 1. Definir regiões de overlap no Output existente ---
            ov_top = None; ov_left = None
            if i > 0: ov_top = output[y_out : y_out + overlap, x_out : x_out + patch_size]
            if j > 0: ov_left = output[y_out : y_out + patch_size, x_out : x_out + overlap]

            # --- 2. Encontrar melhor patch ---
            y_src, x_src = find_best_patch(texture, ov_top, ov_left, patch_size)
            patch = texture[y_src : y_src + patch_size, x_src : x_src + patch_size]

            # --- 3. Criar Máscara de Corte (Cut Mask) ---
            # Inicializa máscara com 1 (tudo é o novo patch)
            # A máscara será esculpida pelos cortes.
            mask = np.ones((patch_size, patch_size), dtype=np.float32)

            if i > 0:
                # Região de overlap do novo patch
                patch_ov_top = patch[:overlap, :]
                # Calcula corte horizontal
                cut_mask_top = create_cut_mask(ov_top, patch_ov_top, axis=1)
                # Aplica na máscara global
                mask[:overlap, :] *= cut_mask_top

            if j > 0:
                # Região de overlap do novo patch
                patch_ov_left = patch[:, :overlap]
                # Calcula corte vertical
                cut_mask_left = create_cut_mask(ov_left, patch_ov_left, axis=0)
                # Aplica na máscara global (multiplicação lida bem com o canto superior esquerdo)
                mask[:, :overlap] *= cut_mask_left

            # --- 4. Blending final ---
            region_curr = output[y_out : y_out + patch_size, x_out : x_out + patch_size]
            
            if i==0 and j==0:
                output[y_out:y_out+patch_size, x_out:x_out+patch_size] = patch
            else:
                # Usa Laplacian Blend para aplicar a máscara de corte suavemente
                blended = laplacian_blend_with_cut(region_curr, patch, mask, levels=4)
                output[y_out:y_out+patch_size, x_out:x_out+patch_size] = blended

            print(f"Patch {i},{j} ok", end='\r')
            
    return output

# ------------------------------------------------------------------
# EXECUÇÃO
# ------------------------------------------------------------------
if __name__ == "__main__":
    if not os.path.exists(TEXTURE_PATH):
        # Cria imagem dummy se não existir
        img = np.zeros((200, 200, 3), dtype=np.float32)
        for y in range(200):
            for x in range(200):
                img[y,x] = [(x%20)/20, (y%20)/20, ((x+y)%40)/40]
        save_image(img, TEXTURE_PATH)

    texture = load_texture(TEXTURE_PATH)
    res = synthesis(texture, OUT_SIZE, PATCH_SIZE, OVERLAP)
    save_image(res, OUTPUT_PATH)
