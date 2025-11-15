import cv2
import numpy as np

def multiband_blend(img1, img2, offset):
    ox, oy = offset

    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # Largura adequada para caber as duas imagens
    W = max(w1, ox + w2)
    H = max(h1, oy + h2)

    # NOVO: canvas grande
    result = np.zeros((H, W, 3), dtype=np.uint8)
    result[:h1, :w1] = img1

    # Calcular overlap válido
    x1 = ox
    x2 = min(w1, ox + w2)
    overlap_w = x2 - x1
    overlap_h = min(h1, h2)

    if overlap_w > 0:
        A = img1[:overlap_h, x1:x2]
        B = img2[:overlap_h, :overlap_w]

        mask = np.zeros((overlap_h, overlap_w), np.float32)
        mask[:, :overlap_w//2] = 1.0
        mask = cv2.GaussianBlur(mask, (51,51), 0)

        blender = cv2.detail_MultiBandBlender()
        blender.setNumBands(5)
        blender.prepare((0,0, overlap_w, overlap_h))

        blender.feed(A.astype(np.int16), (mask*255).astype(np.uint8), (0,0))
        blender.feed(B.astype(np.int16), ((1-mask)*255).astype(np.uint8), (0,0))

        blended, _ = blender.blend(None, None)
        blended = np.clip(blended, 0, 255).astype(np.uint8)

        result[:overlap_h, x1:x2] = blended

    # Copiar parte da direita
    right_w = w2 - overlap_w
    if right_w > 0:
        result[:h2, x2:x2+right_w] = img2[:h2, overlap_w:overlap_w+right_w]

    return result


def main():
    img = cv2.imread("hector.jpeg")
    img = cv2.resize(img, (600, 600))

    blended = multiband_blend(img, img, offset=(150, 0))

    cv2.imwrite("/content/blended.png", blended)
    from google.colab import files
    files.download("/content/blended.png")

main()
