import numpy as np
import cv2 as cv

MIN_MATCH_COUNT = 10

# Abre duas webcams
# Normalmente:
# 0 = webcam principal/notebook
# 1 = webcam externa
cap1 = cv.VideoCapture(0)
cap2 = cv.VideoCapture(1)

if not cap1.isOpened():
    print("Erro: não foi possível abrir a webcam 0")
    exit()

if not cap2.isOpened():
    print("Erro: não foi possível abrir a webcam 1")
    exit()

# Cria detector SIFT
sift = cv.SIFT_create()

# Configuração do FLANN
FLANN_INDEX_KDTREE = 1
index_params = dict(
    algorithm=FLANN_INDEX_KDTREE,
    trees=5
)

search_params = dict(
    checks=50
)

flann = cv.FlannBasedMatcher(index_params, search_params)

while True:
    # Lê frames das duas webcams
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    if not ret1 or not ret2:
        print("Erro ao capturar frame de uma das webcams")
        break

    # Converte para escala de cinza
    img1 = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
    img2 = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)

    # Detecta keypoints e descritores
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    matchesMask = None
    good = []

    # Evita erro caso uma das imagens não tenha descritores suficientes
    if des1 is not None and des2 is not None and len(des1) >= 2 and len(des2) >= 2:

        matches = flann.knnMatch(des1, des2, k=2)

        # Lowe's ratio test
        for match in matches:
            if len(match) == 2:
                m, n = match
                if m.distance < 0.7 * n.distance:
                    good.append(m)

        if len(good) > MIN_MATCH_COUNT:
            src_pts = np.float32(
                [kp1[m.queryIdx].pt for m in good]
            ).reshape(-1, 1, 2)

            dst_pts = np.float32(
                [kp2[m.trainIdx].pt for m in good]
            ).reshape(-1, 1, 2)

            M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)

            if M is not None and mask is not None:
                matchesMask = mask.ravel().tolist()

                h, w = img1.shape
                pts = np.float32([
                    [0, 0],
                    [0, h - 1],
                    [w - 1, h - 1],
                    [w - 1, 0]
                ]).reshape(-1, 1, 2)

                dst = cv.perspectiveTransform(pts, M)

                # Desenha o contorno da região correspondente na imagem 2
                img2 = cv.polylines(
                    img2,
                    [np.int32(dst)],
                    True,
                    255,
                    3,
                    cv.LINE_AA
                )

        else:
            print(f"Not enough matches are found - {len(good)}/{MIN_MATCH_COUNT}")

    else:
        print("Descritores insuficientes em uma das webcams")

    draw_params = dict(
        matchColor=(0, 255, 0),
        singlePointColor=None,
        matchesMask=matchesMask,
        flags=2
    )

    # Desenha correspondências entre os dois frames
    img_matches = cv.drawMatches(
        img1,
        kp1,
        img2,
        kp2,
        good,
        None,
        **draw_params
    )

    # Mostra resultado em janela de vídeo
    cv.imshow("SIFT - Duas Webcams", img_matches)

    # Aperte q para sair
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Libera recursos
cap1.release()
cap2.release()
cv.destroyAllWindows()
