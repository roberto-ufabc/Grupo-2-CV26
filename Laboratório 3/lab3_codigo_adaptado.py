import cv2
import numpy as np

# ==========================================================
# ABERTURA DAS DUAS WEBCAMS
# ==========================================================
cap1 = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture(1)

if not cap1.isOpened():
    print("Erro ao abrir webcam 0")
    exit()

if not cap2.isOpened():
    print("Erro ao abrir webcam 1")
    exit()

sift = cv2.SIFT_create()
bf = cv2.BFMatcher()

while True:

    ret1, img1 = cap1.read()
    ret2, img2 = cap2.read()

    if not ret1 or not ret2:
        print("Erro na captura")
        break

    # ==========================================================
    # CONVERSÃO PARA CINZA
    # ==========================================================
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # ==========================================================
    # DETECÇÃO DE FEATURES
    # ==========================================================
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    if des1 is None or des2 is None:
        continue

    # ==========================================================
    # MATCHING
    # ==========================================================
    matches_brutos = bf.knnMatch(des1, des2, k=2)

    bons_matches = []

    for m, n in matches_brutos:
        if m.distance < 0.75 * n.distance:
            bons_matches.append(m)

    if len(bons_matches) < 10:
        cv2.imshow("Webcam 1", img1)
        cv2.imshow("Webcam 2", img2)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        continue

    # ==========================================================
    # PONTOS CORRESPONDENTES
    # ==========================================================
    src_pts = np.float32(
        [kp1[m.queryIdx].pt for m in bons_matches]
    ).reshape(-1, 1, 2)

    dst_pts = np.float32(
        [kp2[m.trainIdx].pt for m in bons_matches]
    ).reshape(-1, 1, 2)

    # ==========================================================
    # HOMOGRAFIA VIA RANSAC
    # ==========================================================
    H, mask = cv2.findHomography(
        src_pts,
        dst_pts,
        cv2.RANSAC,
        5.0
    )

    if H is None:
        continue

    # ==========================================================
    # VISUALIZAÇÃO DOS MATCHES
    # ==========================================================
    img_matches = cv2.drawMatches(
        img1,
        kp1,
        img2,
        kp2,
        bons_matches,
        None,
        matchColor=(0, 255, 0),
        matchesMask=mask.ravel().tolist(),
        flags=2
    )

    # ==========================================================
    # CRIAÇÃO DO MOSAICO
    # ==========================================================
    largura = img1.shape[1] + img2.shape[1]
    altura = max(img1.shape[0], img2.shape[0])

    warped = cv2.warpPerspective(
        img1,
        H,
        (largura, altura)
    )

    mosaico = warped.copy()
    mosaico[0:img2.shape[0], 0:img2.shape[1]] = img2

    # ==========================================================
    # CONTAGEM DE INLIERS
    # ==========================================================
    inliers = int(np.sum(mask))
    outliers = len(mask) - inliers

    cv2.putText(
        mosaico,
        f"Inliers: {inliers}  Outliers: {outliers}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    # ==========================================================
    # JANELAS
    # ==========================================================
    cv2.imshow("Matches RANSAC", img_matches)
    cv2.imshow("Mosaico", mosaico)

    tecla = cv2.waitKey(1) & 0xFF

    if tecla == ord('q'):
        break

# ==========================================================
# FINALIZAÇÃO
# ==========================================================
cap1.release()
cap2.release()
cv2.destroyAllWindows()
