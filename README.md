"""Sélectionne deux points, redresse l'image et enregistre un zoom."""

import sys

import cv2
import numpy as np


MAX_DISPLAY_WIDTH = 1200
MAX_DISPLAY_HEIGHT = 800
ZOOM_MARGIN = 2.0
ZOOM_MIN_WIDTH = 40
ZOOM_OUTPUT_WIDTH = 800
BLUR_TYPES = ("moyenneur", "gaussien", "median")
MIN_BLUR_KERNEL = 3
MAX_BLUR_KERNEL = 31
BLUR_KERNEL_STEP = 2


def select_points(image):
    """Permet à l'utilisateur de sélectionner deux points sur l'image.

    L'image est affichée à une taille adaptée à l'écran. Les coordonnées des
    clics sont reconverties à l'échelle de l'image d'origine.

    Paramètres :
        image (numpy.ndarray) : image source en niveaux de gris ou en couleur.

    Valeur de retour :
        tuple[numpy.ndarray, numpy.ndarray] | None : coordonnées (x, y) des
        deux points dans l'image source, ou None si l'utilisateur quitte.
    """
    height, width = image.shape[:2]
    scale = min(
        1.0,
        MAX_DISPLAY_WIDTH / width,
        MAX_DISPLAY_HEIGHT / height,
    )
    display_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    display_image = cv2.resize(image, display_size, interpolation=cv2.INTER_AREA)
    points = []
    window_name = "Selection des points"

    def on_mouse(event, x, y, flags, parameter):
        """Ajoute un point sélectionné par clic gauche.

        Paramètres :
            event (int) : type d'événement souris signalé par OpenCV.
            x (int) : coordonnée horizontale du clic dans la fenêtre.
            y (int) : coordonnée verticale du clic dans la fenêtre.
            flags (int) : indicateurs d'état des boutons et touches.
            parameter (object) : donnée utilisateur associée au callback.

        Valeur de retour :
            None : modifie la liste des points uniquement si le clic est valide.
        """
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 2:
            # Convertit les coordonnées de l'affichage vers l'image d'origine.
            original_x = min(width - 1, max(0, round(x / scale)))
            original_y = min(height - 1, max(0, round(y / scale)))
            points.append((original_x, original_y))

    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window_name, on_mouse)

    try:
        while True:
            canvas = display_image.copy()
            for index, (point_x, point_y) in enumerate(points, start=1):
                display_point = (round(point_x * scale), round(point_y * scale))
                cv2.circle(canvas, display_point, 7, (0, 0, 255), -1)
                cv2.putText(
                    canvas,
                    str(index),
                    (display_point[0] + 10, display_point[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

            if len(points) == 2:
                first_display = tuple(round(value * scale) for value in points[0])
                second_display = tuple(round(value * scale) for value in points[1])
                cv2.line(canvas, first_display, second_display, (0, 255, 0), 2)
                instruction = "Entree : valider | r : recommencer | q : quitter"
            elif len(points) == 1:
                instruction = "Clic 2/2 : point droit"
            else:
                instruction = "Clic 1/2 : point gauche"

            cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 34), (0, 0, 0), -1)
            cv2.putText(
                canvas,
                instruction,
                (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.imshow(window_name, canvas)
            key = cv2.waitKey(20) & 0xFF

            if key == ord("q"):
                return None
            if key == ord("r"):
                points.clear()
            if key in (10, 13) and len(points) == 2:
                if points[0] == points[1]:
                    print("Erreur : les deux points doivent être différents.")
                    points.clear()
                else:
                    return (
                        np.array(points[0], dtype=np.float64),
                        np.array(points[1], dtype=np.float64),
                    )
    finally:
        cv2.destroyWindow(window_name)


def rotate_image(image, first_point, second_point):
    """Redresse l'image pour rendre horizontale la droite des deux points.

    Le canevas est agrandi pour contenir toute l'image après rotation. Les
    coordonnées des points sont transformées avec la matrice appliquée à
    l'image.

    Paramètres :
        image (numpy.ndarray) : image source à redresser.
        first_point (numpy.ndarray) : coordonnées (x, y) du point gauche.
        second_point (numpy.ndarray) : coordonnées (x, y) du point droit.

    Valeur de retour :
        tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, float] : image
        redressée, premier point transformé, second point transformé et angle
        initial en degrés (positif dans le sens horaire à l'écran).
    """
    height, width = image.shape[:2]
    delta = second_point - first_point
    angle = float(np.degrees(np.arctan2(delta[1], delta[0])))

    center = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    radians = np.radians(angle)
    cosine = abs(np.cos(radians))
    sine = abs(np.sin(radians))
    new_width = int(np.ceil(height * sine + width * cosine))
    new_height = int(np.ceil(height * cosine + width * sine))

    # Décale la matrice pour centrer l'image tournée dans le canevas agrandi.
    matrix[0, 2] += new_width / 2.0 - center[0]
    matrix[1, 2] += new_height / 2.0 - center[1]

    rotated = cv2.warpAffine(
        image,
        matrix,
        (new_width, new_height),
        flags=cv2.INTER_CUBIC,
    )

    points = np.array([[first_point, second_point]], dtype=np.float64)
    transformed = cv2.transform(points, matrix)[0]
    return rotated, transformed[0], transformed[1], angle


def zoom_on_points(image, first_point, second_point):
    """Recadre et agrandit la zone située autour des deux points.

    Le recadrage est centré sur le milieu des points, avec une largeur égale à
    deux fois leur distance (au minimum ZOOM_MIN_WIDTH) et une hauteur égale
    aux trois quarts de sa largeur. Ses limites sont clampées à l'image.

    Paramètres :
        image (numpy.ndarray) : image redressée.
        first_point (numpy.ndarray) : premier point dans l'image redressée.
        second_point (numpy.ndarray) : second point dans l'image redressée.

    Valeur de retour :
        numpy.ndarray : recadrage redimensionné à ZOOM_OUTPUT_WIDTH pixels de
        largeur, ou image vide si le recadrage ne peut pas être réalisé.
    """
    height, width = image.shape[:2]
    distance = float(np.linalg.norm(second_point - first_point))
    desired_width = max(ZOOM_MIN_WIDTH, ZOOM_MARGIN * distance)
    desired_height = desired_width * 3.0 / 4.0

    # Réduit le rectangle si nécessaire tout en conservant son rapport d'aspect.
    fit_scale = min(1.0, width / desired_width, height / desired_height)
    crop_width = max(1, min(width, int(round(desired_width * fit_scale))))
    crop_height = max(1, min(height, int(round(desired_height * fit_scale))))

    midpoint = (first_point + second_point) / 2.0
    left = round(midpoint[0] - crop_width / 2.0)
    top = round(midpoint[1] - crop_height / 2.0)
    # Clamp les coordonnées de départ afin que le recadrage reste dans l'image.
    left = min(max(0, left), width - crop_width)
    top = min(max(0, top), height - crop_height)

    crop = image[top : top + crop_height, left : left + crop_width]
    if crop.size == 0:
        return crop

    output_height = max(1, round(ZOOM_OUTPUT_WIDTH * crop_height / crop_width))
    return cv2.resize(
        crop,
        (ZOOM_OUTPUT_WIDTH, output_height),
        interpolation=cv2.INTER_CUBIC,
    )


def apply_blur(image, blur_type, kernel_size):
    """Applique un flou à une copie de l'image, sans modifier l'originale.

    Paramètres :
        image (numpy.ndarray) : image d'entrée à flouter.
        blur_type (str) : type de flou, parmi « moyenneur », « gaussien »
            et « median ».
        kernel_size (int) : taille impaire du noyau, comprise entre 3 et 31.

    Valeur de retour :
        numpy.ndarray : copie floutée de l'image d'entrée.

    Lève :
        ValueError : si le type de flou ou la taille du noyau est invalide.
    """
    if blur_type not in BLUR_TYPES:
        raise ValueError(
            f"Type de flou invalide : {blur_type}. "
            f"Valeurs possibles : {', '.join(BLUR_TYPES)}."
        )
    if (
        not isinstance(kernel_size, (int, np.integer))
        or kernel_size < MIN_BLUR_KERNEL
        or kernel_size > MAX_BLUR_KERNEL
        or kernel_size % 2 == 0
    ):
        raise ValueError(
            f"La taille du noyau doit être impaire et comprise entre "
            f"{MIN_BLUR_KERNEL} et {MAX_BLUR_KERNEL}."
        )

    # Chaque filtre est calculé depuis l'image source, jamais depuis un flou précédent.
    if blur_type == "moyenneur":
        return cv2.blur(image.copy(), (kernel_size, kernel_size))
    if blur_type == "gaussien":
        return cv2.GaussianBlur(image.copy(), (kernel_size, kernel_size), 0)
    return cv2.medianBlur(image.copy(), kernel_size)


def main():
    """Charge, redresse et affiche interactivement une image et son zoom.

    Le chemin d'accès est lu depuis le premier argument de la ligne de commande
    ou demandé interactivement si aucun argument n'est fourni. La prévisualisation
    permet d'activer ou désactiver un flou, d'en modifier le type et le noyau,
    puis de sauvegarder le zoom affiché.

    Paramètres :
        Aucun.

    Valeur de retour :
        int : code de sortie du programme, égal à 0 si l'utilisateur quitte la
        prévisualisation avec succès et non nul en cas d'erreur ou d'annulation.
    """
    image_path = sys.argv[1] if len(sys.argv) > 1 else input("Chemin de l'image : ").strip()
    if not image_path:
        print("Erreur : aucun chemin d'image fourni.")
        return 1

    image = cv2.imread(image_path)
    if image is None:
        print(f"Erreur : image introuvable ou illisible : {image_path}")
        return 1

    try:
        selected_points = select_points(image)
        if selected_points is None:
            print("Sélection annulée.")
            return 1

        first_point, second_point = selected_points
        rotated, rotated_first, rotated_second, angle = rotate_image(
            image,
            first_point,
            second_point,
        )

        marked = rotated.copy()
        for index, point in enumerate((rotated_first, rotated_second), start=1):
            location = tuple(np.rint(point).astype(int))
            cv2.circle(marked, location, 9, (0, 0, 255), -1)
            cv2.putText(
                marked,
                str(index),
                (location[0] + 12, location[1] - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        original_zoom = zoom_on_points(rotated, rotated_first, rotated_second)
        if original_zoom.size == 0:
            print("Erreur : impossible de créer le recadrage du zoom.")
            return 1

        print(f"Angle d'inclinaison initial : {angle:.2f} degrés")
        image_window = "Image redressee"
        zoom_window = "Zoom"
        cv2.namedWindow(image_window, cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow(zoom_window, cv2.WINDOW_AUTOSIZE)

        blur_active = False
        blur_type_index = 0
        kernel_size = 7

        while True:
            if blur_active:
                displayed_image = apply_blur(
                    marked,
                    BLUR_TYPES[blur_type_index],
                    kernel_size,
                )
                displayed_zoom = apply_blur(
                    original_zoom,
                    BLUR_TYPES[blur_type_index],
                    kernel_size,
                )
                blur_status = (
                    f"Flou : {BLUR_TYPES[blur_type_index]}, "
                    f"noyau {kernel_size}x{kernel_size}"
                )
            else:
                displayed_image = marked.copy()
                displayed_zoom = original_zoom.copy()
                blur_status = "Flou : OFF"

            zoom_preview = displayed_zoom.copy()
            cv2.putText(
                zoom_preview,
                blur_status,
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                zoom_preview,
                blur_status,
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                zoom_preview,
                "b: flou | m: type | +/-: noyau | s: sauvegarder | q: quitter",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (0, 0, 0),
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                zoom_preview,
                "b: flou | m: type | +/-: noyau | s: sauvegarder | q: quitter",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            cv2.imshow(image_window, displayed_image)
            cv2.imshow(zoom_window, zoom_preview)
            key = cv2.waitKey(30) & 0xFF

            if key == ord("q"):
                return 0
            if key == ord("b"):
                blur_active = not blur_active
            elif key == ord("m"):
                blur_type_index = (blur_type_index + 1) % len(BLUR_TYPES)
            elif key in (ord("+"), ord("=")):
                # Le pas de deux préserve un noyau impair.
                kernel_size = min(
                    MAX_BLUR_KERNEL,
                    kernel_size + BLUR_KERNEL_STEP,
                )
            elif key in (ord("-"), ord("_")):
                kernel_size = max(
                    MIN_BLUR_KERNEL,
                    kernel_size - BLUR_KERNEL_STEP,
                )
            elif key == ord("s"):
                output_path = (
                    "resultat_zoom_flou.png"
                    if blur_active
                    else "resultat_zoom.png"
                )
                if cv2.imwrite(output_path, displayed_zoom):
                    print(f"Zoom sauvegardé dans {output_path}.")
                else:
                    print(f"Erreur : impossible d'enregistrer {output_path}.")
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    sys.exit(main())
