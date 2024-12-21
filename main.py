
import numpy as np
import os
import pytesseract as tes
from jiwer import wer
from collections import Counter
import cv2

tes.pytesseract.tesseract_cmd = (r"C:\Users\DaaNIK\Desktop\dataset1\tesseract.exe")

def load_labels_from_file(file_path):
    labels = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split(' ', 1)  # Разделяем строку на номер и текст
            if len(parts) == 2:
                number = parts[0]  # Номер изображения
                label = parts[1]   # Метка
                labels[number] = label  # Сохраняем в словаре по номеру
    return labels

def character_match_count(reference, hypothesis):
    matches = sum(1 for ref_char, hyp_char in zip(reference, hypothesis) if ref_char == hyp_char)
    return matches


def test_recognition(rec_type, val_type, ds_dir):
    res = dict()
    labels = load_labels_from_file(os.path.join(ds_dir, 'ds.txt'))  # Загружаем метки из файла
    images = os.listdir(ds_dir)
    reswriter = open(f'{ds_dir} {rec_type} {val_type}.txt', 'w', encoding="utf-8")

    if rec_type == "str":
        for name in images:
            if name.endswith(('.png', '.jpg', '.jpeg')):
                img_number = os.path.splitext(name)[0]
                img_path = os.path.join(ds_dir, name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                print(f"{name}")
                if img is None:
                    print(f"Ошибка при загрузке изображения: {name}")
                    continue
                img = cv2.medianBlur(img, 3)
                text = tes.image_to_string(img, lang="rus+eng")
                print(f"{text}")
                res[img_number] = str(text).replace("\n", "")

    if rec_type == "augment":
        for name in images:
            if name.endswith(('.png', '.jpg', '.jpeg')):
                img_number = os.path.splitext(name)[0]
                img_path = os.path.join(ds_dir, name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                print(f"{name}")
                if img is None:
                    print(f"Ошибка при загрузке изображения: {name}")
                    continue

                img = cv2.medianBlur(img, 3)
                responses = []

                for angle in range(-20, 21):
                    (h, w) = img.shape[:2]
                    center = (w // 2, h // 2)
                    M = cv2.getRotationMatrix2D(center, angle, 1.0)
                    cos = np.abs(M[0, 0])
                    sin = np.abs(M[0, 1])
                    new_w = int((h * sin) + (w * cos))
                    new_h = int((h * cos) + (w * sin))
                    M[0, 2] += new_w / 2 - center[0]
                    M[1, 2] += new_h / 2 - center[1]
                    rotated_image = cv2.warpAffine(img, M, (new_w, new_h))
                    text = tes.image_to_string(rotated_image, lang="rus+eng")
                    responses.append(text)

                counted = Counter(responses)

                if '' in counted:
                    del counted['']
                most_common, _ = counted.most_common(1)[0] if counted else ("", 0)
                res[img_number] = str(most_common).replace("\n", "")

    # Оценка точности
    if val_type == "accuracy":
        count = sum(1 for number in labels.keys() if number in res and res[number] == labels[number])
        reswriter.write(f"accuracy: {count / len(labels):.4f}\n")
        for number in labels.keys():
            reswriter.write(f'{res.get(number, "")} : {labels[number]}\n')

    if val_type == "num":
        total_matches = 0
        for number in labels.keys():
            recognized_text = res.get(number, "")
            correct_label = labels[number]
            match_count = character_match_count(correct_label, recognized_text)
            total_matches += match_count
            reswriter.write(f'{recognized_text} : {correct_label}\n')
            total_characters_in_labels = sum(len(label) for label in labels.values())
        average_matches_per_label = total_matches / total_characters_in_labels
        reswriter.write(f"Среднее количество совпадающих символов на метку: {average_matches_per_label:.4f}\n")

    reswriter.close()

# Пример вызова функции
test_recognition("augment", "num", "dataset")