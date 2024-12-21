import numpy as np
import os
import pytesseract as tes
from jiwer import wer
from collections import Counter
import easyocr
import cv2
import re

tes.pytesseract.tesseract_cmd = (r"C:\Users\DaaNIK\Desktop\dataset1\tesseract.exe")

def clean_text(text):
    text = re.sub(r'[^a-zA-Zа-яА-Я\s]', '', text)
    return text.lower()

def augmentation(ds_dir):
    images = os.listdir(f"{ds_dir}")
    for name in images:
        if name.endswith(('.png', '.jpg', '.jpeg')):
            image = cv2.imread(f"{ds_dir}/{name}")
            for angle in range(-20, 21):
                (h, w) = image.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                cos = np.abs(M[0, 0])
                sin = np.abs(M[0, 1])
                new_w = int((h * sin) + (w * cos))
                new_h = int((h * cos) + (w * sin))
                M[0, 2] += new_w / 2 - center[0]
                M[1, 2] += new_h / 2 - center[1]
                rotated_image = cv2.warpAffine(image, M, (new_w, new_h))
                str_im = str(name[:-4]) + "+" + str(angle)
                cv2.imwrite(f"dataset2/{str_im}.jpg", rotated_image)


def load_labels_from_file(file_path):
    labels = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split(' ', 1)
            if len(parts) == 2:
                number = parts[0]
                label = parts[1]
                labels[number] = label
    return labels


def character_match_count(reference, hypothesis):
    matches = sum(1 for ref_char, hyp_char in zip(reference, hypothesis) if ref_char == hyp_char)
    return matches


def test_recognition(rec_type, val_type, ds_dir):
    res = dict()
    labels = load_labels_from_file(os.path.join('dataset/ds.txt'))
    images = os.listdir(ds_dir)
    reswriter = open(f'{ds_dir} {rec_type} {val_type}.txt', 'w', encoding="utf-8")

    if rec_type == "str":
        for name in images:
            if name.endswith(('.png', '.jpg', '.jpeg')):
                img_number = os.path.splitext(name)[0]
                plus_index = img_number.find("+")
                if plus_index != -1:
                    img_number = img_number[:plus_index]
                img_path = os.path.join(ds_dir, name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

                if img is None:
                    print(f"Ошибка при загрузке изображения: {name}")
                    continue

                img = cv2.medianBlur(img, 3)
                text = tes.image_to_string(img, lang="rus+eng")
                print(os.path.splitext(name)[0])
                # Запоминаем текст для текущего изображения
                res[os.path.splitext(name)[0]] = clean_text(str(text).replace("\n", ""))

    if rec_type == "augment":
        for name in images:
            if name.endswith(('.png', '.jpg', '.jpeg')):
                img_number = os.path.splitext(name)[0]
                plus_index = img_number.find("+")
                if plus_index != -1:
                    img_number = img_number[:plus_index]

                img_path = os.path.join(ds_dir, name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

                if img is None:
                    print(f"Ошибка при загрузке изображения: {name}")
                    continue
                print(img_number)
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

                most_common_text = counted.most_common(1)[0][0] if counted else ""

                # Запоминаем текст для текущего изображения
                res[img_number] = clean_text(str(most_common_text).replace("\n", ""))

    elif rec_type == "easyOCR":
        reader = easyocr.Reader(['en', 'ru'], gpu=True)
        for name in images:
            if name.endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(ds_dir, name)
                print(name)
                results = reader.readtext(img_path, detail=0, paragraph=True)

                text_combined = ' '.join(results).replace("\n", "")


                res[os.path.splitext(name)[0]] = clean_text(text_combined)

    count_correct = 0
    match_count = 0
    total_characters_in_labels = 0
    col = 0
    # Оценка точности
    for number in res:
        recognized_text = res.get(number, "")
        col+=1
        plus_index = number.find("+")
        if plus_index != -1:
            number = number[:plus_index]

        kom_index = number.find(".")
        if kom_index != -1:
            number = number[:kom_index]
        print(number)
        correct_label = labels[number]
        total_characters_in_labels += len(correct_label)
        reswriter.write(f'{recognized_text} : {correct_label}\n')

        # Если необходимо оценить точность или количество совпадений

        if val_type == "accuracy":
            count_correct += recognized_text == correct_label
        elif val_type == "num":
            match_count += character_match_count(correct_label, recognized_text)

    if val_type == "accuracy":
        accuracy_score = count_correct / col
        reswriter.write(f"accuracy: {accuracy_score:.4f}\n")

    elif val_type == "num":
        average_matches_per_label = match_count / total_characters_in_labels
        reswriter.write(f"Среднее количество совпадающих символов на метку: {average_matches_per_label:.4f}\n")

    reswriter.close()


# Пример вызова функции
# augmentation("dataset")
test_recognition("easyOCR", "num", "dataset2")