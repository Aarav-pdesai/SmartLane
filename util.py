import string
import easyocr

# Initialize the OCR reader
reader = easyocr.Reader(["en"], gpu=False)

# Mapping dictionaries for character conversion
dict_char_to_int = {"O": "0", "I": "1", "J": "3", "A": "4", "G": "6", "S": "5"}

dict_int_to_char = {"0": "O", "1": "I", "3": "J", "4": "A", "6": "G", "5": "S"}

def license_complies_format(text):
    if len(text) != 7:
        return False

    if (
        (text[0] in string.ascii_uppercase or text[0] in dict_int_to_char.keys())
        and (text[1] in string.ascii_uppercase or text[1] in dict_int_to_char.keys())
        and (
            text[2] in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
            or text[2] in dict_char_to_int.keys()
        )
        and (
            text[3] in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
            or text[3] in dict_char_to_int.keys()
        )
        and (text[4] in string.ascii_uppercase or text[4] in dict_int_to_char.keys())
        and (text[5] in string.ascii_uppercase or text[5] in dict_int_to_char.keys())
        and (text[6] in string.ascii_uppercase or text[6] in dict_int_to_char.keys())
    ):
        return True
    else:
        return False


def format_license(text):
    license_plate_ = ""
    mapping = {
        0: dict_int_to_char,
        1: dict_int_to_char,
        4: dict_int_to_char,
        5: dict_int_to_char,
        6: dict_int_to_char,
        2: dict_char_to_int,
        3: dict_char_to_int,
    }
    for j in [0, 1, 2, 3, 4, 5, 6]:
        if text[j] in mapping[j].keys():
            license_plate_ += mapping[j][text[j]]
        else:
            license_plate_ += text[j]

    return license_plate_


def read_license_plate(license_plate_crop):

    detections = reader.readtext(license_plate_crop)

    for detection in detections:
        bbox, text, score = detection
        print(f"OCR detected text: {text}, Score: {score}")
        text = text.upper().replace(" ", "")
        if len(text) != 7:
            text = text[(len(text) - 7) :]

        if license_complies_format(text):
            formatted_text = format_license(text)
            print(f"Formatted license plate: {formatted_text}")
            return formatted_text, score

    return 0, 0

