import os
os.environ['FLAGS_use_mkldnn'] = '0'

from flask import Flask, request, jsonify
import tempfile
import shutil
import re
import cv2
from paddleocr import PaddleOCR

app = Flask(__name__)
reader = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False)

def verbeter_foto(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

def zoek_datum_in_resultaten(resultaten):
    for _, tekst, _ in resultaten:
        tekst = tekst.replace('O','0').replace('Z','2').replace('l','1')
        tekst = tekst.replace('S','5').replace('G','6').replace('B','8')
        tekst = tekst.replace(' ','.')
        datums = re.findall(r'\d{2}[-./]\d{2}[-./]\d{2,4}', tekst)
        for d in datums:
            delen = re.split(r'[-./]', d)
            try:
                dag, maand, jaar = int(delen[0]), int(delen[1]), int(delen[2])
                if jaar < 100:
                    jaar += 2000
                if 1 <= dag <= 31 and 1 <= maand <= 12 and 2020 <= jaar <= 2035:
                    return f"{dag:02d}-{maand:02d}-{jaar}"
            except:
                pass
    return None

def zoom_op_cijfers(img):
    resultaat = reader.ocr(img, cls=True)
    if not resultaat or not resultaat[0]:
        return [], None

    alle_resultaten = [(lijn[0], lijn[1][0], lijn[1][1]) for lijn in resultaat[0]]
    datum = zoek_datum_in_resultaten(alle_resultaten)
    if datum:
        return alle_resultaten, datum

    # Zoek blokken met cijfers en zoom in
    for lijn in resultaat[0]:
        tekst = lijn[1][0]
        if not re.search(r'\d{2}', tekst):
            continue

        pts = lijn[0]
        x_min = max(0, int(min(p[0] for p in pts)) - 10)
        y_min = max(0, int(min(p[1] for p in pts)) - 10)
        x_max = min(img.shape[1], int(max(p[0] for p in pts)) + 10)
        y_max = min(img.shape[0], int(max(p[1] for p in pts)) + 10)

        uitsnede = img[y_min:y_max, x_min:x_max]
        if uitsnede.size == 0:
            continue

        for zoom in [2, 3, 4]:
            vergroot = cv2.resize(uitsnede, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_CUBIC)
            zoom_resultaat = reader.ocr(vergroot, cls=True)
            if not zoom_resultaat or not zoom_resultaat[0]:
                continue
            zoom_lijnen = [(l[0], l[1][0], l[1][1]) for l in zoom_resultaat[0]]
            datum = zoek_datum_in_resultaten(zoom_lijnen)
            if datum:
                return zoom_lijnen, datum

    return alle_resultaten, None

@app.route('/scan', methods=['POST'])
def scan():
    if 'foto' not in request.files:
        return jsonify({'error': 'geen foto'}), 400

    foto = request.files['foto']

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
    tmpnaam = tmp.name
    tmp.close()
    foto.save(tmpnaam)

    # Kopie bewaren zodat je de laatste scan kan bekijken
    shutil.copyfile(tmpnaam, 'laatste_foto.jpg')

    print("Opgeslagen:", tmpnaam)
    print("Bestaat:", os.path.exists(tmpnaam))
    print("Grootte:", os.path.getsize(tmpnaam))

    img = cv2.imread(tmpnaam)

    if img is None or img.size == 0:
        os.unlink(tmpnaam)
        return jsonify({'error': 'afbeelding ongeldig of leeg'}), 400

    try:
        rotaties = [
            ("0 graden", img),
            ("90 graden rechtsom", cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)),
            ("180 graden", cv2.rotate(img, cv2.ROTATE_180)),
            ("90 graden linksom", cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)),
        ]

        beste_resultaten = []
        datum = None

        for naam, gedraaid in rotaties:
            print(f"Proberen: {naam}...")
            resultaten, datum = zoom_op_cijfers(gedraaid)
            if resultaten:
                beste_resultaten = resultaten
            if datum:
                print(f"Datum gevonden bij {naam}: {datum}")
                break

            # ook proberen met verbeterd contrast, per rotatie
            gedraaid_verbeterd = verbeter_foto(gedraaid)
            resultaten, datum = zoom_op_cijfers(gedraaid_verbeterd)
            if resultaten:
                beste_resultaten = resultaten
            if datum:
                print(f"Datum gevonden bij {naam} (verbeterd contrast): {datum}")
                break

        if datum:
            return jsonify({'datum': datum})

        print("Ruwe OCR-tekst (geen datum gevonden):", [t for _, t, _ in beste_resultaten])
        return jsonify({'datum': None, 'tekst': [t for _, t, _ in beste_resultaten]})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

    finally:
        try:
            os.unlink(tmpnaam)
        except:
            pass

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)