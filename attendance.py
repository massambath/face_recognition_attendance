#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import face_recognition
import time
from datetime import datetime

print("Démarrage du système")

# Chargement des encodages visages connus
known_face_encodings = []
known_face_names = ["mass","lionel","assetou","bola","makhtar"]
face_to_encode_path = [r"C:\Users\massa\Desktop\etudiants\mass.JPG",r"C:\Users\massa\Desktop\etudiants\lionel.jpeg",r"C:\Users\massa\Desktop\etudiants\assetou.jpeg",r"C:\Users\massa\Desktop\etudiants\bola.jpeg",r"C:\Users\massa\Desktop\etudiants\makhtar.png"]

#Pour chaque visage à encoder on charge l'image et on encode le visage
for face_path in face_to_encode_path:
    image = face_recognition.load_image_file(face_path)
    face_encoded = face_recognition.face_encodings(image)[0]
    known_face_encodings.append(face_encoded)


video_capture = cv2.VideoCapture(1)

print("Webcam ON")

#Démmarage du classificateur en cascade 
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

print("Détection")

#Enregistrement du temps de démarrage
start_time = time.time()
# Initialisation des dictionnaires pour suivre la présence et les retards
presence = {name: False for name in known_face_names}
retard = {name: 0 for name in known_face_names}

#Boucle principale pour la capture vidéo
while True:
    # Lecture de la frame actuelle
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Utilisation du classificateur en cascade
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    
    for (x, y, w, h) in faces:
        face_frame = frame[y:y+h, x:x+w]
        rgb_frame = face_frame[:, :, ::-1]

        # Trouver tous les visages et les encodages de visages dans la trame actuelle de la vidéo

        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        # Pour chaque visage détecté, on compare l'encodage du visage aux encodages connus
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.4)

            name = "Inconnu"

            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]
                if not presence[name]:
                    presence[name] = True
                    if (time.time() - start_time) > 2*60:
                        retard[name] = (time.time() - start_time) - 2*60

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y+h - 30), (x+w, y+h), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, name, (x + 2, y+h - 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)

    cv2.imshow('Reconnaissance faciale', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print("Arrêt du système")
video_capture.release()
cv2.destroyAllWindows()

with open("Fiche_de_présence.txt", "w",encoding="utf-8") as file:
    file.write(f'Fiche de présence pour le {datetime.now().strftime("%d/%m/%Y")}\n\n')
    for name in known_face_names:
        if presence[name]:
            if retard[name] > 0:
                line = f'{name} est en retard de {retard[name]/60:.2f} minutes.\n'
                file.write(line)
            else:
                line = f'{name} est présent(e).\n'
                file.write(line)
        else:
            line = f'{name} est absent(e).\n'
            file.write(line)
            





