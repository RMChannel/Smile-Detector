#Importo la libreria che utilizzerò per il programma, ovvero OpenCV
import cv2

#Importo i file con i tutti i dati riguardanti: sorrisi, occhi e facce
facce = cv2.CascadeClassifier('facce.xml')
occhi = cv2.CascadeClassifier('occhi.xml')
sorrisi = cv2.CascadeClassifier('sorrisi.xml')

#Creo la funzione che serve per scansionare ogni frame della webcam e a rilevare la faccia, gli occhi e il sorriso
def detection(grayscale, img):
    face = facce.detectMultiScale(grayscale, 1.3, 5)
    for (x_face, y_face, w_face, h_face) in face:
        faccia=cv2.rectangle(img, (x_face, y_face), (x_face+w_face, y_face+h_face), (255, 0, 0), 2)
        cv2.putText(faccia, 'Faccia rilevata', (x_face, y_face-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 2)
        ri_grayscale = grayscale[y_face:y_face+h_face, x_face:x_face+w_face]
        ri_color = img[y_face:y_face+h_face, x_face:x_face+w_face] 
        eye = occhi.detectMultiScale(ri_grayscale, 1.2, 15)
        for (x_eye, y_eye, w_eye, h_eye) in eye:
            eyes=cv2.rectangle(ri_color,(x_eye, y_eye),(x_eye+w_eye, y_eye+h_eye), (0, 180, 60), 2)
            cv2.putText(eyes, 'Occhio', (x_eye, y_eye-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,180,60), 2)
        smile = sorrisi.detectMultiScale(ri_grayscale, 1.7, 32)
        for (x_smile, y_smile, w_smile, h_smile) in smile: 
            sorriso=cv2.rectangle(ri_color,(x_smile, y_smile),(x_smile+w_smile, y_smile+h_smile), (255, 0, 130), 2)
            cv2.putText(sorriso, 'Sorriso rilevato', (x_smile, y_smile-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,130), 2)
    return img 

#Qui si apre il ciclo per l'apertura e la scansione dell'immagine della webcam, che poi viene mostrata a schermo
video_capture = cv2.VideoCapture(0)
while video_capture.isOpened():
	_, frame = video_capture.read()
	#Prendo l'immagine dalla webcam e la faccia diventare monocromatica					
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	# Richiamo la funzione della rilevazione, poi mostro tutto a video
	img_complete = detection(gray, frame)					
	cv2.imshow('Video', img_complete)
	#Se il tasto "q" viene premuto allora si chiude il programma
	if cv2.waitKey(1) & 0xff == ord('q'):			
		break

video_capture.release()								
cv2.destroyAllWindows()