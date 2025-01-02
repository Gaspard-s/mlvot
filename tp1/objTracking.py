import cv2
import numpy as np
from KalmanFilter import KalmanFilter
from Detector import detect  # Supposant que vous avez ce fichier

def main():
    kf = KalmanFilter(dt=0.1, 
                      u_x=1, 
                      u_y=1, 
                      std_acc=1, 
                      x_std_meas=0.1,
                      y_std_meas=0.1)
    
    cap = cv2.VideoCapture('randomball.avi')

    
    trajectory = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Fin de la vidéo")
            break       
            
        centroid = detect(frame)
        
        if centroid is not None and len(centroid) == 2:
            try:
                centroid_measurement = np.array([[centroid[0]], [centroid[1]]])
                
                predicted_state = kf.predict()
                
                estimated_state = kf.update(centroid_measurement)
                trajectory.append((int(estimated_state[0][0]), int(estimated_state[1][0])))
                
                cv2.circle(frame, (int(centroid[0]), int(centroid[1])), 
                          5, (0, 255, 0), -1)
                cv2.rectangle(frame, 
                             (int(predicted_state[0][0]) - 15, int(predicted_state[1][0]) - 15),
                             (int(predicted_state[0][0]) + 15, int(predicted_state[1][0]) + 15),
                             (255, 0, 0), 2)
                cv2.rectangle(frame,
                             (int(estimated_state[0][0]) - 15, int(estimated_state[1][0]) - 15),
                             (int(estimated_state[0][0]) + 15, int(estimated_state[1][0]) + 15),
                             (0, 0, 255), 2)
                
                if len(trajectory) > 1:
                    for i in range(1, len(trajectory)):
                        cv2.line(frame, trajectory[i-1], trajectory[i], (0, 255, 255), 2)
            
            except Exception as e:
                print(f"Erreur lors du traitement: {e}")
                print(f"Centroid: {centroid}")
                continue
        

        cv2.imshow('Object Tracking', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()