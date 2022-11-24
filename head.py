import math
import cv2
import mediapipe as mp
import numpy as np
from flask import Flask,render_template,Response
from flask_restful import Api, Resource
import requests
from flask import Flask

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5)

#TOPICS
# producer - angles
# consumer  - video
ANGLE_PORT = 5002
#URLS
vid_server = 'http://127.0.0.1:5000/video'

DefaultAngles = np.array([0,0,0]).tobytes()
DEfaultCenter = np.array([0,0,0]).tobytes()

def rotation_matrix_to_angles(rotation_matrix):
    x = math.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
    y = math.atan2(-rotation_matrix[2, 0], math.sqrt(rotation_matrix[0, 0] ** 2 +
                                                     rotation_matrix[1, 0] ** 2))
    z = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    return np.array([x, y, z]) * 180. / math.pi

def gen_angle(image):
    centerNdepth = DEfaultCenter
    angles = DefaultAngles
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image)
    image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
    face_coordination_in_real_world = np.array([
        [285, 528, 200],
        [285, 371, 152],
        [197, 574, 128],
        [173, 425, 108],
        [360, 574, 128],
        [391, 425, 108]
    ], dtype=np.float64)

    h, w, _ = image.shape
    face_coordination_in_image = []
    face_length = []

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for idx, lm in enumerate(face_landmarks.landmark):
                if idx in [10,152]:
                    x, y = int(lm.x * w), int(lm.y * h)
                    face_length.append([x,y])

                if idx in [1, 9, 57, 130, 287, 359]:
                    x, y = int(lm.x * w), int(lm.y * h)
                    face_coordination_in_image.append([x, y])
                    if(idx == 1):
                        center = np.array((x,y))

            face_coordination_in_image = np.array(face_coordination_in_image,
                                                dtype=np.float64)
            face_length = np.array(face_length,
                                        dtype=np.float64)

            focal_length = 1 * w
            cam_matrix = np.array([[focal_length, 0, w / 2],
                                [0, focal_length, h / 2],
                                [0, 0, 1]])

            dist_matrix = np.zeros((4, 1), dtype=np.float64)

            success, rotation_vec, transition_vec = cv2.solvePnP(
                face_coordination_in_real_world, face_coordination_in_image,
                cam_matrix, dist_matrix)

            rotation_matrix, jacobian = cv2.Rodrigues(rotation_vec)
            angles = rotation_matrix_to_angles(rotation_matrix).tobytes()
            depth = np.linalg.norm(face_length[0]-face_length[1])
            centerNdepth = np.array((center[0],center[1],depth)).tobytes()

        return(angles+b'---'+centerNdepth)


app=Flask(__name__)

             
@app.route('/')
def index():
    return render_template('angles.html')


def getAngles(vid):
    part =b''
    for line in vid:
        split = line.find(b'--frame')
        if(split==0):
            half,half2 = line.split(b'image/jpeg\r\n\r\n')
            if(part != b''):
                image = np.asarray(bytearray(part[:-2]), dtype="uint8")
                image = cv2.imdecode(image, cv2.IMREAD_COLOR)
                result = gen_angle(image)
                if(result):
                    yield result
                    #ang.send('angles',result)
                
                #cv2.imshow('live',image)
                #cv2.waitKey(10)
            part =b''
            part = half2
        else:
            part = part + line

        

@app.route('/angles')
def angle():
    r = requests.get(vid_server,stream=True)
    return Response((getAngles(r.iter_content(chunk_size=10*1024))),mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__=="__main__":
    app.run(port=ANGLE_PORT)



