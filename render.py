
import cv2
from flask import Flask,render_template,Response
import numpy as np
from direct.showbase.ShowBase import ShowBase
from panda3d.core import FrameBufferProperties,WindowProperties,GraphicsPipe,GraphicsOutput,Texture
from panda3d.core import PointLight,AmbientLight
from panda3d.core import Material
import requests

RENDER_PORT = 5003
vid_server = 'http://127.0.0.1:5001/video'
ang_server = 'http://127.0.0.1:5002/angles'

app=Flask(__name__)
cap = cv2.VideoCapture(0)
success,frame=cap.read()
h,w,_ = frame.shape
cap.release()

win_prop = WindowProperties.size(w,h)

#scaling parameters
scale = 0.1
headScale = 300
xScale = 330
yScale = 350
mScaleX = 1.8
mScaleY = 0.1


base = ShowBase(fStartDirect=True,windowType='offscreen')
#base = ShowBase()
bgr_tex = Texture()

def renderScene(angles,bgr_tex,pos):
    base.obj.setPos(pos[0],pos[2],pos[1])
    base.obj.setScale(1.3)
    base.obj.setHpr(angles[1],angles[0],-angles[2]) # adjusted according to angles pitch,yaw,roll
    base.graphicsEngine.renderFrame()
    bgr_img = np.frombuffer(bgr_tex.getRamImage(), dtype=np.uint8)
    bgr_img.shape = (bgr_tex.getYSize(), bgr_tex.getXSize(), bgr_tex.getNumComponents())

    return bgr_img

def stich(hair,alpha,image):
    img2 = cv2.bitwise_and(hair,alpha,mask=None)
    alpha = cv2.bitwise_not(alpha)
    img1 = cv2.bitwise_and(alpha,image,mask=None)
    return cv2.add(img1,img2)

def setup():
    base.obj = base.loader.loadModel("test.bam")
    base.obj.reparentTo(base.render)
    base.obj.setPos(0, 10, 0)
    fb_prop = FrameBufferProperties()
    fb_prop.setRgbColor(True)
    fb_prop.setRgbaBits(8, 8, 8, 0)
    fb_prop.setDepthBits(24)
    
    window = base.graphicsEngine.makeOutput(base.pipe, "cameraview", 0, fb_prop, win_prop, GraphicsPipe.BFRefuseWindow)
    disp_region = window.makeDisplayRegion()
    camera1 = base.makeCamera(window)
    disp_region.setCamera(camera1)
    plight = PointLight('plight')
    plight.setColor((0.5, 0.2, 0.2, 1))
    plnp = base.render.attachNewNode(plight)
    plnp.setPos(0, 11, -3)
    base.render.setLight(plnp)

    alight = AmbientLight('alight')
    alight.setColor((0.2, 0.2, 0.2, 1))
    alnp = base.render.attachNewNode(alight)
    base.render.setLight(alnp)

    window.addRenderTexture(bgr_tex, GraphicsOutput.RTMCopyRam, GraphicsOutput.RTPColor)
    myMaterial = Material()
    myMaterial.setShininess(5.0)
    myMaterial.setAmbient((0, 0, 1, 1))
    base.obj.setMaterial(myMaterial)


def gen_frames(vid):
    part =b''
    q = requests.get(ang_server,stream=True).iter_content(chunk_size=10*1024) # get the angles from head server
    for line in vid:
        #line = line.value
        split = line.find(b'--frame')
        if(split==0):
            half,half2 = line.split(b'image/jpeg\r\n\r\n')
            if(part != b''):
                image = np.asarray(bytearray(part[:-2]), dtype="uint8")
                image = cv2.imdecode(image, cv2.IMREAD_COLOR)
                raw = next(q)
                anglesRaw,centerNdepthRaw = raw.split(b'---')
                angles = np.frombuffer(anglesRaw,dtype='float64')
                centerNdepth = np.frombuffer(centerNdepthRaw,dtype='float64')
                depth = headScale - centerNdepth[2] # only depth
                position = (0,0,depth*scale) 
                img = renderScene(angles,bgr_tex,position)

                M = np.float32([[1,0,centerNdepth[0]-xScale-(angles[1]*mScaleX)],[0,1,centerNdepth[1]-yScale-(angles[0]*mScaleY)]])
                img = cv2.warpAffine(img,M,(w,h))
                alpha = cv2.merge((img[:,:,3],img[:,:,3],img[:,:,3]))
        
                stiched = stich(img[:,:,:3],alpha,image)
                ret,buffer=cv2.imencode('.jpg',stiched)
                frame=buffer.tobytes()

                yield(b'--frame\r\n'
                            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

            part =b''
            part = half2
        else:
            part = part + line


@app.route('/')
def index():
    return render_template('render.html')

@app.route('/render')
def render():
    r = requests.get(vid_server,stream=True)
    return Response((gen_frames(r.iter_content(chunk_size=10*1024))),mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__=="__main__":
    setup()
    app.run(port=RENDER_PORT)

