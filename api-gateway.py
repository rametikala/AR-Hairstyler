import requests
from flask import Flask,render_template,Response

API_PORT = 5000

app=Flask(__name__)

vid_server = 'http://127.0.0.1:5001/video'
ang_server = 'http://127.0.0.1:5002/angles'
render_server = 'http://127.0.0.1:5003/render'

@app.route('/')
def index():
    return render_template('index.html')

def getVideo(data):
    for line in data:
        yield line

@app.route('/video')
def video():
    r = requests.get(vid_server,stream=True)
    return Response((getVideo(r.iter_content(chunk_size=10*1024))),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/render')
def render():
    q = requests.get(render_server,stream=True)
    return Response((getVideo(q.iter_content(chunk_size=10*1024))),mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__=="__main__":
    app.run(port=API_PORT)