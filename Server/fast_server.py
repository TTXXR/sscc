import json
from fastapi import FastAPI, Request
import uvicorn

from Server.network_server import Server


app = FastAPI()

server = Server(model_path="C:/Users/rr/Desktop/sftp/mlp/trained1.0-d3",
                epoch=180)


@app.post('/')
def upload(request: Request):
    try:
        x = request.json().__getattribute__("data")
        output = server.forward(x)
    except Exception as e:
        return json.dumps({'message': 'fail', 'output': e}, ensure_ascii=False)
    return json.dumps({'message': 'success', 'output': output}, ensure_ascii=False)


if __name__ == '__main__':
    uvicorn.run('myapp:app', host='0.0.0.0', port=8001 )

