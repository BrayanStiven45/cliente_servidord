import socket
import threading
import numpy as np
import argparse
from sklearn.datasets import fetch_openml
import pickle
import struct
import json
import time
import csv

# ==============================
# SOCKET UTILS
# ==============================

def send_json(sock, obj):
    data = json.dumps(obj).encode()
    sock.sendall(struct.pack(">I", len(data)) + data)

def recv_json(sock):
    raw = recv_all(sock, 4)
    if raw is None:
        return None
    size = struct.unpack(">I", raw)[0]
    data = recv_all(sock, size)
    if data is None:
        return None
    return json.loads(data.decode())

def send_arrays(sock, arrays):
    data = pickle.dumps(arrays)
    sock.sendall(struct.pack(">I", len(data)) + data)

def recv_arrays(sock):
    raw = recv_all(sock, 4)
    if raw is None:
        return None
    size = struct.unpack(">I", raw)[0]
    data = recv_all(sock, size)
    if data is None:
        return None
    return pickle.loads(data)

def recv_all(sock, n):
    data = b''
    while len(data) < n:
        packet = sock.recv(n-len(data))
        if not packet:
            return None
        data += packet
    return data

# ==============================
# DATASET (loaded once)
# ==============================

def load_mnist(train_size):
    mnist = fetch_openml("mnist_784", version=1, as_frame=False)

    X = mnist.data.astype(np.float32)/255.0
    y = mnist.target.astype(np.int32)

    onehot = np.zeros((y.size,10))
    onehot[np.arange(y.size),y]=1

    return X[:train_size], onehot[:train_size], y[:train_size]

# ==============================
# PARAMETER SERVER
# ==============================

class ParameterServer:

    def __init__(self, host, port, epochs, workers, lr, hidden, train_size):

        self.host=host
        self.port=port
        self.epochs=epochs
        self.n_workers=workers
        self.lr=lr

        self.hidden=hidden
        self.train_size=train_size

        self.workers=[]
        self.lock=threading.Lock()

        print("Loading MNIST...")
        self.X,self.Y,self.y_raw=load_mnist(train_size)

        print("Dataset ready")

        input_size=self.X.shape[1]

        self.ws1=np.random.randn(hidden,input_size)*np.sqrt(2./input_size)
        self.bs1=np.zeros((1,hidden))

        self.ws2=np.random.randn(10,hidden)*np.sqrt(2./hidden)
        self.bs2=np.zeros((1,10))

        self.batch_data=np.array_split(self.X,self.n_workers)
        self.batch_targets=np.array_split(self.Y,self.n_workers)

        # CSV files
        self.training_csv=f"training_{workers}_workers.csv"
        self.worker_csv=f"worker_times_{workers}_workers.csv"

        with open(self.training_csv,"w",newline="") as f:
            writer=csv.writer(f)
            writer.writerow(["workers","epoch","accuracy","wall_time"])

        with open(self.worker_csv,"w",newline="") as f:
            writer=csv.writer(f)
            writer.writerow(["workers","epoch","worker_id","train_time"])

    def accuracy(self):

        z1=self.X.dot(self.ws1.T)+self.bs1
        a1=np.maximum(0,z1)

        z2=a1.dot(self.ws2.T)+self.bs2
        pred=np.argmax(z2,axis=1)

        return np.mean(pred==self.y_raw)*100

    def start(self):

        server=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)

        server.bind((self.host,self.port))
        server.listen()

        print("Waiting workers...")

        while len(self.workers)<self.n_workers:

            conn,addr=server.accept()
            worker_id=len(self.workers)

            print("Worker connected",addr)

            send_arrays(conn,{
                "batch_data":self.batch_data,
                "batch_targets":self.batch_targets
            })

            send_json(conn,{
                "worker_id":worker_id,
                "lr":self.lr
            })

            self.workers.append(conn)

        print("All workers ready")

        start=time.time()

        for epoch in range(self.epochs):

            for wid,conn in enumerate(self.workers):

                send_json(conn,{
                    "type":"train",
                    "epoch":epoch,
                    "measure":epoch%60==0
                })

                send_arrays(conn,(self.ws1,self.bs1,self.ws2,self.bs2))

            grads=[]
            worker_times=[]

            for wid,conn in enumerate(self.workers):

                msg=recv_json(conn)

                if msg["type"]=="grad":

                    grads.extend(recv_arrays(conn))

                    if msg["measure"]:
                        worker_times.append((wid,msg["time"]))

            avg=[sum(x)/len(grads) for x in zip(*grads)]

            self.ws1-=self.lr*avg[0]
            self.bs1-=self.lr*avg[1]
            self.ws2-=self.lr*avg[2]
            self.bs2-=self.lr*avg[3]

            if epoch%50==0:

                acc=self.accuracy()
                wall=time.time()-start

                with open(self.training_csv,"a",newline="") as f:
                    csv.writer(f).writerow(
                        [self.n_workers,epoch,acc,wall]
                    )

                print("Epoch",epoch,"Acc",acc)

            for wid,t in worker_times:

                with open(self.worker_csv,"a",newline="") as f:
                    csv.writer(f).writerow(
                        [self.n_workers,epoch,wid,t]
                    )

        total=time.time()-start
        acc=self.accuracy()

        print("Training finished")

        with open("experiment_summary.csv","a",newline="") as f:

            csv.writer(f).writerow(
                [self.n_workers,total,acc]
            )

        for conn in self.workers:

            send_json(conn,{"type":"done"})
            conn.close()


if __name__=="__main__":

    parser=argparse.ArgumentParser()

    parser.add_argument("--host",default="0.0.0.0")
    parser.add_argument("--port",type=int,default=5000)

    parser.add_argument("--workers",type=int,default=4)
    parser.add_argument("--epochs",type=int,default=600)

    parser.add_argument("--lr",type=float,default=0.05)
    parser.add_argument("--hidden",type=int,default=50)
    parser.add_argument("--train_size",type=int,default=60000)

    args=parser.parse_args()

    ps=ParameterServer(
        args.host,
        args.port,
        args.epochs,
        args.workers,
        args.lr,
        args.hidden,
        args.train_size
    )

    ps.start()