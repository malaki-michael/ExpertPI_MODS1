# GENERATED FILE from pySTEM 1.16.1,build 4717,rev b31140633df20050ad44d1030889522d4f5f8256
import grpc

channel = None


def connect(host: str):
    global channel
    channel = grpc.insecure_channel(host)
