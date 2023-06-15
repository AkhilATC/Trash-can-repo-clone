import socket

UDP_IP = "0.0.0.0"
UDP_PORT = 5959
MESSAGE = "This is a test"
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) # UDP
sock.sendto(bytes(MESSAGE, "utf-8"), (UDP_IP, UDP_PORT))