import socket

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind(('localhost', 12345))
server_socket.listen(1)

print("Waiting for a connection...")
client_socket, client_address = server_socket.accept()
print("Connected to", client_address)

while True:
    data = client_socket.recv(1024).decode('utf-8')
    if not data:
        break
    print("Received:", data)

client_socket.close()
server_socket.close()