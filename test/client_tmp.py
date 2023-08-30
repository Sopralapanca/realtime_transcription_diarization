
import socket

def start_client():
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_host = 'dbalboni.cc'  # Replace with the IP address of the remote server
    server_port = 10486              # Use the same port number as the server

    client_socket.connect((server_host, server_port))

    message = "Hello, server!"
    client_socket.sendall(message.encode('utf-8'))

    response = client_socket.recv(1024).decode('utf-8')
    print("Received response from server: ", response)

    client_socket.close()

if __name__ == "__main__":
    start_client()
