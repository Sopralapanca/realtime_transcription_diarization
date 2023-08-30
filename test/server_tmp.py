import socket

print("server host: ", socket.gethostname())

def get_local_ip():
    try:
        local_ip = socket.gethostbyname(socket.gethostname())
        return local_ip
    except socket.gaierror:
        return "Could not retrieve local IP"

def get_public_ip():
    try:
        public_ip = socket.gethostbyname(socket.getfqdn())
        return public_ip
    except socket.gaierror:
        return "Could not retrieve public IP"
def start_server():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_host = '0.0.0.0'  # Listen on all available interfaces
    server_port = 8080     # Choose a suitable port number

    server_socket.bind((server_host, server_port))
    server_socket.listen(5)
    print("Server listening on {}:{}".format(server_host, server_port))


    local_ip = get_local_ip()
    public_ip = get_public_ip()

    print("Server listening on {}:{}".format(server_host, server_port))
    print("Local IP address: {}".format(local_ip))
    print("Public IP address: {}".format(public_ip))

    while True:
        client_socket, client_address = server_socket.accept()
        print("Connection established with: ", client_address)

        data = client_socket.recv(1024).decode('utf-8')
        print("Received data from client: ", data)

        # Process the data (in this example, we just echo it back)
        response = "Server says: " + data

        client_socket.sendall(response.encode('utf-8'))
        client_socket.close()

if __name__ == "__main__":
    start_server()
