// Create Event
socket = network_create_socket(network_socket_tcp);

// Connect to Python server
network_connect(socket, "localhost", 12345);

// Send data to Python server
network_send(socket, "Hello from GameMaker!");

// Clean up
network_destroy(socket);