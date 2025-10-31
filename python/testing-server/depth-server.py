import asyncio
import websockets
import logging
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
HOST = '0.0.0.0'  # Listen on all available network interfaces
PORT = 8765       # The port the server will listen on
IMAGE_DIM = 256
PIXEL_COUNT = IMAGE_DIM * IMAGE_DIM
MAX_MSG_SIZE = 5 * 1024 * 1024  # 5 MB limit

# A set to keep track of all connected clients (browsers, phones, etc.)
connected_clients = set()

async def handle_connection(websocket):
    """Handles connections and broadcasts valid depth frames."""
    connected_clients.add(websocket)
    logging.info(f"Client connected: {websocket.remote_address} (Total clients: {len(connected_clients)})")
    
    try:
        async for message in websocket:
            # We parse the message here to check if it's a valid depth frame
            try:
                data = json.loads(message)
                
                # Check if it looks like our depth data
                if isinstance(data, list) and len(data) == PIXEL_COUNT:
                    logging.info(f"Received valid {IMAGE_DIM}x{IMAGE_DIM} frame. Broadcasting to {len(connected_clients) - 1} other clients.")
                    
                    # Create a list of tasks to send to all *other* clients
                    broadcast_tasks = []
                    for client in connected_clients:
                        if client != websocket: # Don't send back to the sender (the phone)
                            broadcast_tasks.append(client.send(message))
                    
                    # Run all send tasks concurrently
                    if broadcast_tasks:
                        await asyncio.gather(*broadcast_tasks)
                
                else:
                    logging.warning(f"Received non-frame message or bad data length: {len(data)}. Ignoring.")
            
            except json.JSONDecodeError:
                logging.warning("Received a message that was not valid JSON. Ignoring.")
            except Exception as e:
                logging.error(f"Error processing message: {e}")

    except websockets.exceptions.ConnectionClosedOK:
        logging.info(f"Client disconnected gracefully: {websocket.remote_address}")
    except websockets.exceptions.ConnectionClosedError as e:
        logging.error(f"Client connection closed with error: {websocket.remote_address} - {e}")
    finally:
        # Remove the client from the set upon disconnection
        connected_clients.remove(websocket)
        logging.info(f"Client removed: {websocket.remote_address} (Total clients: {len(connected_clients)})")

async def main():
    """Starts the WebSocket server."""
    logging.info(f"Starting WebSocket broadcast server on ws://{HOST}:{PORT}")
    async with websockets.serve(handle_connection, HOST, PORT, max_size=MAX_MSG_SIZE):
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Server stopped manually.")