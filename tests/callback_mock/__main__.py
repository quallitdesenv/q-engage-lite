#!/usr/bin/env python3
"""
Callback Mock Server
Simple HTTP server to receive and store tracker detection callbacks
"""
from http.server import HTTPServer, BaseHTTPRequestHandler
import json
from datetime import datetime
from pathlib import Path

class CallbackHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        """Handle POST requests and save payload to results folder"""
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length)
        
        try:
            # Parse JSON
            data = json.loads(body.decode('utf-8'))
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            filename = f"detections_{timestamp}.json"
            filepath = Path(__file__).parent / 'results' / filename
            
            # Save to file
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            
            print(f"✓ Saved payload to {filename}")
            print(f"  Detections count: {len(data.get('detections', []))}")
            
            # Send success response
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            response = {'status': 'success', 'message': 'Data received'}
            self.wfile.write(json.dumps(response).encode())
            
        except Exception as e:
            print(f"✗ Error processing request: {e}")
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            response = {'status': 'error', 'message': str(e)}
            self.wfile.write(json.dumps(response).encode())
    
    def log_message(self, format, *args):
        """Suppress default logging"""
        pass

def main():
    port = 8080
    server_address = ('', port)
    httpd = HTTPServer(server_address, CallbackHandler)
    
    print(f"Callback Mock Server")
    print(f"Listening on http://localhost:{port}")
    print(f"Saving results to: ./results/")
    print("Press Ctrl+C to stop\n")
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n✓ Server stopped")

if __name__ == "__main__":
    main()
