from fastapi import FastAPI, Request, Form
from fastapi.responses import StreamingResponse, HTMLResponse
import asyncio

app = FastAPI()

# Store connected clients
clients = []

# SSE stream generator
async def event_generator(request: Request):
    queue = asyncio.Queue()
    clients.append(queue)

    try:
        while True:
            if await request.is_disconnected():
                break

            # Wait for new message
            message = await queue.get()
            yield f"data: {message}\n\n n:atc"
    finally:
        clients.remove(queue)

@app.get("/events")
async def sse(request: Request):
    return StreamingResponse(event_generator(request), media_type="text/event-stream")

# Endpoint to send message
@app.post("/send")
async def send_message(message: str = Form(...)):
    for queue in clients:
        await queue.put(message)
    return {"status": "message sent", "message": message}

# Simple HTML client
@app.get("/", response_class=HTMLResponse)
async def chat_ui():
    return """
    <!DOCTYPE html>
    <html>
    <head><title>SSE Chat</title></head>
    <body>
      <h2>🔴 Real-time Chat Notifications</h2>
      <form method="POST" action="/send">
        <input type="text" name="message" placeholder="Type message..." required>
        <button type="submit">Send</button>
      </form>
      <h3>Messages:</h3>
      <div id="messages"></div>

      <script>
        const eventSource = new EventSource("/events");
        eventSource.onmessage = function(event) {
          const msgDiv = document.getElementById("messages");
          msgDiv.innerHTML += "<p>" + event.data+event.n  + "</p>";
        };
      </script>
    </body>
    </html>
    """
