import asyncio
import websockets

async def test():
    try:
        async with websockets.connect("ws://127.0.0.1:5000") as ws:
            print("Connected! Waiting for message...")
            msg = await asyncio.wait_for(ws.recv(), timeout=5.0)
            print("Received:", msg[:100], "...")
    except Exception as e:
        print("Error:", e)

asyncio.run(test())
