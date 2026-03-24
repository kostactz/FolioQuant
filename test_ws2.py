import asyncio
import websockets

async def test():
    try:
        async with websockets.connect('ws://localhost:5000') as ws:
            print("Connected")
            msg1 = await ws.recv()
            print("Message 1 received:", len(msg1), "bytes")
            msg2 = await ws.recv()
            print("Message 2 received:", len(msg2), "bytes")
            await asyncio.sleep(5)
            print("Still connected!")
    except Exception as e:
        print("Error:", type(e).__name__, e)

asyncio.run(test())
