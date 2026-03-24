import asyncio
import websockets
import json

async def test():
    try:
        async with websockets.connect("ws://127.0.0.1:5000") as ws:
            print("Connected!")
            msg = await asyncio.wait_for(ws.recv(), timeout=5.0)
            data = json.loads(msg)
            print("fast metrics length:", len(data.get("fast", [])))
            print("fast[0]:", str(data.get("fast", [])[0])[:200])
    except Exception as e:
        print("Error:", e)

asyncio.run(test())
