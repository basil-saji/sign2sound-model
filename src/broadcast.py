import asyncio
import threading
import uuid

from supabase import create_async_client


class Broadcaster:
    def __init__(self, url: str, key: str):
        self.session_id = "OFFLINE"
        self.enabled = False
        self.channel = None

        self._loop = asyncio.new_event_loop()
        threading.Thread(
            target=self._run_loop,
            daemon=True
        ).start()

        asyncio.run_coroutine_threadsafe(
            self._init_async(url, key),
            self._loop
        )

    def _run_loop(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    async def _init_async(self, url: str, key: str):
        try:
            self.client = await create_async_client(url, key)

            self.session_id = uuid.uuid4().hex[:6].upper()
            self.channel = self.client.channel(f"room_{self.session_id}")

            await self.channel.subscribe()

            self.enabled = True
            print(f"Broadcast active on room_{self.session_id}")

        except Exception as e:
            print(f"Broadcast async init failed: {e}")
            self.enabled = False
            self.channel = None

    def send(self, payload: dict):
        """
        payload examples:
        { "type": "word", "text": "HELLO" }
        { "type": "sentence_end" }
        """
        if not self.enabled or self.channel is None:
            return

        async def _send():
            try:
                await self.channel.send_broadcast("translation", payload)
            except Exception as e:
                print(f"Broadcast send error: {e}")

        asyncio.run_coroutine_threadsafe(_send(), self._loop)
