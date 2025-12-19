"""
This module was created to reproduce the issue faced in `DEV-49-critical-prevent-ai-enhancement-deadlock-nested-semaphore-acquisition`.
Then after fixing the issue, we check via the test whether the issue will still happen

"""


import asyncio
import types
import pytest

from devdox_ai_locust.hybrid_loctus_generator import HybridLocustGenerator

class _FakeMessage:
    """
    `HybridLocustGenerator` expects an object like `AsyncTogether` that can do:
    `self.ai_client.chat.completions.create(...)` and returns a response shaped like:
    `response.choices[0].message.content`
    
    This class mimics `response.choices[0].message.content`:
    - Represents the `message` object inside a completion choice.
    - Only contains `.content`.
    """
    def __init__(self, content: str):
        self.content = content


class _FakeChoice:
    """
    `HybridLocustGenerator` expects an object like `AsyncTogether` that can do:
    `self.ai_client.chat.completions.create(...)` and returns a response shaped like:
    `response.choices[0].message.content`

    This class mimics `response.choices[0].message`:
        - Represents one “choice” returned by the AI.
        - Stores a message.
    """
    def __init__(self, content: str):
        self.message = _FakeMessage(content)


class _FakeResponse:
    """
    `HybridLocustGenerator` expects an object like `AsyncTogether` that can do:
    `self.ai_client.chat.completions.create(...)` and returns a response shaped like:
    `response.choices[0].message.content`

    This class mimics `response.choices`:
    - Represents the full response object.
    - Has `.choices` list, with one choice.
    """
    def __init__(self, content: str):
        self.choices = [_FakeChoice(content)]


class FakeAsyncTogether:
    
    """
        This is a fake replacement for Together’s real client.
        
    """
    
    class _Completions:
        def create(self, **kwargs):
            """
                In the real Together client, create(...) returns something awaitable.
                Here, we make it return an async coroutine.
            """
            async def _coro():
                """When awaited, it returns a fake response containing <code>print('ok')</code>."""
                # Return something that the generator can parse.
                return _FakeResponse("<code>print('ok')</code>")
            
            return _coro()
    
    class _Chat:
        """
        Mimics `client.chat.completions`
        """
        def __init__(self):
            self.completions = FakeAsyncTogether._Completions()
    
    def __init__(self):
        """the generator can call: ai_client.chat.completions.create(...) normally."""
        self.chat = FakeAsyncTogether._Chat()


class TestAiDeadlockNestedSemaphore:
    @pytest.mark.asyncio
    # @pytest.mark.xfail(
    #     reason="Known deadlock before fix: nested semaphore acquisition in _call_ai_service -> _make_api_call",
    #     strict=True,
    # )
    async def test_ai_calls_can_deadlock_due_to_nested_semaphore_acquisition(self):
        """
        EXPECTED (after fix):
          - Both concurrent calls complete quickly.
          - Each returns cleaned Python code text.
    
        CURRENT (before fix):
          - This test will HANG unless we wrap it in a timeout.
          - The hang is a deadlock:
              * each task acquires the semaphore once in _call_ai_service
              * then each task tries to acquire it again in _make_api_call
              * since all permits are already held, nobody can proceed
              * no exception is raised from your code, it just waits forever till operations time out
        """
        # ===================================================
        # ARRANGE
        # ===================================================
        
        # create the generator using the fake AI client so no real network calls happen
        gen = HybridLocustGenerator(ai_client=FakeAsyncTogether())
    
        # Make the deadlock easier to trigger deterministically with 2 concurrent tasks.
        # We set semaphore capacity to 2 because we will run 2 concurrent calls.
        # Deadlock happens when concurrency "fills the semaphore" at the outer level.
        # In production code it’s Semaphore(5), the same deadlock occurs when you hit 5 concurrent calls.
        gen._api_semaphore = asyncio.Semaphore(2)
    
        # Barrier to force both tasks to:
        #   1) acquire the outer semaphore in _call_ai_service
        #   2) arrive at _make_api_call
        #   3) only then proceed together into the inner acquisition attempt
        barrier = asyncio.Event()
        arrived_lock = asyncio.Lock()
        arrived = 0
    
        original_make_api_call = gen._make_api_call
    
        async def patched_make_api_call(self, messages):
            nonlocal arrived
            async with arrived_lock:
                arrived += 1
                if arrived == 2:
                    barrier.set()
    
            # Both tasks pause here, *while still holding the outer semaphore permit*.
            # Once they are released, they will both attempt the inner "async with semaphore"
            # inside original _make_api_call, when no permits remain -> deadlock.
            await barrier.wait()
            return await original_make_api_call(messages)
    
        gen._make_api_call = types.MethodType(patched_make_api_call, gen)
    
        async def run_two_calls():
            results = await asyncio.gather(
                gen._call_ai_service("prompt-1"),
                gen._call_ai_service("prompt-2"),
            )
            # After the fix, it is expected that both with return code (not hang).
            assert len(results) == 2
            assert all("print('ok')" in r or 'print("ok")' in r for r in results)

        # ===================================================
        # ACT & ASSERT
        # ===================================================
        
        # If deadlock happens, this timeout fires and the test fails (xfail for now).
        await asyncio.wait_for(run_two_calls(), timeout=0.5)
