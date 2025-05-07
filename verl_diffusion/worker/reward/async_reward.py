import threading
import queue
import torch

class AsyncRewardWorker:
    def __init__(self, reward_calculator, config=None):
        """
        Initialize an asynchronous reward worker that processes molecules on CPU
        
        Args:
            reward_calculator: Reward calculator instance that has calculate_rewards method
            config: Configuration for the worker
        """
        self.reward_calculator = reward_calculator
        self.config = config or {}
        self.input_queue = queue.Queue(maxsize=self.config.get("queue_size", 5))
        self.output_queue = queue.Queue(maxsize=self.config.get("queue_size", 5))
        self.running = False
        self.thread = None
        
    def start(self):
        """Start the asynchronous reward calculation thread"""
        if self.thread is not None and self.thread.is_alive():
            raise RuntimeError("Async reward worker already running")
            
        self.running = True
        self.thread = threading.Thread(target=self._reward_loop)
        self.thread.daemon = True  # Thread will exit when main program exits
        self.thread.start()
        return self.input_queue, self.output_queue
    
    def _reward_loop(self):
        """Internal method that runs in a separate thread to calculate rewards"""
        try:
            while self.running:
                # Get a batch from the input queue, block until available
                batch = self.input_queue.get()
                
                # None signals end of processing
                if batch is None:
                    self.output_queue.put(None)  # Signal downstream that we're done
                    break
                
                # Check if the batch is an exception
                if isinstance(batch, Exception):
                    self.output_queue.put(batch)  # Forward the exception
                    continue
                
                # Transfer data to CPU if it's on GPU
                cpu_batch = {}
                for key, value in batch.items():
                    if isinstance(value, torch.Tensor):
                        cpu_batch[key] = value.cpu()
                    else:
                        cpu_batch[key] = value
                
                # Calculate rewards
                reward = self.reward_calculator.calculate_rewards(cpu_batch)
                
                # Put the result in the output queue
                self.output_queue.put((cpu_batch, reward))
                
                # Mark task as done
                self.input_queue.task_done()
                
        except Exception as e:
            # Put the exception in the output queue
            self.output_queue.put(e)
        finally:
            # Signal that we're done
            self.output_queue.put(None)
    
    def stop(self):
        """Stop the asynchronous reward calculation thread"""
        self.running = False
        # Put None in the input queue to signal the thread to exit
        self.input_queue.put(None)
        if self.thread is not None:
            self.thread.join(timeout=10)  # Wait for thread to finish
            self.thread = None 