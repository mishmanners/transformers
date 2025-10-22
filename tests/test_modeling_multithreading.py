# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Tests for multithreaded model loading to ensure thread safety.
"""

import os
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor

import torch
from safetensors.torch import save_file

from transformers.modeling_utils import load_state_dict
from transformers.testing_utils import require_torch


@require_torch
class TestMultithreadedLoading(unittest.TestCase):
    """Test that model loading is thread-safe when using safetensors."""

    def test_concurrent_safetensors_loading(self):
        """
        Test that multiple threads can safely load the same safetensors file.
        This test verifies the fix for: https://github.com/huggingface/transformers/issues/XXXXX
        where concurrent calls to safe_open could cause race conditions.
        """
        # Create a test safetensors file
        temp_dir = tempfile.mkdtemp()
        test_file = os.path.join(temp_dir, "model.safetensors")

        # Create a simple state dict
        state_dict = {
            "weight1": torch.randn(10, 10),
            "weight2": torch.randn(5, 5),
            "bias": torch.randn(10),
        }

        save_file(state_dict, test_file)

        try:
            def load_in_thread(thread_id):
                """Load the state dict from a thread"""
                try:
                    # Load with map_location="meta" which triggers the safetensors code path
                    loaded_state = load_state_dict(test_file, map_location="meta", weights_only=True)
                    # Verify we loaded the correct number of tensors
                    self.assertEqual(len(loaded_state), len(state_dict))
                    # Verify all keys are present
                    self.assertEqual(set(loaded_state.keys()), set(state_dict.keys()))
                    return True
                except Exception as e:
                    print(f"Thread {thread_id} failed with error: {e}")
                    return False

            # Test with multiple threads loading the same file
            num_threads = 5
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = [executor.submit(load_in_thread, i) for i in range(num_threads)]
                results = [future.result() for future in futures]

            # All threads should succeed
            self.assertTrue(all(results), f"Some threads failed to load: {results}")

        finally:
            # Cleanup
            if os.path.exists(test_file):
                os.remove(test_file)
            os.rmdir(temp_dir)

    def test_concurrent_safetensors_loading_cpu_device(self):
        """
        Test that multiple threads can safely load the same safetensors file with cpu device.
        """
        # Create a test safetensors file
        temp_dir = tempfile.mkdtemp()
        test_file = os.path.join(temp_dir, "model.safetensors")

        # Create a simple state dict
        state_dict = {
            "weight1": torch.randn(10, 10),
            "weight2": torch.randn(5, 5),
            "bias": torch.randn(10),
        }

        save_file(state_dict, test_file)

        try:
            def load_in_thread(thread_id):
                """Load the state dict from a thread"""
                try:
                    # Load with map_location="cpu" 
                    loaded_state = load_state_dict(test_file, map_location="cpu", weights_only=True)
                    # Verify we loaded the correct number of tensors
                    self.assertEqual(len(loaded_state), len(state_dict))
                    # Verify all keys are present
                    self.assertEqual(set(loaded_state.keys()), set(state_dict.keys()))
                    # Verify tensors are on CPU
                    for tensor in loaded_state.values():
                        self.assertEqual(tensor.device.type, "cpu")
                    return True
                except Exception as e:
                    print(f"Thread {thread_id} failed with error: {e}")
                    return False

            # Test with multiple threads loading the same file
            num_threads = 5
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = [executor.submit(load_in_thread, i) for i in range(num_threads)]
                results = [future.result() for future in futures]

            # All threads should succeed
            self.assertTrue(all(results), f"Some threads failed to load: {results}")

        finally:
            # Cleanup
            if os.path.exists(test_file):
                os.remove(test_file)
            os.rmdir(temp_dir)
