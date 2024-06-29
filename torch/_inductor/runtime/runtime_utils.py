# mypy: allow-untyped-defs
from __future__ import annotations

import functools
import getpass
import inspect
import operator
import os
import re
import tempfile
import time

import torch


def conditional_product(*args):
    return functools.reduce(operator.mul, [x for x in args if x])


def ceildiv(numer: int, denom: int) -> int:
    return -(numer // -denom)


def is_power_of_2(n: int) -> bool:
    """Returns whether n = 2 ** m for some integer m."""
    return n > 0 and n & n - 1 == 0


def next_power_of_2(n: int) -> int:
    """Return the smallest power of 2 greater than or equal to n"""
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    n |= n >> 32
    n += 1
    return n


def get_num_bytes(*args: torch.Tensor, num_in_out_args: int = 0) -> int:
    """
    Return the total number of bytes the arguments of tensor type takes.

    For in/out args, tensor sizes are counted twice: once for reading and
    once for writing.

    The first num_in_out_args arguments are in out tensors.
    """
    return sum(
        arg.numel() * arg.element_size() * (1 + int(i < num_in_out_args))
        for i, arg in enumerate(args)
        if isinstance(arg, torch.Tensor)
    )


def triton_config_to_hashable(cfg):
    """
    Convert triton config to a tuple that can uniquely identify it. We can use
    the return value as a dictionary key.
    """
    items = sorted(cfg.kwargs.items())
    items.append(("num_warps", cfg.num_warps))
    items.append(("num_stages", cfg.num_stages))
    return tuple(items)


def create_bandwidth_info_str(ms, num_gb, gb_per_s, prefix="", suffix="", color=True):
    info_str = f"{prefix}{ms:.3f}ms    \t{num_gb:.3f} GB \t {gb_per_s:7.2f}GB/s{suffix}"
    slow = ms > 0.012 and gb_per_s < 650
    return red_text(info_str) if color and slow else info_str


def get_max_y_grid():
    return 65535


def do_bench(fn, fn_args, fn_kwargs, **kwargs):
    from torch._inductor.utils import is_cpu_device

    args = list(fn_args)
    args.extend(fn_kwargs.values())
    if is_cpu_device(args):
        return do_bench_cpu(lambda: fn(*fn_args, **fn_kwargs), **kwargs)
    else:
        return do_bench_gpu(lambda: fn(*fn_args, **fn_kwargs), **kwargs)


def do_bench_gpu(fn_or_fns, estimation_iters=5, memory_warmup_iters=100, benchmark_iters=100, max_benchmark_duration=25, testing=False):
    @functools.lru_cache(None)
    def get_cache_size():
        device = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(device)
        return properties.l2CacheSize

    def get_event_pairs(iters):
        return [
            (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            for _ in range(iters)
        ]
    
    def get_interleaved_event_pairs(fns, iters):
        return [get_event_pairs(len(fns)) for _ in range(iters)]
    
    def get_timing(event_pairs):
        return min([start_event.elapsed_time(end_event) for start_event, end_event in event_pairs])
    
    def get_interleaved_timing(interleaved_event_pairs):
        return [get_timing(event_pairs) for event_pairs in zip(*interleaved_event_pairs)]
    
    def memory_warmup(buffer, iters):
        for _ in range(iters):
            buffer.zero_()
    
    def benchmark(fn, buffer, iters):
        event_pairs = get_event_pairs(iters)
        for start_event, end_event in event_pairs:
            buffer.zero_()
            start_event.record()
            fn()
            end_event.record()
        torch.cuda.synchronize()
        return get_timing(event_pairs)
    
    def benchmark_interleaved(fns, buffer, iters):
        interleaved_event_pairs = [get_event_pairs(len(fns)) for _ in range(iters)]
        for event_pairs in interleaved_event_pairs:
            for fn, (start_event, end_event) in zip(fns, event_pairs):
                buffer.zero_()
                start_event.record()
                fn()
                end_event.record()
        torch.cuda.synchronize()
        return get_interleaved_timing(interleaved_event_pairs)
    
    def do_noninterleaved(fn, buffer, memory_warmup_iters, benchmark_iters):
        memory_warmup(buffer, memory_warmup_iters)
        timing = benchmark(fn, buffer, benchmark_iters)
        return timing

    def do_interleaved(fns, buffer, memory_warmup_iters, benchmark_iters):
        memory_warmup(buffer, memory_warmup_iters)
        timings = benchmark_interleaved(fns, buffer, benchmark_iters)
        return timings

    buffer = torch.empty(int(get_cache_size() // 4), dtype=torch.int, device="cuda")

    interleaved = isinstance(fn_or_fns, list)
    if interleaved:
        fns = fn_or_fns

        estimation_timings = benchmark_interleaved(fns, buffer, estimation_iters)
        benchmark_iters = min(benchmark_iters, max(int(max_benchmark_duration / max(estimation_timings)), 1))

        groups = 5
        iters_per_group = max(benchmark_iters // groups, 1)
        
        grouped_timings = []
        for _ in range(groups):
            memory_warmup(buffer, memory_warmup_iters)
            timings = benchmark_interleaved(fns, buffer, iters_per_group)
            grouped_timings.append(timings)
        
        del buffer
        
        timings = [min(ungrouped_timings) for ungrouped_timings in zip(*grouped_timings)]
       
        return timings
    else:
        fn = fn_or_fns

        estimation_timing = benchmark(fn, buffer, estimation_iters)
        benchmark_iters = min(benchmark_iters, max(int(max_benchmark_duration / estimation_timing), 1))

        groups = 5
        iters_per_group = max(benchmark_iters // groups, 1)

        grouped_timings = []
        for _ in range(groups):
            memory_warmup(buffer, memory_warmup_iters)
            timings = benchmark(fn, buffer, iters_per_group)
            grouped_timings.append(timings)

        del buffer

        timing = min(grouped_timings)

        return timing


def do_bench_cpu(fn, warmup=5, times=20):
    assert times > 0
    for _ in range(warmup):
        fn()
    durations = []
    for _ in range(times):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        durations.append((t1 - t0) * 1000)
    # return the median time
    sorted_durations = sorted(durations)
    if times % 2 == 0:
        return (sorted_durations[times // 2 - 1] + sorted_durations[times // 2]) / 2
    else:
        return sorted_durations[times // 2]


def cache_dir() -> str:
    cache_dir = os.environ.get("TORCHINDUCTOR_CACHE_DIR")
    if cache_dir is None:
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = cache_dir = default_cache_dir()
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def default_cache_dir():
    sanitized_username = re.sub(r'[\\/:*?"<>|]', "_", getpass.getuser())
    return os.path.join(
        tempfile.gettempdir(),
        "torchinductor_" + sanitized_username,
    )


HAS_COLORAMA = True
try:
    import colorama
except ImportError:
    HAS_COLORAMA = False


def _color_text(msg, color):
    if not HAS_COLORAMA:
        return msg

    return getattr(colorama.Fore, color.upper()) + msg + colorama.Fore.RESET


def green_text(msg):
    return _color_text(msg, "green")


def yellow_text(msg):
    return _color_text(msg, "yellow")


def red_text(msg):
    return _color_text(msg, "red")


def blue_text(msg):
    return _color_text(msg, "blue")


def get_first_attr(obj, *attrs):
    """
    Return the first available attribute or throw an exception if none is present.
    """
    for attr in attrs:
        if hasattr(obj, attr):
            return getattr(obj, attr)

    raise AssertionError(f"{obj} does not has any of the attributes: {attrs}")


try:
    dynamo_timed = torch._dynamo.utils.dynamo_timed
except AttributeError:  # Compile workers only have a mock version of torch

    def dynamo_timed(original_function=None, phase_name=None, fwd_only=True):
        if original_function:
            return original_function
        return dynamo_timed
