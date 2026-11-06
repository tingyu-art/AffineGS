# **** event_loarder.py **************
import h5py
import numpy as np
from generator.images import Sequence2events, config_events
# import cv2
# import scipy.special as ss
# from scipy.integrate import cumtrapz
# import copy
# from addict import Dict
import torch


class EventManager:
    def __init__(self, source_type, path=None, cfg=None, flow_func=None, timestamp_origin_us=0):
        """
        source_type: 'simulator', 'real', 'evk5', 'npz'
        path: 文件路径（模拟器无需）
        cfg: 模拟器配置
        flow: 模拟器使用的速度场函数
        """
        self.source_type = source_type.lower()
        self.path = path
        self.cfg = cfg
        self.flow_func = flow_func
        self.timestamp_origin_us = timestamp_origin_us
        self.evts = None
        self.flow = None

        self.source = self._create_source()

    def _create_source(self):
        if self.source_type == "simulator":
            if self.flow_func is None:
                raise ValueError("模拟器模式必须提供 flow 函数")
            return Sequence2events(flow_func=self.flow_func, cfg=self.cfg)
        elif self.source_type == "real":
            return RealEventFileSource(self.path, timestamp_origin_us=self.timestamp_origin_us)
        elif self.source_type == "evk5":
            return EVK5EventFileSource(self.path, timestamp_origin_us=self.timestamp_origin_us)
        elif self.source_type == "npz":
            return NpzEventFileSource(self.path)
        else:
            raise ValueError(f"Unsupported source_type: {self.source_type}")

    def load(self):
        self.evts, self.flow = self.source.load()
        return self.evts, self.flow


class EventDataSource:
    def __call__(self, cfg=None):
        return self.load()
        
    def load(self):
        raise NotImplementedError


def _recorded_events(rows, columns, timestamps_us, polarity, origin_us=0):
    """Preserve timestamp precision until windowing/rebasing, then callers cast.

    The default preserves absolute milliseconds, as before, using float64.
    For long recordings pass origin_us='start' or an explicit integer origin;
    integer timestamps are rebased before any floating-point conversion.
    """
    raw = np.asarray(timestamps_us)
    if raw.ndim != 1:
        raise ValueError("recorded timestamps must be one-dimensional")
    if origin_us == 'start':
        origin_us = int(raw.min()) if raw.size and raw.dtype.kind in 'iu' else (float(raw.min()) if raw.size else 0)
    if not isinstance(origin_us, (int, float, np.integer, np.floating)) or not np.isfinite(origin_us):
        raise ValueError("timestamp_origin_us must be finite or 'start'")
    if raw.dtype.kind in 'iu':
        if raw.size and (raw.max() > np.iinfo(np.int64).max or raw.min() < 0):
            raise ValueError("recorded integer timestamps must fit nonnegative int64")
        if float(origin_us).is_integer() and 0 <= origin_us <= np.iinfo(np.int64).max:
            shifted = (raw.astype(np.int64) - int(origin_us)).astype(np.float64)
        else:
            shifted = raw.astype(np.float64) - float(origin_us)
    else:
        shifted = raw.astype(np.float64) - float(origin_us)
    time_ms = shifted / 1000.0
    # Preserve the existing float32-ms values for short integer timestamps that
    # were represented exactly before unit conversion. Large/rebased timestamps
    # use the high-precision path, preventing loss before subtracting the origin.
    if origin_us == 0 and raw.dtype.kind in 'iu':
        safe = raw <= 2**24
        time_ms[safe] = (raw[safe].astype(np.float32) / np.float32(1000.0)).astype(np.float64)
    arrays = [np.asarray(rows), np.asarray(columns), time_ms, np.asarray(polarity)]
    if any(a.ndim != 1 or len(a) != len(raw) for a in arrays):
        raise ValueError("recorded event columns must be one-dimensional with equal lengths")
    events = np.column_stack(arrays).astype(np.float64, copy=False)
    if not np.isfinite(events).all():
        raise ValueError("recorded events contain non-finite values")
    return events, origin_us


class RealEventFileSource(EventDataSource):
    def __init__(self, file_path, timestamp_origin_us=0):
        self.file_path = file_path
        self.timestamp_origin_us = timestamp_origin_us

    def load(self):
        with h5py.File(self.file_path, 'r') as h5f:
            evts, self.resolved_timestamp_origin_us = _recorded_events(
                h5f["events/y"],
                h5f["events/x"],
                h5f["events/t"],
                h5f["events/p"],
                origin_us=self.timestamp_origin_us,
            )
        return evts, (None,None)


class EVK5EventFileSource(EventDataSource):
    def __init__(self, file_path, timestamp_origin_us=0):
        self.file_path = file_path
        self.timestamp_origin_us = timestamp_origin_us

    def load(self):
        with h5py.File(self.file_path, 'r') as h5f:
            evts, self.resolved_timestamp_origin_us = _recorded_events(
                h5f["CD/events"]["y"],
                h5f["CD/events"]["x"],
                h5f["CD/events"]["t"],
                h5f["CD/events"]["p"],
                origin_us=self.timestamp_origin_us,
            )
        return evts, (None,None)


class NpzEventFileSource(EventDataSource):
    def __init__(self, file_path):
        self.file_path = file_path

    def load(self):
        evts = np.load(self.file_path)['evts']
        return evts, (None,None)

