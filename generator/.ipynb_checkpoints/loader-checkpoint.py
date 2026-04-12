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
    def __init__(self, source_type, path=None, cfg=None, flow_func=None):
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
        self.evts = None
        self.flow = None

        self.source = self._create_source()

    def _create_source(self):
        if self.source_type == "simulator":
            if self.flow_func is None:
                raise ValueError("模拟器模式必须提供 flow 函数")
            return Sequence2events(flow_func=self.flow_func, cfg=self.cfg)
        elif self.source_type == "real":
            return RealEventFileSource(self.path)
        elif self.source_type == "evk5":
            return EVK5EventFileSource(self.path)
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


class RealEventFileSource(EventDataSource):
    def __init__(self, file_path):
        self.file_path = file_path

    def load(self):
        with h5py.File(self.file_path, 'r') as h5f:
            evts = np.transpose(np.array([
                h5f["events/y"],
                h5f["events/x"],
                h5f["events/t"],
                h5f["events/p"]
            ]))
        evts= np.float32(evts) 
        evts[:,2] = np.float32(evts[:,2])/1000.0 
        return evts, (None,None)


class EVK5EventFileSource(EventDataSource):
    def __init__(self, file_path):
        self.file_path = file_path

    def load(self):
        with h5py.File(self.file_path, 'r') as h5f:
            evts = np.transpose(np.array([
                h5f["CD/events"]["y"],
                h5f["CD/events"]["x"],
                h5f["CD/events"]["t"],
                h5f["CD/events"]["p"]
            ]))
        evts= np.float32(evts) 
        evts[:,2] = np.float32(evts[:,2])/1000.0 
        return evts, (None,None)


class NpzEventFileSource(EventDataSource):
    def __init__(self, file_path):
        self.file_path = file_path

    def load(self):
        evts = np.load(self.file_path)['evts']
        return evts, (None,None)

