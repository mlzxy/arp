from runstats import Statistics
from copy import copy

def item(v):
    if hasattr(v, 'item'):
        return v.item()
    else:
        return v

class StatisticsDict:
    def __init__(self) -> None:
        self._run_stats = {}
        self._loss_dict = {}
        self._stat_dict = {}
    
    def push(self, loss_dict):
        for k in loss_dict.keys():
            if 'loss' in k and k not in self._run_stats:
                self._run_stats[k] = Statistics()
        self._loss_dict = copy(loss_dict)

        self._stat_dict.clear()
        for k, _ in self._run_stats.items():
            if k in loss_dict:
                self._run_stats[k].push(item(loss_dict[k]))
            self._stat_dict[k] = self._run_stats[k].mean()
    
    @property
    def current(self):
        return self._loss_dict
    
    @property
    def running(self):
        return self._stat_dict
    
    def reset(self):
        self._run_stats.clear()
        self._loss_dict.clear()
        self._stat_dict.clear()
    