from dataclasses import dataclass
import numpy as np


@dataclass
class BoundedStat :
    low: float
    median: float
    average: float
    high: float


@dataclass
class ConfMatrix :
    TP: int
    TN: int
    FP: int
    FN: int


@dataclass
class BoundedConfMatrix :
    TP: BoundedStat
    TN: BoundedStat
    FP: BoundedStat
    FN: BoundedStat


@dataclass
class Performance :
    cm : ConfMatrix
    accuracy: float
    sensitivity: float
    specificity: float
    PPV: float # aka Precision
    NPV: float
    F1: float


@dataclass
class BoundedPerformance :
    cm : BoundedConfMatrix
    accuracy: BoundedStat
    sensitivity: BoundedStat
    specificity: BoundedStat
    PPV: BoundedStat # aka Precision
    NPV: BoundedStat
    F1: BoundedStat


def array2BoundedStat(a,q=0.975) :
    a = np.float32(a)
    bs = BoundedStat(np.quantile(a,1-q),
                     np.quantile(a,0.5),
                     np.mean(a),
                     np.quantile(a,q))
    return bs

