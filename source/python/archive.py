
import math
import pickle
from dataclasses import dataclass

def saveHistory(filename,H) :
    with open(filename, 'wb') as file_pi:
            pickle.dump(H.history, file_pi)
    
    history_dict = H.history
    print(history_dict.keys())


@dataclass
class History :
    history: dict


def loadHistory(filename) :
    H = None
    with open(filename,'rb') as file_pi:
        Hdata = pickle.load(file_pi)
        H = History(Hdata)
    return H

