from __future__ import absolute_import

import os
from got10k.experiments import *

from siamfc import TrackerSiamFC


if __name__ == '__main__':
    net_path = 'pretrained/siamfc_alexnet_e50.pth'
    tracker = TrackerSiamFC(net_path=net_path)

    root_dir = os.path.expanduser('/home/papillon/datasets/OTB100')
    e = ExperimentOTB(root_dir, version=2015)
    e.run(tracker,visualize=False)
    e.report([tracker.name])
