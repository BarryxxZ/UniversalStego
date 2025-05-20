# --------------------------------------------------------
# Universal Low Bit-Rate Speech Steganalysis
# Licensed under The MIT License
# Code written by Yiqin Qiu
# --------------------------------------------------------

from dataclasses import dataclass


@dataclass
class TrainConfigs:
    """
    The class used to store the training folders
    Applied in the universal training and testing case
    """
    def __init__(self, args):
        super().__init__()
        # Original samples
        self.FOLDERS = [
                    {"class": 0,
                     "folder": "/home/barryxxz/audiodata/AMR_NB/FCB/Geiser/Universe/TXT/Chinese/{}s/0".format(args.length)},
                    {"class": 0,
                     "folder": "/home/barryxxz/audiodata/AMR_NB/FCB/Geiser/Universe/TXT/English/{}s/0".format(args.length)},
                    {"class": 0,
                     "folder": "/home/barryxxz/audiodata/AMR_NB/LPC/CNV/Universe/TXT/Chinese/{}s/0".format(args.length)},
                    {"class": 0,
                     "folder": "/home/barryxxz/audiodata/AMR_NB/LPC/CNV/Universe/TXT/English/{}s/0".format(args.length)},
                    {"class": 0,
                     "folder": "/home/barryxxz/audiodata/AMR_NB/ACB/Huang/Universe/TXT/Chinese/{}s/0".format(args.length)},
                    {"class": 0,
                     "folder": "/home/barryxxz/audiodata/AMR_NB/ACB/Huang/Universe/TXT/English/{}s/0".format(args.length)},
                    {"class": 1,
                     "folder": "/home/barryxxz/audiodata/AMR_NB/FCB/Geiser/Universe/TXT/Chinese/{}s/{}".format(args.length, args.em_rate)},
                    {"class": 1,
                     "folder": "/home/barryxxz/audiodata/AMR_NB/FCB/Geiser/Universe/TXT/English/{}s/{}".format(args.length, args.em_rate)},
                    {"class": 1,
                     "folder": "/home/barryxxz/audiodata/AMR_NB/LPC/CNV/Universe/TXT/Chinese/{}s/{}".format(args.length, args.em_rate)},
                    {"class": 1,
                     "folder": "/home/barryxxz/audiodata/AMR_NB/LPC/CNV/Universe/TXT/English/{}s/{}".format(args.length, args.em_rate)},
                    {"class": 1,
                     "folder": "/home/barryxxz/audiodata/AMR_NB/ACB/Huang/Universe/TXT/Chinese/{}s/{}".format(args.length, args.em_rate)},
                    {"class": 1,
                     "folder": "/home/barryxxz/audiodata/AMR_NB/ACB/Huang/Universe/TXT/English/{}s/{}".format(args.length, args.em_rate)}
        ]
        # Recompressed samples
        self.RE_FOLDERS = [
            {"class": 0,
             "folder": "/home/barryxxz/audiodata/AMR_NB/FCB/Geiser/Universe/ReCompress/TXT/Chinese/{}s/0".format(args.length)},
            {"class": 0,
             "folder": "/home/barryxxz/audiodata/AMR_NB/FCB/Geiser/Universe/ReCompress/TXT/English/{}s/0".format(args.length)},
            {"class": 0,
             "folder": "/home/barryxxz/audiodata/AMR_NB/LPC/CNV/Universe/ReCompress/TXT/Chinese/{}s/0".format(args.length)},
            {"class": 0,
             "folder": "/home/barryxxz/audiodata/AMR_NB/LPC/CNV/Universe/ReCompress/TXT/English/{}s/0".format(args.length)},
            {"class": 0,
             "folder": "/home/barryxxz/audiodata/AMR_NB/ACB/Huang/Universe/ReCompress/TXT/Chinese/{}s/0".format(args.length)},
            {"class": 0,
             "folder": "/home/barryxxz/audiodata/AMR_NB/ACB/Huang/Universe/ReCompress/TXT/English/{}s/0".format(args.length)},
            {"class": 1,
             "folder": "/home/barryxxz/audiodata/AMR_NB/FCB/Geiser/Universe/ReCompress/TXT/Chinese/{}s/{}".format(args.length, args.em_rate)},
            {"class": 1,
             "folder": "/home/barryxxz/audiodata/AMR_NB/FCB/Geiser/Universe/ReCompress/TXT/English/{}s/{}".format(args.length, args.em_rate)},
            {"class": 1,
             "folder": "/home/barryxxz/audiodata/AMR_NB/LPC/CNV/Universe/ReCompress/TXT/Chinese/{}s/{}".format(args.length, args.em_rate)},
            {"class": 1,
             "folder": "/home/barryxxz/audiodata/AMR_NB/LPC/CNV/Universe/ReCompress/TXT/English/{}s/{}".format(args.length, args.em_rate)},
            {"class": 1,
             "folder": "/home/barryxxz/audiodata/AMR_NB/ACB/Huang/Universe/ReCompress/TXT/Chinese/{}s/{}".format(args.length, args.em_rate)},
            {"class": 1,
             "folder": "/home/barryxxz/audiodata/AMR_NB/ACB/Huang/Universe/ReCompress/TXT/English/{}s/{}".format(args.length, args.em_rate)}
        ]


@dataclass
class TestConfigs:
    """
    The class used to store the test folders
    Only applied in the specific test case
    """
    def __init__(self, args):
        super().__init__()
        mode_list = ['all', 'FCB/Geiser', 'LPC/CNV', 'ACB/Huang']
        mode = mode_list[args.test_mode]
        # Original samples
        self.FOLDERS = [
            {"class": 0,
             "folder": "/home/barryxxz/audiodata/AMR_NB/{}/Universe/TXT/Chinese/{}s/0".format(mode, args.length)},
            {"class": 0,
             "folder": "/home/barryxxz/audiodata/AMR_NB/{}/Universe/TXT/English/{}s/0".format(mode, args.length)},
            {"class": 1,
             "folder": "/home/barryxxz/audiodata/AMR_NB/{}/Universe/TXT/Chinese/{}s/{}".format(mode, args.length, args.em_rate)},
            {"class": 1,
             "folder": "/home/barryxxz/audiodata/AMR_NB/{}/Universe/TXT/English/{}s/{}".format(mode, args.length, args.em_rate)}
        ]
        # Recompressed samples
        self.RE_FOLDERS = [
            {"class": 0,
             "folder": "/home/barryxxz/audiodata/AMR_NB/{}/Universe/ReCompress/TXT/Chinese/{}s/0".format(mode, args.length)},
            {"class": 0,
             "folder": "/home/barryxxz/audiodata/AMR_NB/{}/Universe/ReCompress/TXT/English/{}s/0".format(mode, args.length)},
            {"class": 1,
             "folder": "/home/barryxxz/audiodata/AMR_NB/{}/Universe/ReCompress/TXT/Chinese/{}s/{}".format(mode, args.length, args.em_rate)},
            {"class": 1,
             "folder": "/home/barryxxz/audiodata/AMR_NB/{}/Universe/ReCompress/TXT/English/{}s/{}".format(mode, args.length, args.em_rate)}
        ]
