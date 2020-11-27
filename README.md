# profile_fiber_seg

This repository provides the code underlying our Publication ["Automated Segmentation of Profiled Fibers in cross-sectional Micrographs for Quality Control"](https://ieeexplore.ieee.org/abstract/document/9128511) presented at I2MTC2020


## Introduction & Installation

The code is written in `Python 3.8` using open source image processing libraries `open-cv` + `skimage` and was tested under `Ubuntu 18.04`.
All required dependencies can be installed with the `conda` environment as specified in the `environment.yml`.

To setup the environment, execute the following in a terminal:

    git clone https://github.com/ORippler/profiled_fiber_seg.git

    cd <path-to-cloned-repo>

    conda env create -f environment.yml


## Running Fiber Segmentation

To run the published Fiber Segmentation algorithms, place your images inside the `Input` folder and execute `python main.py`.

By default, the best performing setting `gc_20_EFD_50` will be executed, but you can further specify the configurations inside `settings_all.yml`,

## Citation and Contact

If you find our work useful, please consider citing our paper presented at I2MTC2020

```
@inproceedings{rippel2020automated,
  title={Automated Segmentation of Profiled Fibers in cross-sectional Micrographs for Quality Control},
  author={Rippel, Oliver and Schnabel, Maximilian and Paar, Georg-Philipp and Gries, Thomas and Merhof, Dorit},
  booktitle={2020 IEEE International Instrumentation and Measurement Technology Conference (I2MTC)},
  pages={1--6},
  year={2020},
  organization={IEEE}
}
```

If you wish to contact us, you can do so at rippel@lfb.rwth-aachen.de

## License

Copyright (C) 2020 by RWTH Aachen University
http://www.rwth-aachen.de

This software is dual-licensed under:
* Commercial license (please contact: lfb@lfb.rwth-aachen.de)
* AGPL (GNU Affero General Public License) open source license
