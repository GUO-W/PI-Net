## PI-Net: Pose Interacting Network for Multi-Person Monocular 3D Pose Estimation

This is the code for the paper 
Wen Guo, Enric Corona, Francesc Moreno-Noguer, Xavier Alameda-Pineda,
[PI-Net: Pose Interacting Network for Multi-Person Monocular 3D Pose Estimation](https://openaccess.thecvf.com/content/WACV2021/papers/Guo_PI-Net_Pose_Interacting_Network_for_Multi-Person_Monocular_3D_Pose_Estimation_WACV_2021_paper.pdf),
in WACV2021.

### Dependencies
Our code is tested on CUDA9, Python3.6, Pytorch1.3.0 
MATLAB is needed for evaluating the 3DPCK errors.

### Directory
```
ROOT
|-- data
|-- model
|-- utils 
`-- output
    |-- log
    |-- result 
    |-- snapshot 
        `-- snapshot_24.pth.tar
    |-- tensorboard_log 
    `-- vis 
```

```shell script
data
|-- MuCo
    |-- MuCo.py
    `-- data
        |-- augmented_set
        |-- annotations
            |-- MuCo-3DHP_with_posenent_result_filter.json
            |-- MuCo_id2pairId.json
            `-- split_gt.py
|--MuPoTS_skeleton
    |-- MuPoTS_skeleton.py
    |-- bbox_root
    `-- data
        |-- MultiPersonTestSet
        |-- eval
        |-- MuPoTS-3D_with_posenent_result.json
        |-- MuPoTS-3D_id2pairId.json 
        `-- split_gt.py  
```


### Preparing data
* Download Training and testing data MuCo and MuPoTS from [SingleShot](http://gvv.mpi-inf.mpg.de/projects/SingleShotMultiPerson/)
or from [3DMPPE](https://github.com/mks0601/3DMPPE_POSENET_RELEASE.git). 
* Run baseline model [3DMPPE](https://github.com/mks0601/3DMPPE_POSENET_RELEASE.git) to get pririor poses, and save the result in MuCo-3DHP_with_posenent_result_filter.json
and MuPoTS-3D_id2pairId.json. (To save the result, please refer to evaluation code in data/MuCo/MuCo.py). If you want to work on another baseline, just save the results in the same format.
* Get the ids of related instances by split_gt.py, and save the result in MuCo_id2pairId.json and MuPoTS-3D_id2pairId.json
* Easy start: MuCo-3DHP_with_posenent_result_filter.json, MuCo_id2pairId.json, MuPoTS-3D_with_posenent_result.json, MuPoTS-3D_id2pairId.json,
and our pretrained model snapshot_24.pth.tar could be downloaded [here](https://drive.google.com/drive/folders/1y99pX4uGVnOemL8G24RetlNesB23-7kH?usp=sharing).


### Training and testing
* To run train the model and test on MPJPE, please just uncommand the corresponding line in model/run.sh and run it directly.
* To evaluate the result by 3DPCK, please use the --save_mat_result option to get the 2D and 3D result in .m, and 
use the evaluation code in [SingleShot](http://gvv.mpi-inf.mpg.de/projects/SingleShotMultiPerson/). You can follow the instructions in 
[3DMPPE](https://github.com/mks0601/3DMPPE_POSENET_RELEASE.git) to download and set up the evaluation codes.

### Citing
If you use our code, please cite our work
@inproceedings{guo2021pi,
    title={PI-Net: Pose Interacting Network for Multi-Person Monocular 3D Pose Estimation},
    author={Guo, Wen and Corona, Enric and Moreno-Noguer, Francesc and Alameda-Pineda, Xavier},
    booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
    pages={2796--2806},
    year={2021}
}


### Acknowledgments
The overall code framework is adapted from [3DMPPE](https://github.com/mks0601/3DMPPE_POSENET_RELEASE.git) and
[Torchseg](https://github.com/ycszen/TorchSeg.git).
The predictor model code is adapted from [SeeWoLook](https://github.com/LourencoVazPato/seeing-without-looking.git).


### Licence
MIT


