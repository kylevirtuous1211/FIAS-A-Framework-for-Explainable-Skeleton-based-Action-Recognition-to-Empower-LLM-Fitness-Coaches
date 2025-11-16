## Goal
In the folder, contains 2 scripts `json2pkl_2D.py` and `json2pkl_3D.py`. These are the basic scripts to generate the annotation files.

## TODO
To generate, you have to modify 

* `--json_root`: The path to your json files. (IMPORTANT: the script assumes your data is like our dataset, has coarse_label / fine-grain_label)
* `--out_pkl`: Output path for the .pkl file
```
    parser.add_argument('--json_root', default='data/json_rtmpose_all_view_3D', help='Root directory of JSON files (e.g., data/json)')
    
    parser.add_argument('--out_pkl', default='data/pickle/rtmpose_3D_all.pkl', help='Output path for the .pkl file (e.g., data/custom_data.pkl)')
``` 