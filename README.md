# boundary_issues

## Setting up environment  

In boundary_issues/ run

```bash
source setup.sh
```

##### Dataset Information
* Dominik's data
    * Location: /mnt/efs/aimbl_2025/student_data/S-DR  
22 .zarr files of structure  
```
/mnt/efs/aimbl_2025/student_data/S-DR/puncta_x100_1_2.zarr [zgroup]
 - version: 0.4
 - metadata
   - Multiscales
   - OMERO
 - data
   - (3, 268, 2304, 2304)
/mnt/efs/aimbl_2025/student_data/S-DR/puncta_x100_1_2.zarr/labels [zgroup] (hidden)
 - version: 
 - metadata
   - Labels
 - data
/mnt/efs/aimbl_2025/student_data/S-DR/puncta_x100_1_2.zarr/labels/mask [zgroup] (hidden)
 - version: 0.4
 - metadata
   - Label
   - Multiscales
 - data
   - (268, 2304, 2304)
```
    ch0 (488) - NMIIC  
    ch1 (569) - Phalloidin (F-actin)  
    ch2 (647) - ZO-1 ab  

* Mady's Data 
    * Location: /mnt/efs/aimbl_2025/student_data/S-MC
    * 20 z-stack files of interest for each time point (4dpf_zarrconversion folder is priority)
```
  "multiscales" : [ {
    "metadata" : {
      "method" : "loci.common.image.SimpleImageScaler",
      "version" : "Bio-Formats 7.3.1"
    },
    "axes" : [ {
      "name" : "t",
      "type" : "time"
    }, {
      "name" : "c",
      "type" : "channel"
    }, {
      "unit" : "micrometer",
      "name" : "z",
      "type" : "space"
    }, {
      "unit" : "micrometer",
      "name" : "y",
      "type" : "space"
    }, {
      "unit" : "micrometer",
      "name" : "x",
      "type" : "space"
    } ],
    "name" : "250627_93a_4dpf_nm4_em1_001_WellA01_ChannelX_Seq0000.nd2",
    "datasets" : [ {
      "path" : "0",
      "coordinateTransformations" : [ {
        "scale" : [ 1.0, 1.0, 0.3, 0.108333333333333, 0.108333333333333 ],
        "type" : "scale"

  "chunks" : [ 1, 1, 1, 1024, 1024 ],
  "compressor" : {
    "clevel" : 5,
    "blocksize" : 0,
    "shuffle" : 1,
    "cname" : "lz4",
    "id" : "blosc"
  },
  "dtype" : ">u2",
  "fill_value" : 0,
  "filters" : null,
  "order" : "C",
  "shape" : [ 1, 3, 110, 2048, 2048 ],
  "dimension_separator" : "/",
  "zarr_format" : 2
```