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
*   