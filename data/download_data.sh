OPTIONS="-q --show-progress"
DARUS=https://darus.uni-stuttgart.de/api/access/datafile
wget $OPTIONS $DARUS/478872 -O 2d_microstructures.h5
wget $OPTIONS $DARUS/478873 -O 3d_microstructures.h5
wget $OPTIONS $DARUS/478877 -O weights_uno_mechanical_2d_dir.pt
wget $OPTIONS $DARUS/478889 -O weights_uno_mechanical_2d_mixed.pt
wget $OPTIONS $DARUS/478876 -O weights_uno_mechanical_2d_per.pt
wget $OPTIONS $DARUS/478878 -O weights_uno_mechanical_3d_dir.pt
wget $OPTIONS $DARUS/478883 -O weights_uno_mechanical_3d_mixed.pt
wget $OPTIONS $DARUS/478880 -O weights_uno_mechanical_3d_per.pt
wget $OPTIONS $DARUS/478875 -O weights_uno_naive_thermal_2d_per.pt
wget $OPTIONS $DARUS/478874 -O weights_uno_thermal_2d_per.pt
wget $OPTIONS $DARUS/478881 -O weights_uno_thermal_3d_dir.pt
wget $OPTIONS $DARUS/478882 -O weights_uno_thermal_3d_per.pt
