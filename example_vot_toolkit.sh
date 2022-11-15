mkdir vot
cd vot
mkdir vot2022stb
vot initialize vot2022/stb --workspace vot2022stb
cp trackers/ostrack/trackers.ini ./
'change path of trackers.ini'
vot test ostrackNeighbor
vot evaluate --workspace vot2022stb ostrackNeighbor
vot analysis --workspace vot2022stb ostrackNeighbor #--output json
