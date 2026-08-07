// Predeclared ticket-03 execution; do not edit after native outcomes exist.
scanId = "scan_b";
frameDir = "C:/Users/rlewa/Documents/PROJEKTY/trackleed/data/26a21b_ScanE";
outputDir = "C:/Users/rlewa/Documents/PROJEKTY/trackleed/.scratch/trackleed-cross-contract-manuscript/principal-viperleed-execution/scan_b/native";
patternPath = "C:/Users/rlewa/Documents/PROJEKTY/trackleed/manuscript/cross_contract/viperleed_pilot/frozen/pattern.csv";
screenfitPath = "C:/Users/rlewa/Documents/PROJEKTY/trackleed/manuscript/cross_contract/viperleed_pilot/frozen/registration_screenfit.csv";
checkpointPath = outputDir + "/macro_checkpoint.txt";
File.saveString("macro started\nconfiguration predeclared; no manual outcome annotation\n", checkpointPath);

run("Image Sequence...", "open=[" + frameDir + "] sort");
rename("principal_scan_b");
File.append("46-frame stack opened\n", checkpointPath);

newImage("principal_mask", "8-bit black", 1000, 1000, 1);
setColor(255);
makeOval(21.000308, 2.011221, 940, 940);
fill();
setColor(0);
makeOval(366.000308, 347.011221, 250, 250);
fill();
run("Select None");
run("Invert LUT");
File.append("frozen annular mask built\n", checkpointPath);

energies = newArray(
    40, 43, 47.1, 51.2, 55.4, 59.5, 63.6, 67.8,
    71.9, 76, 80.1, 84.2, 88.4, 92.5, 96.6, 100.8,
    104.9, 109, 113.5, 117.6, 121.8, 126.2, 130.4, 134.5,
    138.6, 142.8, 146.9, 151, 155.5, 159.6, 163.8, 167.9,
    172, 176.1, 180.2, 184.4, 188.5, 192.2, 196.4, 200.5,
    204.6, 208.8, 212.5, 216.6, 220.4, 220.4
);
fitText = replace(File.openAsString(screenfitPath), "\n", "");
fitItems = split(fitText, ",");
indices = newArray(fitItems.length);
for (i = 0; i < fitItems.length; i++)
    indices[i] = parseFloat(fitItems[i]);

call("LEED_Spot_Tracker.startMacro");
File.append("tracker panel started\n", checkpointPath);
Ext.LEEDsetArray("energy", energies);
Ext.LEEDsetInput("iv", "principal_scan_b");
Ext.LEEDsetInput("mask", "principal_mask");
File.append("stack, energy and mask attached\n", checkpointPath);
Ext.LEEDsetValue("minRange_eV", 0);
Ext.LEEDsetValue("savePlots", 0);
Ext.LEEDsetValue("saveStack", 0);
Ext.LEEDsetValue("neighborBackground", 0);
Ext.LEEDsetValue("minSignificanceSelecting", 4);
Ext.LEEDsetPattern(patternPath);
Ext.LEEDselectSliceForE(188.5);
File.append("pattern attached; transferring frozen indices\n", checkpointPath);
Ext.LEEDsetIndices(indices);
File.append("indices set once; no per-frame correction permitted\n", checkpointPath);
print("Principal scan_b pre-track status:\n" + Ext.LEEDgetStatus("all"));
Ext.LEEDtrackSpots();
File.append("spot tracking completed\n", checkpointPath);
print("Principal scan_b post-track status:\n" + Ext.LEEDgetStatus("all"));
Ext.LEEDsaveData(outputDir, "viperleed_scan_b");
File.append("native whole-pattern data saved before proposal reduction\n", checkpointPath);
if (!File.exists(outputDir + "/viperleed_scan_b_x.csv"))
    exit("Principal scan_b failed: native x export is missing");
File.append("macro completed\n", checkpointPath);
Ext.LEEDclose("tracker stack");
close("*");
run("Quit");
