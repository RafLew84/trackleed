// Predeclared ticket-03 execution; do not edit after native outcomes exist.
scanId = "scan_a";
frameDir = "C:/Users/rlewa/Documents/PROJEKTY/trackleed/data/26a21_ScanE";
outputDir = "C:/Users/rlewa/Documents/PROJEKTY/trackleed/.scratch/trackleed-cross-contract-manuscript/principal-viperleed-execution/scan_a/native";
patternPath = "C:/Users/rlewa/Documents/PROJEKTY/trackleed/manuscript/cross_contract/viperleed_pilot/frozen/pattern.csv";
screenfitPath = "C:/Users/rlewa/Documents/PROJEKTY/trackleed/manuscript/cross_contract/viperleed_principal/execution/frozen_pilot_screenfit.csv";
checkpointPath = outputDir + "/macro_checkpoint.txt";
File.saveString("macro started\nconfiguration predeclared; no manual outcome annotation\n", checkpointPath);

run("Image Sequence...", "open=[" + frameDir + "] sort");
rename("principal_scan_a");
File.append("35-frame stack opened\n", checkpointPath);

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
    40, 44, 49.5, 55, 60.5, 66, 71.5, 77,
    82.5, 88, 93.5, 99, 104.5, 110, 115.5, 121.5,
    127, 133, 138.5, 144, 149.5, 155, 161, 166.5,
    172, 177.5, 183, 188.5, 193.5, 199, 204.5, 210,
    215, 220.5, 220.5
);
fitText = replace(File.openAsString(screenfitPath), "\n", "");
fitItems = split(fitText, ",");
indices = newArray(fitItems.length);
for (i = 0; i < fitItems.length; i++)
    indices[i] = parseFloat(fitItems[i]);

call("LEED_Spot_Tracker.startMacro");
File.append("tracker panel started\n", checkpointPath);
Ext.LEEDsetArray("energy", energies);
Ext.LEEDsetInput("iv", "principal_scan_a");
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
print("Principal scan_a pre-track status:\n" + Ext.LEEDgetStatus("all"));
Ext.LEEDtrackSpots();
File.append("spot tracking completed\n", checkpointPath);
print("Principal scan_a post-track status:\n" + Ext.LEEDgetStatus("all"));
Ext.LEEDsaveData(outputDir, "viperleed_scan_a");
File.append("native whole-pattern data saved before proposal reduction\n", checkpointPath);
if (!File.exists(outputDir + "/viperleed_scan_a_x.csv"))
    exit("Principal scan_a failed: native x export is missing");
File.append("macro completed\n", checkpointPath);
Ext.LEEDclose("tracker stack");
close("*");
run("Quit");
