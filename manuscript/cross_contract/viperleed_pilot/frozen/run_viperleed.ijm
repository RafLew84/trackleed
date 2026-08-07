// Frozen 15-frame ViPErLEED run for 26a24_ScanE, frames 54-68.
artifactDir = "C:/Users/rlewa/Documents/PROJEKTY/trackleed/.scratch/trackleed-physics-method-paper/artifacts/domain-baseline-high-energy";
pilotDir = "C:/Users/rlewa/Documents/PROJEKTY/trackleed/.scratch/trackleed-physics-method-paper/artifacts/domain-baseline-pilot";
frameDir = artifactDir + "/frames";
outputDir = artifactDir + "/viperleed";
patternPath = pilotDir + "/viperleed/ag111_hexagonal_pattern.csv";
screenfitPath = pilotDir + "/viperleed/ag111_registration_screenfit.csv";
checkpointPath = outputDir + "/macro_checkpoint.txt";
File.saveString("macro started\n", checkpointPath);

run("Image Sequence...", "open=[" + frameDir + "] sort");
rename("scane_high_energy");
File.append("15-frame stack opened\n", checkpointPath);

newImage("high_energy_mask", "8-bit black", 1000, 1000, 1);
setColor(255);
makeOval(21.000308, 2.011221, 940, 940);
fill();
setColor(0);
makeOval(366.000308, 347.011221, 250, 250);
fill();
run("Select None");
run("Invert LUT");
File.append("annular mask built\n", checkpointPath);

energies = newArray(
    187.9, 190.6, 193.3, 195.9, 198.6,
    201.3, 204.0, 206.6, 209.3, 212.0,
    214.7, 217.1, 219.8, 220.2, 220.2
);
fitText = replace(File.openAsString(screenfitPath), "\n", "");
fitItems = split(fitText, ",");
indices = newArray(fitItems.length);
for (i = 0; i < fitItems.length; i++)
    indices[i] = parseFloat(fitItems[i]);

call("LEED_Spot_Tracker.startMacro");
File.append("tracker panel started\n", checkpointPath);
Ext.LEEDsetArray("energy", energies);
Ext.LEEDsetInput("iv", "scane_high_energy");
Ext.LEEDsetInput("mask", "high_energy_mask");
File.append("stack, energy and mask attached\n", checkpointPath);
Ext.LEEDsetValue("minRange_eV", 0);
Ext.LEEDsetValue("savePlots", 0);
Ext.LEEDsetValue("saveStack", 0);
Ext.LEEDsetValue("neighborBackground", 0);
Ext.LEEDsetValue("minSignificanceSelecting", 4);
Ext.LEEDsetPattern(patternPath);
Ext.LEEDselectSliceForE(187.9);
File.append("pattern attached; setting indices\n", checkpointPath);
Ext.LEEDsetIndices(indices);
File.append("indices set\n", checkpointPath);
print("ViPErLEED pre-track status:\n" + Ext.LEEDgetStatus("all"));
Ext.LEEDtrackSpots();
File.append("spot tracking completed\n", checkpointPath);
print("ViPErLEED post-track status:\n" + Ext.LEEDgetStatus("all"));
Ext.LEEDsaveData(outputDir, "viperleed_high_energy");
File.append("native data saved\n", checkpointPath);
if (!File.exists(outputDir + "/viperleed_high_energy_x.csv"))
    exit("ViPErLEED high-energy run failed: native x export is missing");
File.append("macro completed\n", checkpointPath);
Ext.LEEDclose("tracker stack");
close("*");
run("Quit");
