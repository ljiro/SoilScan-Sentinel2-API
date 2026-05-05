# Renames soilgrids tiles to expected names and zips them for upload.
# Run from anywhere — edit $SrcRoot if your path differs.

$SrcRoot  = "D:\Github\SoilScan-Sentinel2\data\raw\soilgrids"
$OutZip   = "D:\Github\SoilScan-Sentinel2-API\scripts\soilgrids.zip"
$TmpDir   = "$env:TEMP\soilgrids_upload"

# property -> depths that have a tile
$Props = @{
    phh2o    = @("0-5cm", "5-15cm")
    soc      = @("0-5cm", "5-15cm")
    nitrogen = @("0-5cm", "5-15cm")
    clay     = @("0-5cm", "5-15cm")
    sand     = @("0-5cm", "5-15cm")
    cec      = @("0-5cm", "5-15cm")
}

if (Test-Path $TmpDir) { Remove-Item $TmpDir -Recurse -Force }
if (Test-Path $OutZip) { Remove-Item $OutZip -Force }

foreach ($prop in $Props.Keys) {
    foreach ($depth in $Props[$prop]) {
        $tile = Get-ChildItem "$SrcRoot\$prop\${prop}_${depth}_mean" -Filter "*.tif" -Recurse | Select-Object -First 1
        if (-not $tile) {
            Write-Warning "Missing tile: $prop $depth"
            continue
        }
        $destDir = "$TmpDir\$prop"
        New-Item -ItemType Directory -Path $destDir -Force | Out-Null
        $destFile = "$destDir\${prop}_${depth}_mean.tif"
        Copy-Item $tile.FullName $destFile
        Write-Host "  $prop/$depth  ->  $destFile"
    }
}

Compress-Archive -Path "$TmpDir\*" -DestinationPath $OutZip
Write-Host ""
Write-Host "Done. Zip saved to: $OutZip"
Write-Host "Size: $([math]::Round((Get-Item $OutZip).Length / 1MB, 2)) MB"
