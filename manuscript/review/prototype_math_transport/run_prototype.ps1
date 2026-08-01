param(
    [string]$PandocPath = "C:\Users\rlewa\AppData\Local\Pandoc\pandoc.exe"
)

$ErrorActionPreference = "Stop"
Add-Type -AssemblyName System.IO.Compression.FileSystem
$prototypeDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
$resultsDirectory = Join-Path $prototypeDirectory "results"
New-Item -ItemType Directory -Force -Path $resultsDirectory | Out-Null

$variants = @(
    @{
        Name = "current-default"
        Source = Join-Path $prototypeDirectory "single_backslash.md"
        Reader = "markdown"
    },
    @{
        Name = "dollar-default"
        Source = Join-Path $prototypeDirectory "dollar_math.md"
        Reader = "markdown"
    },
    @{
        Name = "single-backslash-reader"
        Source = Join-Path $prototypeDirectory "single_backslash.md"
        Reader = "markdown+tex_math_single_backslash"
    }
)

$summaries = @()
$proportionalTo = [string][char]0x221D
$sigma = [string][char]0x03C3
$theta = [string][char]0x03B8
$gamma = [string][char]0x03B3
$minus = [string][char]0x2212
$lessThanOrEqual = [string][char]0x2264
$greaterThanOrEqual = [string][char]0x2265
$expectedMathText = @(
    "D(E)${proportionalTo}1/E",
    "I(x,y)=B+Aexp${minus}12u2${sigma}x2+v2${sigma}y2.",
    "(xc,yc)",
    "(${sigma}x,${sigma}y)",
    $theta,
    "mij(E)=[pi(E)+pj(E)]/2",
    "Pvis=[1${minus}sigmoid(o)][1${minus}sigmoid(d)],",
    "Pvis>0.5",
    "r${lessThanOrEqual}30",
    "eFB${lessThanOrEqual}8px",
    "${sigma}=5",
    "R2${greaterThanOrEqual}0.8",
    "${gamma}=8"
)

foreach ($variant in $variants) {
    $docxPath = Join-Path $resultsDirectory ($variant.Name + ".generated.docx")
    & $PandocPath $variant.Source --from=$($variant.Reader) --output=$docxPath
    if ($LASTEXITCODE -ne 0) {
        throw "Pandoc failed for $($variant.Name)."
    }

    $archive = [System.IO.Compression.ZipFile]::OpenRead($docxPath)
    try {
        $documentEntry = $archive.GetEntry("word/document.xml")
        $reader = [System.IO.StreamReader]::new($documentEntry.Open())
        try {
            $xmlText = $reader.ReadToEnd()
        }
        finally {
            $reader.Dispose()
        }
    }
    finally {
        $archive.Dispose()
    }

    [xml]$documentXml = $xmlText
    $namespaceManager = [System.Xml.XmlNamespaceManager]::new(
        $documentXml.NameTable
    )
    $namespaceManager.AddNamespace(
        "m",
        "http://schemas.openxmlformats.org/officeDocument/2006/math"
    )
    $mathNodes = $documentXml.SelectNodes("//m:oMath", $namespaceManager)
    $actualMathText = @($mathNodes | ForEach-Object {
        (($_.SelectNodes(".//m:t", $namespaceManager) |
            ForEach-Object { $_.'#text' }) -join "") -replace "\s", ""
    })
    $mathCount = $mathNodes.Count
    $formulaTextMatches = $actualMathText.Count -eq $expectedMathText.Count
    if ($formulaTextMatches) {
        for ($index = 0; $index -lt $expectedMathText.Count; $index++) {
            if ($actualMathText[$index] -cne $expectedMathText[$index]) {
                $formulaTextMatches = $false
                break
            }
        }
    }
    $symbolChecks = [ordered]@{
        formula_texts_exact = $formulaTextMatches
        radical_count = (
            $documentXml.SelectNodes("//m:rad", $namespaceManager).Count -eq 1
        )
        fraction_count = (
            $documentXml.SelectNodes("//m:f", $namespaceManager).Count -eq 3
        )
        subscript_count = (
            $documentXml.SelectNodes("//m:sSub", $namespaceManager).Count -eq 10
        )
        superscript_count = (
            $documentXml.SelectNodes("//m:sSup", $namespaceManager).Count -eq 3
        )
    }
    $allSymbolsPresent = -not ($symbolChecks.Values -contains $false)
    $summaries += [ordered]@{
        name = $variant.Name
        reader = $variant.Reader
        source = Split-Path -Leaf $variant.Source
        omath_count = $mathCount
        required_symbol_checks = $symbolChecks
        pass = ($mathCount -eq 13 -and $allSymbolsPresent)
    }
}

$summary = [ordered]@{
    expected_omath_count = 13
    required_math_checks = @(
        "formula_texts_exact",
        "radical_count",
        "fraction_count",
        "subscript_count",
        "superscript_count"
    )
    variants = $summaries
}
$summary | ConvertTo-Json -Depth 6 | Set-Content -Encoding utf8 -LiteralPath (
    Join-Path $resultsDirectory "summary.json"
)
$summary | ConvertTo-Json -Depth 6
