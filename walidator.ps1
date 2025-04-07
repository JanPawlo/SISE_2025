$JavaCmd = "C:\Users\qemil\Documents\narzedzia_dev\java8\bin\java.exe"
$JarPath = "C:\Users\qemil\Downloads\puzzleval.jar"
$SolFilenameRegex = '^[a-zA-Z0-9]+_[0-9]+_[0-9]+_[a-zA-Z]+_[a-zA-Z]+_sol.txt$'

$NumCorrectSols = 0
$NumIncorrectSols = 0
[System.Collections.ArrayList]$IncorrectSolFilenames = @()
Get-ChildItem -Path .\results -File | Where-Object { $_.Name -match $SolFilenameRegex } | ForEach-Object {
    $SplitFilename = $_.Name.Split('_');
    $InitFilename = $('{0}_{1}_{2}.txt' -f $SplitFilename[0], $SplitFilename[1], $SplitFilename[2])
    $InitFilePath = Join-Path .\puzzles $InitFilename
    Write-Host $('{0}: ' -f $_.Name) -NoNewline
    & $JavaCmd -jar $JarPath $InitFilename $_.Name
    if ($LastExitCode -eq 0) {
        $NumCorrectSols++
    } elseif ($LastExitCode -eq 1) {
        $NumIncorrectSols++;
        [void]$IncorrectSolFilenames.Add($_.Name)
    } else {
        Write-Host 'Fatal error.'
		$NumIncorrectSols++;
    }
}

Write-Host '----- Summary -----'
Write-Host $('Correct solutions: {0}' -f $NumCorrectSols) -ForegroundColor Green
Write-Host $('Incorrect solutions: {0}' -f $NumIncorrectSols) -ForegroundColor Red
foreach($Filename in $IncorrectSolFilenames) {
    Write-Host $Filename -ForegroundColor Red
}