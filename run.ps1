param([string]$strategy, [string]$param, [string]$inputFile, [string]$outputSol, [string]$outputStats)

# Ścieżka do głównego programu
$Progcmd = 'python main.py'

# Walidacja argumentów
if (-not $strategy -or -not $param -or -not $inputFile -or -not $outputSol -or -not $outputStats) {
    Write-Error "Wszystkie argumenty muszą zostać przekazane."
    exit 1
}

# Budowanie polecenia do uruchomienia programu
$command = "$Progcmd $strategy $param $inputFile $outputSol $outputStats"

# Wykonanie polecenia
Invoke-Expression $command
