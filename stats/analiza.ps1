# Regex do weryfikacji nazw plików
$StatsFilenameRegex = '^[a-zA-Z0-9]+_(\d+)_\d+_([a-zA-Z]+)_([a-zA-Z]+)_stats\.txt$'

# Inicjalizacja struktur danych
$strategie = @("astr", "bfs", "dfs")
$results = @{}
$allFileData = @()

foreach ($strat in $strategie) {
    $results[$strat] = 1..7 | ForEach-Object {
        [PSCustomObject]@{
            Depth = $_
            AverageTime = 0.0
            AverageSolutionLength = 0
            AverageVisitedStates = 0
            AverageProcessedStates = 0
            AverageMaxDepth = 0
            SolutionCount = 0              # Tylko przypadki z rozwiązaniem (długość != -1)
            CaseCount = 0                  # Wszystkie przypadki
        }
    }
}

# Przetwarzanie plików
Get-ChildItem -File | Where-Object { $_.Name -match $StatsFilenameRegex } | ForEach-Object {
    $depth = [int]($_.Name -replace $StatsFilenameRegex, '$1')
    $strategy = ($_.Name -replace $StatsFilenameRegex, '$2')
    $param = ($_.Name -replace $StatsFilenameRegex, '$3')

    if ($strategie -contains $strategy -and $depth -ge 1 -and $depth -le 7) {
        try {
            $content = Get-Content $_.FullName -Raw
            $lines = $content -split "`r?`n" | Where-Object { $_ -match '\S' }

            if ($lines.Count -ge 5) {
                $solutionLength = [int]$lines[0]
                $visitedStates = [int]$lines[1]
                $processedStates = [int]$lines[2]
                $maxDepth = [int]$lines[3]
                $executionTime = [double]$lines[4]

                # Zapisz pełne dane pliku
                $fileData = [PSCustomObject]@{
                    File = $_.Name
                    Strategy = $strategy
                    Parameter = $param
                    Depth = $depth
                    SolutionLength = $solutionLength
                    VisitedStates = $visitedStates
                    ProcessedStates = $processedStates
                    MaxDepth = $maxDepth
                    ExecutionTimeMs = $executionTime
                    HasSolution = ($solutionLength -ne -1)
                }
                $allFileData += $fileData

                # Aktualizuj statystyki
                $current = $results[$strategy] | Where-Object { $_.Depth -eq $depth }
                if ($current) {
                    $current.CaseCount++

                    if ($solutionLength -ne -1) {
                        $current.SolutionCount++
                        $current.AverageSolutionLength += $solutionLength
                        $current.AverageTime += $executionTime
                        $current.AverageVisitedStates += $visitedStates
                        $current.AverageProcessedStates += $processedStates
                        $current.AverageMaxDepth += $maxDepth
                    }
                }
            }
        } catch {
            Write-Warning "Błąd przetwarzania pliku $($_.Name): $_"
        }
    }
}

# Oblicz finalne średnie
foreach ($strat in $strategie) {
    foreach ($depthStats in $results[$strat]) {
        if ($depthStats.SolutionCount -gt 0) {
            $depthStats.AverageSolutionLength = [math]::Round($depthStats.AverageSolutionLength / $depthStats.SolutionCount, 2)
            $depthStats.AverageTime = [math]::Round($depthStats.AverageTime / $depthStats.SolutionCount, 3)
            $depthStats.AverageVisitedStates = [math]::Round($depthStats.AverageVisitedStates / $depthStats.SolutionCount, 2)
            $depthStats.AverageProcessedStates = [math]::Round($depthStats.AverageProcessedStates / $depthStats.SolutionCount, 2)
            $depthStats.AverageMaxDepth = [math]::Round($depthStats.AverageMaxDepth / $depthStats.SolutionCount, 2)
        }
    }
}

# Eksport wyników
foreach ($strat in $strategie) {
    $filename = "${strat}_statystyki.csv"
    $results[$strat] | Export-Csv -Path $filename -NoTypeInformation -Encoding UTF8
    Write-Host "Zapisano statystyki dla $strat do $filename" -ForegroundColor Green
}

# Znajdź najwolniejsze przypadki z rozwiązaniem
$slowestWithSolution = $allFileData |
    Where-Object { $_.HasSolution } |
    Sort-Object ExecutionTimeMs -Descending |
    Select-Object -First 5

Write-Host "`nTOP 5 NAJWOLNIEJSZYCH PRZYPADKÓW Z ROZWIĄZANIEM:" -ForegroundColor Cyan
$slowestWithSolution | Format-Table @(
    @{Label="Plik"; Expression={$_.File}},
    @{Label="Strategia"; Expression={$_.Strategy}},
    @{Label="Parametr"; Expression={$_.Parameter}},
    @{Label="Głębokość"; Expression={$_.Depth}},
    @{Label="Czas (ms)"; Expression={[math]::Round($_.ExecutionTimeMs, 3)}},
    @{Label="Długość rozwiązania"; Expression={$_.SolutionLength}},
    @{Label="Stany odwiedzone"; Expression={$_.VisitedStates}},
    @{Label="Stany przetworzone"; Expression={$_.ProcessedStates}}
) -AutoSize

# Podsumowanie statystyk
foreach ($strat in $strategie) {
    Write-Host "`nPODSUMOWANIE STRATEGII $($strat.ToUpper()):" -ForegroundColor Yellow
    Write-Host "Gł.| Śr. czas (ms) | Dł. roz. | Stany odw. | Stany prz. | Maks gł. | Rozw./Przyp."
    Write-Host "---|---------------|---------|-----------|-----------|---------|------------"

    foreach ($depth in 1..7) {
        $stats = $results[$strat] | Where-Object { $_.Depth -eq $depth }
        if ($stats.CaseCount -gt 0) {
            $successRate = [math]::Round(($stats.SolutionCount / $stats.CaseCount) * 100, 1)
            Write-Host ("{0,2} | {1,13} | {2,7} | {3,9} | {4,9} | {5,7} | {6,3}% ({7}/{8})" -f
                $depth,
                $stats.AverageTime,
                $stats.AverageSolutionLength,
                $stats.AverageVisitedStates,
                $stats.AverageProcessedStates,
                $stats.AverageMaxDepth,
                $successRate,
                $stats.SolutionCount,
                $stats.CaseCount)
        }
    }
}

# Dodatkowa analiza dla BFS
if ($results.ContainsKey('bfs')) {
    $bfsStats = $allFileData | Where-Object { $_.Strategy -eq 'bfs' -and $_.HasSolution }

    Write-Host "`nDODATKOWA ANALIZA DLA BFS:" -ForegroundColor Magenta
    $bfsStats | Group-Object Depth | ForEach-Object {
        $times = $_.Group.ExecutionTimeMs | Measure-Object -Average -Minimum -Maximum
        Write-Host "Głębokość $($_.Name): Śr. czas = $([math]::Round($times.Average, 1)) ms (min: $($times.Minimum), max: $($times.Maximum))"
    }
}
